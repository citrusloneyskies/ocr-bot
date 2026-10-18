"""Region recognition with self-checking.

For every text region the recogniser:

1. crops it (masking neighbouring regions so text is never read twice),
2. straightens rotated marginal notes,
3. finds the text lines and cuts tall regions *between* lines into chunks,
4. scales each chunk so lines have a consistent height, pads it with white,
5. reads the chunks with the OCR model (batched),
6. checks every result — line count against the image, runaway loops,
   token-budget overruns — and re-reads failures in smaller pieces,
7. records line boxes so the searchable PDF can place the text exactly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from . import geometry as G
from . import imaging
from .document import FOOTER, FOOTNOTE, HEADER, MARGIN, PAGE_NUMBER, TABLE, TEXT_KINDS, Line, Region
from .text import clean_model_output, html_to_text, strip_markdown

log = logging.getLogger(__name__)

ProgressFn = Callable[[dict], None]


@dataclass
class Chunk:
    region: Region
    order: int
    bands: list[imaging.LineBand]  # in (rotated) crop coordinates
    box: tuple[int, int, int, int]  # chunk box in crop coordinates
    image: Image.Image | None = None
    scale: float = 1.0
    budget: int = 256
    depth: int = 0
    variant: str = ""
    result: object | None = None
    text: str = ""
    ok: bool = True
    previous: "Chunk | None" = None  # earlier attempt of a re-read single line


@dataclass
class RegionJob:
    region: Region
    crop: Image.Image  # straightened crop
    crop_origin: tuple[int, int]  # top-left of the crop on the page (before rotation)
    pre_size: tuple[int, int]  # crop size before rotation
    angle: float
    line_height: float
    bands: list[imaging.LineBand]
    chunks: list[Chunk] = field(default_factory=list)
    alt: "RegionJob | None" = None  # 180° flipped alternative for rotated notes


class Recognizer:
    def __init__(self, ocr_model, settings, progress: ProgressFn | None = None, cancel: Callable[[], bool] | None = None):
        self.ocr = ocr_model
        self.settings = settings
        self.progress = progress or (lambda e: None)
        self.cancel = cancel or (lambda: False)

    # ------------------------------------------------------------------ #
    def recognize_page(self, page_img: Image.Image, regions: list[Region], page_index: int) -> None:
        jobs: list[RegionJob] = []
        for r in regions:
            if r.kind not in TEXT_KINDS:
                continue
            job = self._prepare_region(page_img, r, regions)
            if job is None:
                r.text = ""
                continue
            jobs.append(job)
            if job.alt is not None:
                jobs.append(job.alt)
        all_chunks = [c for j in jobs for c in j.chunks]
        self._run_chunks(all_chunks, page_index)
        # Self-check and retry
        if self.settings.recognition.verify_and_retry:
            for depth in range(4):
                retry: list[Chunk] = []
                for job in jobs:
                    new_chunks: list[Chunk] = []
                    for c in job.chunks:
                        if c.ok or c.depth > depth:
                            new_chunks.append(c)
                            continue
                        replacement = self._split_or_rescale(job, c)
                        if replacement:
                            retry.extend(replacement)
                            new_chunks.extend(replacement)
                        else:
                            new_chunks.append(c)
                    job.chunks = new_chunks
                if not retry or self.cancel():
                    break
                self.progress({"type": "status", "page": page_index, "message": f"Re-reading {len(retry)} uncertain piece(s)…"})
                self._run_chunks(retry, page_index)
        # Pick the better orientation for rotated notes
        by_region: dict[str, list[RegionJob]] = {}
        for job in jobs:
            by_region.setdefault(job.region.id, []).append(job)
        for region_id, candidates in by_region.items():
            best = max(candidates, key=_job_score)
            self._finalize(best)

    # ------------------------------------------------------------------ #
    def _prepare_region(self, page_img: Image.Image, r: Region, regions: list[Region]) -> RegionJob | None:
        lay = self.settings.layout
        pad = lay.region_padding
        W, H = page_img.size
        x0, y0, x1, y1 = r.bbox
        cx0, cy0 = int(max(0, math.floor(x0 - pad))), int(max(0, math.floor(y0 - pad)))
        cx1, cy1 = int(min(W, math.ceil(x1 + pad))), int(min(H, math.ceil(y1 + pad)))
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return None
        crop = page_img.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        # White out other regions that intrude into this crop.
        draw = ImageDraw.Draw(crop)
        for o in regions:
            if o is r or G.intersection_area(o.bbox, (cx0, cy0, cx1, cy1)) <= 0:
                continue
            if G.containment(o.bbox, r.bbox) > 0.6 and o.kind not in (MARGIN, FOOTNOTE, HEADER, FOOTER, PAGE_NUMBER):
                continue  # nested piece of this region (e.g. a title inside a text block)
            if G.containment(r.bbox, o.bbox) > 0.9:
                continue  # we are inside it; don't erase ourselves
            if G.containment(o.bbox, r.bbox) > 0.3:
                pieces = [o.bbox]  # a block sitting inside our box (e.g. a margin note): erase it
            else:
                pieces = _box_difference(o.bbox, r.bbox)  # only erase what lies outside our own box
            for ox0, oy0, ox1, oy1 in pieces:
                draw.rectangle([ox0 - cx0, oy0 - cy0, ox1 - cx0, oy1 - cy0], fill=(255, 255, 255))
        if lay.use_polygons and r.polygon and len(r.polygon) >= 3:
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).polygon([(px - cx0, py - cy0) for px, py in r.polygon], fill=255)
            # dilate the polygon slightly so descenders survive
            mask = Image.fromarray(np.asarray(mask)).filter(_max_filter(pad))
            white = Image.new("RGB", crop.size, (255, 255, 255))
            crop = Image.composite(crop, white, mask)
        pre_size = crop.size
        angle = 0.0
        gray = imaging.to_gray(crop)
        rotate_candidate = r.kind == MARGIN or r.label == "vertical_text" or (r.height > 2.4 * r.width and r.height > 120)
        if lay.straighten_rotated_text and rotate_candidate:
            angle = imaging.estimate_text_angle(gray)
            if angle:
                crop = imaging.rotate_image(crop, angle, expand=True)
                gray = imaging.to_gray(crop)
        mask_arr = imaging.ink_mask(gray)
        bands = imaging.segment_lines(gray, mask_arr)
        if bands:
            # rules, underlines and separator lines confuse the reader: paint them out
            erased = _erase_rules(crop, mask_arr, imaging.median_line_height(bands) or bands[0].height)
            if erased is not None:
                crop = erased
                gray = imaging.to_gray(crop)
                mask_arr = imaging.ink_mask(gray)
                bands = imaging.segment_lines(gray, mask_arr) or bands
        if not bands:
            ink = imaging.tight_ink_box(mask_arr)
            if ink is None:
                return None
            bands = [imaging.LineBand(ink[1], ink[3], ink[0], ink[2])]
        lh = imaging.median_line_height(bands) or float(bands[0].height)
        job = RegionJob(r, crop, (cx0, cy0), pre_size, angle, lh, bands)
        job.chunks = self._make_chunks(job, bands, depth=0)
        if r.kind == FOOTNOTE:
            first = gray[bands[0].top : bands[0].bottom, :]
            last = gray[bands[-1].top : bands[-1].bottom, :]
            r.notes = [n for n in r.notes if n not in ("starts_with_equals", "ends_with_equals")]
            if imaging.detect_equals_sign(first, "right", lh):
                r.notes.append("starts_with_equals")
            if imaging.detect_equals_sign(last, "left", lh):
                r.notes.append("ends_with_equals")
        if angle and abs(angle) > 10:
            flipped = imaging.rotate_image(crop, 180, expand=True)
            fg = imaging.to_gray(flipped)
            fb = imaging.segment_lines(fg) or bands
            alt = RegionJob(r, flipped, (cx0, cy0), pre_size, angle + 180, imaging.median_line_height(fb) or lh, fb)
            alt.chunks = self._make_chunks(alt, fb, depth=0, variant="flipped")
            job.alt = alt
        return job

    def _make_chunks(self, job: RegionJob, bands: list[imaging.LineBand], depth: int, variant: str = "") -> list[Chunk]:
        """Group lines for reading.  Arabic-GLM-OCR is excellent on single lines and on
        tall paragraphs, but unreliable on strips of 2–5 lines (it skips or reorders
        lines there), so groups are either one line or a real block."""
        rec = self.settings.recognition
        groups: list[list[imaging.LineBand]] = []
        rest = list(bands)
        n = len(rest)
        if job.region.kind == TABLE:
            groups = [rest]  # tables need their whole structure in view
        elif job.region.kind == FOOTNOTE:
            # footnotes: markers like (٢) must stay on their own line, and the first
            # line alone guarantees an opening "=" continuation is never lost
            per = max(1, rec.footnote_lines_per_chunk)
            if per > 1 and n > 1:
                groups.append([rest.pop(0)])
            groups.extend(_blocks(rest, per, min_size=1) if per > 1 else [[b] for b in rest])
        elif rec.granularity == "lines":
            groups = [[b] for b in rest]
        else:
            min_block = max(2, rec.min_block_lines) if rec.granularity == "auto" else 2
            if n < min_block:
                groups = [[b] for b in rest] if n > 1 else [rest]
            else:
                groups = _blocks(rest, max(min_block, rec.max_lines_per_chunk), min_size=min_block)
        return [self._chunk_from_bands(job, g, bands, depth, variant, i) for i, g in enumerate(groups)]

    def _chunk_from_bands(self, job: RegionJob, group, all_bands, depth: int, variant: str, order: int) -> Chunk:
        W, H = job.crop.size
        lh = job.line_height or group[0].height
        first_i = all_bands.index(group[0]) if group[0] in all_bands else 0
        last_i = all_bands.index(group[-1]) if group[-1] in all_bands else len(all_bands) - 1
        top = group[0].top
        bottom = group[-1].bottom
        # extend half-way to neighbouring lines (keeps diacritics) but never into them
        prev_bottom = all_bands[first_i - 1].bottom if first_i > 0 else 0
        next_top = all_bands[last_i + 1].top if last_i + 1 < len(all_bands) else H
        y0 = max(prev_bottom, int(top - lh * 0.35), 0) if first_i > 0 else max(0, int(top - lh * 0.5))
        y1 = min(next_top, int(bottom + lh * 0.35), H) if last_i + 1 < len(all_bands) else min(H, int(bottom + lh * 0.5))
        left = min(b.left for b in group)
        right = max(b.right for b in group)
        x0 = max(0, int(left - lh * 0.6))
        x1 = min(W, int(right + lh * 0.6))
        chunk = Chunk(job.region, order, list(group), (x0, y0, x1, y1), depth=depth, variant=variant)
        self._render_chunk(job, chunk)
        return chunk

    def _render_chunk(self, job: RegionJob, chunk: Chunk, extra_scale: float = 1.0) -> None:
        rec = self.settings.recognition
        sub = job.crop.crop(chunk.box)
        if len(chunk.bands) == 1:
            band = chunk.bands[0]
            sub = _isolate_line(sub, band.top - chunk.box[1], band.bottom - chunk.box[1])
        lh = job.line_height or (chunk.box[3] - chunk.box[1])
        target = int(rec.target_line_height * extra_scale) if rec.target_line_height else 0
        if not target and extra_scale != 1.0:
            target = int(lh * extra_scale)
        img, scale = imaging.prepare_crop(sub, target, lh, rec.max_upscale * extra_scale, rec.max_pixels)
        chunk.image, chunk.scale = img, scale
        n = max(1, len(chunk.bands))
        width_ratio = (chunk.box[2] - chunk.box[0]) / max(8.0, lh)
        budget = int(n * (8.5 * width_ratio + 10) + 32)
        if not rec.adaptive_token_budget:
            budget = rec.max_new_tokens
        chunk.budget = max(48, min(rec.max_new_tokens, budget))

    # ------------------------------------------------------------------ #
    def _run_chunks(self, chunks: list[Chunk], page_index: int) -> None:
        if not chunks:
            return
        bs = self.settings.recognition.batch_size or _auto_batch(self.ocr)
        ordered = sorted(chunks, key=lambda c: c.image.size[0] * c.image.size[1])
        done = 0
        for i in range(0, len(ordered), bs):
            if self.cancel():
                return
            batch = ordered[i : i + bs]
            ids = sorted({c.region.id for c in batch})
            self.progress({"type": "reading", "page": page_index, "regions": ids, "done": done, "total": len(ordered)})
            try:
                results = self.ocr.recognize([c.image for c in batch], [c.budget for c in batch], self.settings, self.cancel)
            except Exception as exc:
                if len(batch) > 1 and _is_oom(exc):
                    log.warning("Out of memory with batch %d; retrying one by one", len(batch))
                    import gc

                    gc.collect()
                    from ..models import hardware

                    hardware.empty_cache()
                    results = []
                    for c in batch:
                        results.extend(self.ocr.recognize([c.image], [c.budget], self.settings, self.cancel))
                else:
                    raise
            for c, res in zip(batch, results):
                c.result = res
                c.text = res.text
                c.ok = self._verify(c)
            done += len(batch)

    def _verify(self, c: Chunk) -> bool:
        res = c.result
        if res is None:
            return False
        if c.region.kind == TABLE:
            return not res.looped
        raw = res.text or ""
        if "<" in raw:
            raw = html_to_text(raw)
        raw = strip_markdown(raw)
        lines = [l for l in raw.split("\n") if l.strip()]
        expected = len(c.bands)
        if res.looped or res.hit_limit:
            return False
        if expected >= 1 and not lines:
            return False
        # The model must not drop lines: small chunks get no tolerance at all,
        # large ones a little (line segmentation can over-count diacritic rows).
        tolerance = 0 if expected <= 4 else max(1, int(expected * 0.12))
        if len(lines) < expected - tolerance:
            return False
        if expected >= 2 and len(lines) > expected * 1.6 + 1:
            return False
        if expected >= 2 and len(lines) == expected:
            # a line far longer than its width allows = merged lines or an invented tail
            ratios = [len(t.strip()) / max(1, b.right - b.left) for t, b in zip(lines, c.bands)]
            med = sorted(ratios)[len(ratios) // 2] if expected >= 3 else min(ratios)
            if med > 0 and max(ratios) > 2.2 * med and max(len(t) for t in lines) > 40:
                return False
        return True

    def _split_or_rescale(self, job: RegionJob, c: Chunk) -> list[Chunk] | None:
        if len(c.bands) >= 2:
            # a block that failed its self-check is re-read line by line
            out = []
            for k, band in enumerate(c.bands):
                nc = self._chunk_from_bands(job, [band], job.bands, c.depth + 1, c.variant, 0)
                nc.order = c.order + (k + 1) / 1000.0
                out.append(nc)
            return out
        if c.previous is None:
            # a single line that failed: read it once more, a little larger
            nc = Chunk(c.region, c.order, c.bands, c.box, depth=c.depth + 1, variant=c.variant, previous=c)
            self._render_chunk(job, nc, extra_scale=1.3)
            return [nc]
        return None

    # ------------------------------------------------------------------ #
    def _finalize(self, job: RegionJob) -> None:
        r = job.region
        chunks = sorted(job.chunks, key=lambda c: c.order)
        # for rescaled single-line retries keep whichever attempt is better
        texts, conf_num, conf_den, flagged, notes = [], 0.0, 0, False, []
        for c in chunks:
            prev = c.previous
            if prev is not None and prev.result is not None and c.result is not None:
                if _chunk_score(prev) > _chunk_score(c):
                    c = prev
            res = c.result
            if res is None:
                flagged = True
                continue
            cleaned, cn = clean_model_output(res.text, self.settings)
            notes.extend(cn)
            texts.append(cleaned)
            if res.confidence is not None:
                conf_num += res.confidence * max(1, res.tokens)
                conf_den += max(1, res.tokens)
            if not c.ok:
                flagged = True
                notes.append("could not fully verify")
            if res.looped:
                notes.append("loop stopped")
        r.text = "\n".join(t for t in texts if t)
        if self.settings.text.line_mode == "paragraphs":
            from .text import reflow_paragraphs

            r.text = reflow_paragraphs(r.text)
        r.confidence = conf_num / conf_den if conf_den else None
        r.angle = job.angle % 360 if job.angle else 0.0
        if r.angle > 180:
            r.angle -= 360
        threshold = self.settings.recognition.flag_confidence_below
        r.flagged = flagged or (r.confidence is not None and r.confidence < threshold)
        r.line_height = job.line_height
        keep = [n for n in r.notes if n.startswith("lines:") or n in ("starts_with_equals", "ends_with_equals", "recovered")]
        r.notes = keep + sorted(set(notes))
        r.lines = self._line_boxes(job, r.text)

    def _line_boxes(self, job: RegionJob, text: str) -> list[Line]:
        """Map recognised lines onto the line bands found in the image (page coordinates)."""
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return []
        bands = job.bands
        # transform from straightened crop coords -> page coords
        cw, ch = job.crop.size
        pw, ph = job.pre_size
        if job.angle:
            # crop was rotated with expand=True around its centre
            back = G.multiply(
                G.multiply(G.translate(-cw / 2, -ch / 2), G.rotate_about(-job.angle, 0, 0)),
                G.translate(pw / 2 + job.crop_origin[0], ph / 2 + job.crop_origin[1]),
            )
        else:
            back = G.translate(job.crop_origin[0], job.crop_origin[1])
        boxes: list[tuple[float, float, float, float]] = []
        if len(bands) == len(lines):
            boxes = [(b.left, b.top, b.right, b.bottom) for b in bands]
        elif bands:
            # distribute lines proportionally over the bands' vertical extent
            top, bottom = bands[0].top, bands[-1].bottom
            left = min(b.left for b in bands)
            right = max(b.right for b in bands)
            step = (bottom - top) / len(lines)
            boxes = [(left, top + i * step, right, top + (i + 1) * step) for i in range(len(lines))]
        out = []
        for text_line, box in zip(lines, boxes):
            pts = G.apply_box(back, box)
            quad = [(round(x, 2), round(y, 2)) for x, y in pts]
            out.append(Line(bbox=G.bounds(pts), text=text_line, angle=-job.angle if job.angle else 0.0, quad=quad))
        return out


def _chunk_score(c: Chunk) -> float:
    res = c.result
    if res is None:
        return -1e9
    s = res.confidence or 0.0
    if res.looped:
        s -= 1.0
    if res.hit_limit:
        s -= 0.5
    if not c.ok:
        s -= 0.25
    return s


def _job_score(job: RegionJob) -> float:
    scores = [_chunk_score(c) for c in job.chunks]
    return sum(scores) / len(scores) if scores else -1e9


def _erase_rules(img: Image.Image, mask: np.ndarray, line_height: float) -> Image.Image | None:
    """White out long, thin, stand-alone horizontal strokes (rules, underlines).
    Kashida/tatweel strokes are attached to letters, so they are never touched."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lh = max(8.0, float(line_height))
    hits = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= 2.5 * lh and stats[i, cv2.CC_STAT_HEIGHT] <= max(4.0, 0.14 * lh)]
    if not hits:
        return None
    arr = np.asarray(img.convert("RGB")).copy()
    erase = np.isin(labels, hits)
    # include the anti-aliased fringe of the stroke
    erase = cv2.dilate(erase.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    arr[erase] = 255
    return Image.fromarray(arr)


def _auto_batch(ocr) -> int:
    device = getattr(ocr, "device", "cpu") or "cpu"
    if device == "cpu":
        return 1
    if device == "mps":
        return 2
    from ..models import hardware

    mem = hardware.device_memory_gb(device) or 8
    return 8 if mem >= 20 else 6 if mem >= 12 else 4 if mem >= 8 else 2


def _isolate_line(img: Image.Image, top: int, bottom: int) -> Image.Image:
    """Erase ink that belongs to the neighbouring lines (descenders from above,
    ascenders from below) from a single-line crop."""
    import cv2

    gray = imaging.to_gray(img)
    mask = imaging.ink_mask(gray)
    h = mask.shape[0]
    if h < 8 or mask.sum() == 0:
        return img
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    erase = np.zeros(n, dtype=bool)
    for i in range(1, n):
        y, ch = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        y1 = y + ch
        overlap = min(y1, bottom) - max(y, top)
        if overlap <= 0:
            erase[i] = True  # entirely outside this line's band
        elif (y <= 1 or y1 >= h - 1) and overlap < 0.25 * ch:
            erase[i] = True  # a neighbour's stroke poking in from the edge
    if not erase.any():
        return img
    arr = np.asarray(img.convert("RGB")).copy()
    arr[erase[labels]] = 255
    return Image.fromarray(arr)


def _blocks(bands: list, max_size: int, min_size: int) -> list[list]:
    """Split ``bands`` into balanced consecutive groups of at most ``max_size``
    (and, when possible, at least ``min_size``) lines."""
    n = len(bands)
    if n == 0:
        return []
    k = max(1, math.ceil(n / max_size))
    size = math.ceil(n / k)
    if size < min_size and n >= min_size:
        k = max(1, n // min_size)
        size = math.ceil(n / k)
    groups = []
    for i in range(0, n, size):
        groups.append(bands[i : i + size])
    # a short tail would fall in the unreliable range: merge it into the previous group
    if len(groups) > 1 and len(groups[-1]) < min_size:
        groups[-2].extend(groups.pop())
    return groups


def _box_difference(a, b) -> list[tuple[float, float, float, float]]:
    """Parts of box ``a`` that lie outside box ``b`` (up to four rectangles)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if G.intersection_area(a, b) <= 0:
        return [tuple(a)]
    out = []
    if ay0 < by0:
        out.append((ax0, ay0, ax1, min(ay1, by0)))
    if ay1 > by1:
        out.append((ax0, max(ay0, by1), ax1, ay1))
    my0, my1 = max(ay0, by0), min(ay1, by1)
    if ax0 < bx0:
        out.append((ax0, my0, min(ax1, bx0), my1))
    if ax1 > bx1:
        out.append((max(ax0, bx1), my0, ax1, my1))
    return [r for r in out if r[2] > r[0] and r[3] > r[1]]


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "oom" in text or exc.__class__.__name__ == "OutOfMemoryError"


def _max_filter(pad: int):
    from PIL import ImageFilter

    size = max(3, (pad // 2) * 2 + 1)
    return ImageFilter.MaxFilter(min(size, 15))
