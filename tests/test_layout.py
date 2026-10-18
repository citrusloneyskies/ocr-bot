import numpy as np
from PIL import Image, ImageDraw

from warraq.core import imaging, layout as L
from warraq.core.document import BODY, FOOTNOTE, HEADER, MARGIN, Region
from warraq.models.layout_model import Detection
from warraq.settings import Settings


def r(id, x0, y0, x1, y1, kind=BODY):
    return Region(id=id, kind=kind, label="text", bbox=(x0, y0, x1, y1))


def test_xy_cut_reads_columns_right_to_left():
    title = r("t", 100, 50, 900, 100)
    right = r("right", 520, 150, 900, 900)
    left = r("left", 100, 150, 480, 900)
    order = [x.id for x in L.xy_cut_rtl([left, title, right])]
    assert order == ["t", "right", "left"]


def _page_with_text_blocks():
    """Synthetic page: main column, a margin note, a footnote rule and smaller footnote lines."""
    w, h = 1000, 1400
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i in range(10):  # body lines, tall strokes
        y = 150 + i * 60
        d.rectangle((150, y, 820, y + 30), fill=0)
    for i in range(3):  # margin note lines
        y = 300 + i * 40
        d.rectangle((860, y, 960, y + 18), fill=0)
    d.rectangle((500, 1080, 820, 1082), fill=0)  # footnote rule (right side, short)
    for i in range(3):  # footnote lines, smaller
        y = 1100 + i * 36
        d.rectangle((150, y, 820, y + 16), fill=0)
    return np.asarray(img)


def test_margins_and_footnotes_are_classified():
    gray = _page_with_text_blocks()
    ctx = L.build_context(gray)
    dets = [
        Detection("text", 0.9, (145, 140, 825, 760), 1, None),
        Detection("text", 0.8, (855, 290, 965, 400), 2, None),
        Detection("text", 0.8, (145, 1095, 825, 1190), 3, None),
    ]
    regions = L.analyze(dets, ctx, Settings(), 0)
    kinds = sorted(x.kind for x in regions)
    assert kinds == sorted([BODY, MARGIN, FOOTNOTE]), [(x.kind, x.bbox) for x in regions]
    margin = [x for x in regions if x.kind == MARGIN][0]
    assert margin.side == "right"


def test_line_segmentation_counts_lines():
    gray = _page_with_text_blocks()[140:760, 145:825]
    bands = imaging.segment_lines(gray)
    assert len(bands) == 10


def test_equals_sign_detection():
    img = Image.new("L", (600, 60), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((560, 22, 590, 26), fill=0)  # "=" at the right edge
    d.rectangle((560, 34, 590, 38), fill=0)
    for x in range(40, 500, 60):
        d.rectangle((x, 12, x + 40, 48), fill=0)  # words
    gray = np.asarray(img)
    assert imaging.detect_equals_sign(gray, "right", 50)
    plain = Image.new("L", (600, 60), 255)
    ImageDraw.Draw(plain).rectangle((40, 12, 590, 48), fill=0)
    assert not imaging.detect_equals_sign(np.asarray(plain), "right", 50)


def test_refine_page_number():
    from warraq.core.document import Page

    page = Page(index=0, source_index=0, source_path="", width=1000, height=1400)
    pn = r("n", 480, 1340, 520, 1370, kind=HEADER)
    pn.text = "١٢"
    L.refine_after_ocr([pn], None, page)
    assert pn.kind == "page_number" and page.printed_number == "١٢"


def test_number_beside_text_is_a_margin_note():
    """A short vertical note mid-page that the detector calls "number" is a marginal note."""
    gray = _page_with_text_blocks()
    ctx = L.build_context(gray)
    dets = [
        Detection("text", 0.9, (145, 140, 825, 760), 1, None),
        Detection("number", 0.6, (855, 290, 965, 400), 2, None),
        Detection("number", 0.7, (470, 1300, 530, 1340), 3, None),  # a real folio number at the foot
    ]
    regions = L.analyze(dets, ctx, Settings(), 0)
    by_box = {tuple(round(v) for v in x.bbox[:2]): x for x in regions}
    note = [x for x in regions if x.bbox[0] > 840]
    assert note and note[0].kind == MARGIN and note[0].side == "right", [(x.kind, x.bbox) for x in regions]
    folio = [x for x in regions if x.bbox[1] > 1250]
    assert all(x.kind != MARGIN for x in folio)


def test_recovered_note_hugs_its_ink():
    """Ink nobody detected becomes a region whose box stays close to the ink."""
    w, h = 1000, 1000
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i in range(8):
        d.rectangle((300, 150 + i * 60, 900, 180 + i * 60), fill=0)
    for i in range(4):  # an undetected note on the left: short words, like real text
        for x in range(60, 220, 32):
            d.rectangle((x, 300 + i * 40, x + 24, 318 + i * 40), fill=0)
    ctx = L.build_context(np.asarray(img))
    regions = L.analyze([Detection("text", 0.9, (295, 140, 905, 640), 1, None)], ctx, Settings(), 0)
    notes = [x for x in regions if x.label == "recovered"]
    assert len(notes) == 1 and notes[0].kind == MARGIN
    x0, y0, x1, y1 = notes[0].bbox
    assert 40 <= x0 <= 60 and 220 <= x1 <= 245 and y1 <= 300 + 3 * 40 + 18 + 25
