"""Wrapper around Arabic-GLM-OCR-v2 (GLM-OCR architecture) for region recognition."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

from PIL import Image

from . import hardware, hub

log = logging.getLogger(__name__)


@dataclass
class Recognition:
    text: str
    confidence: float | None
    tokens: int
    hit_limit: bool  # stopped because of the token budget
    looped: bool  # stopped by the repetition detector
    seconds: float = 0.0


def has_loop(ids: list[int]) -> bool:
    """True if the tail of ``ids`` repeats itself: a short pattern many times, or a
    long passage (a whole line or two) twice in a row — both are degenerate output."""
    n = len(ids)
    if n < 16:
        return False
    tail = ids[-160:]
    m = len(tail)
    for period in range(1, 41):
        reps = 12 if period == 1 else (6 if period <= 3 else 3)
        span = period * reps
        if span > m:
            break
        if period > 3 and span < 24:
            continue
        seg = tail[-span:]
        unit = seg[:period]
        if all(seg[i] == unit[i % period] for i in range(span)):
            return True
    # long periods: the last `p` tokens equal the `p` tokens before them
    for period in range(41, min(400, n // 2) + 1):
        if ids[-period:] == ids[-2 * period : -period]:
            return True
    return False


class OCRModel:
    """Loads the OCR model once and recognises batches of image crops."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = None
        self.model_id = ""
        self.loaded_signature: tuple | None = None
        self.lock = threading.RLock()
        self.last_used = 0.0

    # ------------------------------------------------------------------ #
    def signature(self, settings) -> tuple:
        hw, ms = settings.hardware, settings.models
        return (ms.ocr_model, ms.ocr_revision, hw.device, hw.gpu_index, hw.dtype, hw.attention, hw.quantization, hw.torch_compile)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, settings, progress: Callable[[dict], None] | None = None) -> None:
        with self.lock:
            sig = self.signature(settings)
            if self.model is not None and sig == self.loaded_signature:
                return
            self.unload()
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            hub.configure_environment(settings.models)
            if progress:
                progress({"stage": "model", "message": "Locating OCR model…"})
            path = hub.ensure_model(settings.models.ocr_model, settings.models, settings.models.ocr_revision, progress, "Arabic-GLM-OCR")
            device, dtype = hardware.resolve(settings.hardware)
            if progress:
                progress({"stage": "model", "message": f"Loading OCR model on {hardware.describe(device)}…"})
            kwargs: dict = {"dtype": dtype}
            attn = settings.hardware.attention
            if attn != "auto":
                kwargs["attn_implementation"] = attn
            quant = settings.hardware.quantization
            if quant != "none" and device.startswith("cuda"):
                from transformers import BitsAndBytesConfig

                if quant == "8bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
                    )
                kwargs["device_map"] = {"": device}
            t0 = time.time()
            processor = AutoProcessor.from_pretrained(path)
            try:
                model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
            except (ValueError, ImportError) as exc:
                if "attn_implementation" in kwargs:
                    log.warning("Attention '%s' unavailable (%s); using default.", attn, exc)
                    kwargs.pop("attn_implementation")
                    model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
                else:
                    raise
            if "device_map" not in kwargs:
                model = model.to(device)
            model.eval()
            if settings.hardware.torch_compile:
                try:
                    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
                except Exception as exc:  # pragma: no cover
                    log.warning("torch.compile failed: %s", exc)
            self.model, self.processor = model, processor
            self.device, self.dtype = device, dtype
            self.model_id = settings.models.ocr_model
            self.loaded_signature = sig
            self.last_used = time.time()
            log.info("OCR model loaded on %s (%s) in %.1fs", device, dtype, time.time() - t0)

    def unload(self) -> None:
        with self.lock:
            if self.model is None:
                return
            self.model = None
            self.processor = None
            self.loaded_signature = None
            import gc

            gc.collect()
            hardware.empty_cache()

    # ------------------------------------------------------------------ #
    def recognize(self, images: list[Image.Image], budgets: list[int], settings, cancel: Callable[[], bool] | None = None) -> list[Recognition]:
        """Recognise a batch of crops. ``budgets`` caps new tokens per crop."""
        if not images:
            return []
        with self.lock:
            if self.model is None:
                raise RuntimeError("OCR model is not loaded")
            import torch
            from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

            rec = settings.recognition
            prompt = rec.prompt or "Text Recognition:"
            conversations = [
                [{"role": "user", "content": [{"type": "image", "image": im.convert("RGB")}, {"type": "text", "text": prompt}]}]
                for im in images
            ]
            processor = self.processor
            template_kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            if len(conversations) > 1:
                try:
                    inputs = processor.apply_chat_template(conversations, processor_kwargs={"padding": True}, **template_kwargs)
                except TypeError:  # older transformers
                    inputs = processor.apply_chat_template(conversations, padding=True, **template_kwargs)
            else:
                inputs = processor.apply_chat_template(conversations[0], **template_kwargs)
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.device)
            if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype and self.dtype is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            prompt_len = inputs["input_ids"].shape[1]
            batch = len(images)
            beams = rec.num_beams if rec.decoding == "beam" else 1

            logprobs: list[list[float]] = [[] for _ in range(batch)]
            stopped_loop = [False] * batch
            budget_hit = [False] * batch

            class Recorder(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    if beams == 1:
                        lp = torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.tolist()
                        for i, v in enumerate(lp):
                            logprobs[i].append(v)
                    return scores

            class Stopper(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    rows = input_ids.shape[0]
                    done = torch.zeros(rows, dtype=torch.bool, device=input_ids.device)
                    if cancel is not None and cancel():
                        done[:] = True
                        return done
                    gen = input_ids[:, prompt_len:]
                    length = gen.shape[1]
                    for r in range(rows):
                        i = r // beams
                        if length >= budgets[i]:
                            done[r] = True
                            budget_hit[i] = True
                        elif length >= 16 and length % 4 == 0 and has_loop(gen[r].tolist()):
                            done[r] = True
                            stopped_loop[i] = True
                    return done

            gen_kwargs: dict = {
                "max_new_tokens": int(max(budgets)),
                "logits_processor": LogitsProcessorList([Recorder()]),
                "stopping_criteria": StoppingCriteriaList([Stopper()]),
            }
            if rec.decoding == "sampling":
                gen_kwargs.update(do_sample=True, temperature=rec.temperature, top_p=rec.top_p)
            elif rec.decoding == "beam":
                gen_kwargs.update(do_sample=False, num_beams=rec.num_beams, early_stopping=True)
            else:
                gen_kwargs.update(do_sample=False, num_beams=1)
            if rec.repetition_penalty and rec.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = rec.repetition_penalty
            t0 = time.time()
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - t0
            self.last_used = time.time()
            sequences = out if isinstance(out, torch.Tensor) else out.sequences
            eos_ids = set(self._eos_ids())
            results: list[Recognition] = []
            for i in range(batch):
                ids = sequences[i][prompt_len:].tolist()
                # trim padding / eos
                trimmed = []
                for t in ids:
                    if t in eos_ids:
                        break
                    trimmed.append(t)
                text = processor.decode(trimmed, skip_special_tokens=True)
                lp = logprobs[i][: len(trimmed)] if logprobs[i] else []
                conf = math.exp(sum(lp) / len(lp)) if lp else None
                results.append(
                    Recognition(
                        text=text,
                        confidence=conf,
                        tokens=len(trimmed),
                        hit_limit=budget_hit[i] and not stopped_loop[i],
                        looped=stopped_loop[i],
                        seconds=elapsed / batch,
                    )
                )
            return results

    def _eos_ids(self) -> list[int]:
        cfg = getattr(self.model, "generation_config", None)
        eos = getattr(cfg, "eos_token_id", None)
        if eos is None:
            return []
        ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        pad = getattr(cfg, "pad_token_id", None)
        if pad is not None:
            ids.append(pad)
        return ids
