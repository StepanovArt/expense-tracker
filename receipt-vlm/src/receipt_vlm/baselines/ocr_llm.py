"""Baseline B0: Tesseract OCR -> text-only LLM with the same instruction as the VLMs.

Tesseract 5 with ``eng+ara+rus`` traineddata covers all three target scripts offline
(PaddleOCR needs model downloads; DECISIONS.md #6). Page segmentation mode 6 ("uniform block
of text") was chosen on 24 validation receipts: it recovers 75 % of printed amounts vs 31 % for
modes 3/4, which drop the right-hand amount column (DECISIONS.md). The LLM sees the OCR text with line breaks
preserved, followed by the exact same JSON instruction the VLMs get.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from PIL import Image

from receipt_vlm.models.inference import GenOutput
from receipt_vlm.prompts import text_messages


def tesseract_ocr(im: Image.Image, langs: str = "eng+ara+rus", psm: int = 6) -> str:
    import pytesseract

    return str(pytesseract.image_to_string(im, lang=langs, config=f"--psm {psm}"))


def dominant_script(text: str) -> str:
    """'ara' / 'rus' / 'eng' by letter counts (Arabic block, Cyrillic block, Latin)."""
    ar = sum("\u0600" <= c <= "\u06ff" for c in text)
    ru = sum("\u0400" <= c <= "\u04ff" for c in text)
    la = sum(c.isascii() and c.isalpha() for c in text)
    best = max((ar, "ara"), (ru, "rus"), (la, "eng"))
    return best[1]


def script_aware_ocr(im: Image.Image, langs: str = "eng+ara+rus", psm: int = 6) -> str:
    """Two passes: all languages to find the dominant script, then that script (+ English for
    codes/numbers). Loading eng+ara+rus together makes Tesseract mix scripts within words
    (e.g. Latin receipts read as Cyrillic look-alikes), which would make B0 a strawman."""
    first = tesseract_ocr(im, langs, psm)
    script = dominant_script(first)
    second = {"ara": "ara+eng", "rus": "rus+eng", "eng": "eng"}[script]
    return tesseract_ocr(im, second, psm) if second != langs else first


class OCRLLMGenerator:
    def __init__(
        self,
        llm: Callable[[list[dict[str, Any]]], tuple[str, int]],
        ocr_langs: str = "eng+ara+rus",
        name: str = "ocr_llm",
        ocr: Callable[[Image.Image, str], str] = script_aware_ocr,
    ) -> None:
        self.name = name
        self.llm = llm
        self.ocr_langs = ocr_langs
        self.ocr = ocr

    def generate(self, images: Sequence[Image.Image]) -> list[GenOutput]:
        outs = []
        for im in images:
            t0 = time.perf_counter()
            text = self.ocr(im, self.ocr_langs)
            t_ocr = time.perf_counter() - t0
            raw, n_tok = self.llm(text_messages(text))
            extra = {"ocr_s": round(t_ocr, 3), "ocr_chars": len(text)}
            outs.append(GenOutput(raw, time.perf_counter() - t0, n_tok, extra))
        return outs
