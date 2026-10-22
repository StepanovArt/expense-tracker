"""Image -> validated Receipt with one retry: the logic shared by the API and the Gradio demo."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from receipt_vlm.eval.metrics import is_arith_consistent, is_fully_consistent
from receipt_vlm.eval.parse import parse_prediction
from receipt_vlm.prompts import RETRY_REMINDER
from receipt_vlm.serve.backends import Backend

MAX_ATTEMPTS = 2


@dataclass
class ExtractResult:
    ok: bool
    receipt: dict[str, Any] | None
    raw: str
    attempts: int
    latency_s: float
    status: str
    error: str | None = None
    consistency: dict[str, Any] = field(default_factory=dict)


def consistency(receipt: dict[str, Any]) -> dict[str, Any]:
    items = receipt.get("items") or []
    items_sum = round(sum(float(i["total"]) for i in items if i.get("total") is not None), 2)
    return {
        "items_sum": items_sum if items else None,
        "items_match_subtotal": is_arith_consistent(receipt),
        "all_consistent": is_fully_consistent(receipt),
    }


def extract_receipt(
    backend: Backend, image: Image.Image, max_attempts: int = MAX_ATTEMPTS
) -> ExtractResult:
    """Ask the model; if the answer is not schema-valid JSON, retry once with a reminder."""
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    raw, status, error = "", "parse_fail", None
    for attempt in range(1, max_attempts + 1):
        raw, _ = backend.chat(image, history)
        pr = parse_prediction(raw)
        status, error = pr.status, pr.error
        if pr.status == "ok" and pr.receipt is not None:
            receipt = pr.receipt.model_dump(mode="json")
            return ExtractResult(
                True,
                receipt,
                raw,
                attempt,
                time.perf_counter() - t0,
                status,
                None,
                consistency(receipt),
            )
        history = [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": RETRY_REMINDER},
        ]
    return ExtractResult(False, None, raw, max_attempts, time.perf_counter() - t0, status, error)
