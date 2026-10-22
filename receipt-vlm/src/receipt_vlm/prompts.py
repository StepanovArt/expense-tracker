"""The one fixed prompt used for every model, in training and in evaluation (CLAUDE.md §4.5, §5.5).

Changing any string here changes every experiment: bump PROMPT_VERSION and re-run baselines.
"""

from __future__ import annotations

from typing import Any

from receipt_vlm.schema import PAYMENT_METHODS

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a receipt parser. You read a photo of a shop receipt and return its content as "
    "a single JSON object. You never add explanations, markdown or code fences."
)

INSTRUCTION = f"""Extract the receipt in this image as JSON with exactly these keys, in this order:
{{"merchant": string|null, "date": "YYYY-MM-DD"|null, "time": "HH:MM"|null, "currency": ISO 4217 code|null, "items": [{{"name": string, "quantity": number|null, "unit_price": number|null, "total": number|null}}], "subtotal": number|null, "tax": number|null, "total": number|null, "payment_method": {"|".join(PAYMENT_METHODS)}|null, "language": ISO 639-1 code|null}}
Rules:
- Copy merchant and item names as printed (keep the original script and language).
- Numbers are JSON numbers without currency symbols or thousands separators, e.g. 1250.5.
- items: one entry per purchased line, in printed order; total is the line amount.
- Use null for anything that is not printed on the receipt; use [] if there are no items.
- language is the main language of the receipt text, e.g. en, ar, ru.
Return only the JSON object."""

TEXT_INSTRUCTION_PREFIX = (
    "The text below was produced by OCR from a photo of a receipt "
    "(line breaks preserved, may contain recognition errors).\n\nOCR TEXT:\n"
)

RETRY_REMINDER = (
    "Your previous answer was not valid JSON for the required schema. "
    "Return valid JSON only: one object with exactly the keys listed above, nothing else."
)


def vlm_messages(image: Any) -> list[dict[str, Any]]:
    """Chat messages for a vision-language model; ``image`` is a PIL image or a path/URL."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": INSTRUCTION}],
        },
    ]


def text_messages(ocr_text: str) -> list[dict[str, Any]]:
    """Same task and instruction for the OCR + text-LLM baseline (B0)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEXT_INSTRUCTION_PREFIX + ocr_text + "\n\n" + INSTRUCTION},
    ]
