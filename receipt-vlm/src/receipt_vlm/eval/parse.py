"""Model output -> (status, strict Receipt, lenient dict).

* ``parse_fail``: no JSON object could be decoded (after stripping ``<think>`` blocks and
  markdown code fences, and locating the first balanced ``{...}``);
* ``schema_fail``: a JSON object, but it does not validate against :class:`Receipt`;
* ``ok``: strictly valid.

The strict prediction (primary metric) is the validated Receipt, or an empty prediction for
any failure. The lenient prediction coerces each field independently (e.g. ``"12.50"`` ->
12.5, ``"25/12/2018"`` -> ``"2018-12-25"``) and is reported as a secondary metric only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from receipt_vlm.schema import ITEM_FIELDS, NUMERIC_FIELDS, SCALAR_FIELDS, Receipt
from receipt_vlm.values import parse_amount, parse_date, parse_time

Status = Literal["ok", "schema_fail", "parse_fail"]

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


@dataclass(frozen=True)
class ParseResult:
    status: Status
    receipt: Receipt | None
    data: dict[str, Any] | None
    lenient: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    extra_text: bool = False

    @property
    def strict_dict(self) -> dict[str, Any]:
        """The prediction scored by the primary (strict) metric."""
        if self.receipt is None:
            return Receipt().model_dump(mode="json")
        return self.receipt.model_dump(mode="json")


def _first_object(text: str) -> tuple[str | None, bool]:
    """Return the first balanced top-level JSON object substring and whether text surrounds it."""
    start = text.find("{")
    if start < 0:
        return None, False
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                obj = text[start : i + 1]
                extra = bool(text[:start].strip() or text[i + 1 :].strip())
                return obj, extra
    return None, False


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    """(object, had_extra_text, error)."""
    t = _THINK.sub("", text or "").strip()
    fence = _FENCE.search(t)
    fenced = fence is not None
    if fence:
        t = fence.group(1).strip()
    candidate, extra = _first_object(t)
    if candidate is None:
        return None, False, "no JSON object found"
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        return None, False, f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return None, False, "top-level JSON is not an object"
    return obj, extra or fenced, None


def _lenient_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        return parse_amount(v)
    return None


def _lenient_str(v: Any) -> str | None:
    if v is None or isinstance(v, dict | list):
        return None
    s = str(v).strip()
    return s or None


def lenient_coerce(data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort per-field coercion of a parsed (possibly schema-invalid) object."""
    out: dict[str, Any] = {}
    for f in SCALAR_FIELDS:
        v = data.get(f)
        if f in NUMERIC_FIELDS:
            out[f] = _lenient_number(v)
        elif f == "date":
            out[f] = parse_date(_lenient_str(v), order="DMY") if v is not None else None
            if out[f] is None and isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                out[f] = v
        elif f == "time":
            out[f] = parse_time(_lenient_str(v))
        elif f in ("currency", "language"):
            s = _lenient_str(v)
            out[f] = s.upper() if (s and f == "currency") else (s.lower() if s else None)
        else:
            out[f] = _lenient_str(v)
    items: list[dict[str, Any]] = []
    raw_items = data.get("items")
    if isinstance(raw_items, list):
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            name = _lenient_str(it.get("name"))
            if name is None:
                continue
            row: dict[str, Any] = {"name": name}
            for k in ITEM_FIELDS[1:]:
                row[k] = _lenient_number(it.get(k))
            items.append(row)
    out["items"] = items
    return out


def parse_prediction(text: str) -> ParseResult:
    data, extra, err = extract_json(text)
    if data is None:
        return ParseResult("parse_fail", None, None, lenient_coerce({}), err, False)
    lenient = lenient_coerce(data)
    try:
        receipt = Receipt.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        msg = f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
        return ParseResult("schema_fail", None, data, lenient, msg, extra)
    return ParseResult("ok", receipt, data, lenient, None, extra)
