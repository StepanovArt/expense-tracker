"""Automatic error taxonomy for RQ4 (CLAUDE.md §5.6), built on the evaluation code.

Every prediction is reduced to a list of :class:`ErrorEvent` (one per wrong field instance or
item, or one per output-level failure). Categories:

=====================  ==================================================================
truncated_output       no parseable JSON and the output hit max_new_tokens / is unbalanced
parse_fail             no parseable JSON (otherwise)
schema:date_format     JSON parsed, but ``date`` is not a valid YYYY-MM-DD date
schema:number_as_str   JSON parsed, but a number was emitted as a string / bool
schema:other           any other schema violation (unknown key, bad time/currency, ...)
wrong_number           money/quantity present in both, |pred - gold| > 0.01
wrong_date / wrong_time
wrong_string           merchant or item name present in both, differs after normalisation
script_mismatch        a wrong string written in a different script than the gold
                       (e.g. Arabic merchant transliterated to Latin) - language-specific
wrong_code             currency / language / payment_method differs
missing_field          gold has a value, prediction is null
hallucinated_field     gold is null (not printed), prediction has a value
missing_item / extra_item   unmatched gold / predicted line item
=====================  ==================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from receipt_vlm.eval.items_matching import match_items
from receipt_vlm.eval.metrics import count_instance
from receipt_vlm.eval.parse import ParseResult, parse_prediction
from receipt_vlm.schema import ITEM_FIELDS, NUMERIC_FIELDS, SCALAR_FIELDS

CATEGORIES: Final[tuple[str, ...]] = (
    "truncated_output",
    "parse_fail",
    "schema:date_format",
    "schema:number_as_str",
    "schema:other",
    "wrong_number",
    "wrong_date",
    "wrong_time",
    "wrong_string",
    "script_mismatch",
    "wrong_code",
    "missing_field",
    "hallucinated_field",
    "missing_item",
    "extra_item",
)


@dataclass(frozen=True)
class ErrorEvent:
    category: str
    field: str
    gold: Any = None
    pred: Any = None


def script_of(text: str) -> str:
    counts = {"arabic": 0, "cyrillic": 0, "latin": 0}
    for c in text:
        if "؀" <= c <= "ۿ":
            counts["arabic"] += 1
        elif "Ѐ" <= c <= "ӿ":
            counts["cyrillic"] += 1
        elif c.isascii() and c.isalpha():
            counts["latin"] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else "none"


def _wrong_category(field: str, pred: Any, gold: Any) -> str:
    attr = field.split(".", 1)[-1]
    if field in NUMERIC_FIELDS or attr in ("quantity", "unit_price", "total"):
        return "wrong_number"
    if field == "date":
        return "wrong_date"
    if field == "time":
        return "wrong_time"
    if field in ("currency", "language", "payment_method"):
        return "wrong_code"
    if isinstance(pred, str) and isinstance(gold, str) and script_of(pred) != script_of(gold):
        return "script_mismatch"
    return "wrong_string"


def _field_event(field: str, pred: Any, gold: Any, nospace: bool) -> ErrorEvent | None:
    tp, fp, fn = count_instance(field, pred, gold, nospace)
    if tp or not (fp or fn):
        return None
    if fp and fn:
        return ErrorEvent(_wrong_category(field, pred, gold), field, gold, pred)
    if fn:
        return ErrorEvent("missing_field", field, gold, None)
    return ErrorEvent("hallucinated_field", field, None, pred)


def _schema_category(pr: ParseResult) -> str:
    err = pr.error or ""
    if err.startswith("date"):
        return "schema:date_format"
    if "expected a JSON number" in err:
        return "schema:number_as_str"
    return "schema:other"


def _looks_truncated(raw: str, n_output_tokens: int | None, max_new_tokens: int | None) -> bool:
    if (
        n_output_tokens is not None
        and max_new_tokens is not None
        and n_output_tokens >= max_new_tokens
    ):
        return True
    return raw.count("{") > raw.count("}")


def classify(
    sample: Mapping[str, Any],
    raw: str,
    n_output_tokens: int | None = None,
    max_new_tokens: int | None = None,
) -> list[ErrorEvent]:
    pr = parse_prediction(raw)
    if pr.status == "parse_fail":
        cat = (
            "truncated_output"
            if _looks_truncated(raw, n_output_tokens, max_new_tokens)
            else "parse_fail"
        )
        return [ErrorEvent(cat, "*")]
    if pr.status == "schema_fail":
        return [ErrorEvent(_schema_category(pr), (pr.error or "?").split(":")[0])]
    gold = sample["target"]
    pred = pr.strict_dict
    lab = set(sample["labelled"])
    nospace = sample.get("string_compare") == "nospace"
    events: list[ErrorEvent] = []
    for f in SCALAR_FIELDS:
        if f in lab:
            ev = _field_event(f, pred.get(f), gold.get(f), nospace)
            if ev:
                events.append(ev)
    if "items.name" in lab:
        gi, pi = gold.get("items") or [], pred.get("items") or []
        pairs = match_items(pi, gi, nospace)
        for p_idx, g_idx in pairs:
            for a in ITEM_FIELDS:
                if f"items.{a}" in lab:
                    ev = _field_event(f"items.{a}", pi[p_idx].get(a), gi[g_idx].get(a), nospace)
                    if ev:
                        events.append(ev)
        matched_g = {g for _, g in pairs}
        matched_p = {p for p, _ in pairs}
        events += [
            ErrorEvent("missing_item", "items", g.get("name"), None)
            for k, g in enumerate(gi)
            if k not in matched_g
        ]
        events += [
            ErrorEvent("extra_item", "items", None, p.get("name"))
            for k, p in enumerate(pi)
            if k not in matched_p
        ]
    return events
