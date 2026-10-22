"""Value normalisation used before every comparison (CLAUDE.md §5.2).

* strings: Unicode NFKC, lowercase, collapse whitespace, strip punctuation at both ends;
  ``nospace`` mode (WildReceipt, whose labels have spaces removed) drops all whitespace;
* numbers: equal iff ``|pred - gold| <= 0.01``;
* dates/times: compared as ISO strings (strict schema already guarantees the format);
* currency: ISO code compared case-insensitively (symbols are mapped in converters, not here);
* payment_method: free text is mapped onto the canonical vocabulary of the prompt.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from rapidfuzz.distance import Levenshtein

NUM_TOL: Final[float] = 0.01
_WS = re.compile(r"\s+")

PAYMENT_SYNONYMS: Final[dict[str, str]] = {
    "cash": "cash",
    "card": "card",
    "credit card": "card",
    "debit card": "card",
    "credit": "card",
    "debit": "card",
    "visa": "card",
    "mastercard": "card",
    "amex": "card",
    "mobile": "mobile",
    "apple pay": "mobile",
    "google pay": "mobile",
    "e-wallet": "mobile",
    "ewallet": "mobile",
    "voucher": "voucher",
    "gift card": "voucher",
    "coupon": "voucher",
    "other": "other",
}


def _strip_edge_punct(s: str) -> str:
    start, end = 0, len(s)
    while start < end and unicodedata.category(s[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(s[end - 1]).startswith("P"):
        end -= 1
    return s[start:end]


def norm_str(s: str | None, nospace: bool = False) -> str:
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).lower()
    t = _WS.sub(" ", t).strip()
    t = _strip_edge_punct(t).strip()
    if nospace:
        t = _WS.sub("", t)
    return t


def str_equal(pred: str | None, gold: str | None, nospace: bool = False) -> bool:
    return norm_str(pred, nospace) == norm_str(gold, nospace)


def str_similarity(pred: str | None, gold: str | None, nospace: bool = False) -> float:
    """1 - normalised Levenshtein distance on normalised strings, in [0, 1]."""
    a, b = norm_str(pred, nospace), norm_str(gold, nospace)
    if not a and not b:
        return 1.0
    return float(Levenshtein.normalized_similarity(a, b))


def num_equal(pred: float | None, gold: float | None, tol: float = NUM_TOL) -> bool:
    if pred is None or gold is None:
        return False
    return abs(float(pred) - float(gold)) <= tol + 1e-9


def norm_payment(s: str | None) -> str:
    t = norm_str(s)
    if t in PAYMENT_SYNONYMS:
        return PAYMENT_SYNONYMS[t]
    for k, v in PAYMENT_SYNONYMS.items():
        if k in t:
            return v
    return t


def norm_code(s: str | None) -> str:
    return norm_str(s).replace(" ", "")


def field_equal(field: str, pred: object, gold: object, nospace: bool = False) -> bool:
    """Equality for a scalar field or item attribute, both values non-null."""
    if field in (
        "subtotal",
        "tax",
        "total",
        "quantity",
        "unit_price",
        "items.total",
        "items.quantity",
        "items.unit_price",
    ):
        if isinstance(pred, bool) or not isinstance(pred, int | float):
            return False
        if isinstance(gold, bool) or not isinstance(gold, int | float):
            return False
        return num_equal(float(pred), float(gold))
    if field in ("date", "time"):
        return str(pred).strip() == str(gold).strip()
    if field in ("currency", "language"):
        return norm_code(str(pred)) == norm_code(str(gold))
    if field == "payment_method":
        return norm_payment(str(pred)) == norm_payment(str(gold))
    # merchant, items.name
    return str_equal(str(pred), str(gold), nospace)
