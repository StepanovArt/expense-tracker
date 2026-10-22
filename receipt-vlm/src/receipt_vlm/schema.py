"""Unified receipt schema: the single contract between data, models, eval, rewards and API.

Validation is deliberately strict because "schema-valid JSON" is itself a headline metric:

* unknown keys are rejected (``extra="forbid"``);
* numbers must be JSON numbers: ``"12.50"`` (a string) or ``true`` is invalid, an int is fine;
* ``date`` must be a real calendar date in ISO 8601 ``YYYY-MM-DD``; ``time`` must be ``HH:MM``;
* ``currency`` must look like an ISO 4217 code and ``language`` like an ISO 639-1 code.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Annotated, Any, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

SCALAR_FIELDS: Final[tuple[str, ...]] = (
    "merchant",
    "date",
    "time",
    "currency",
    "subtotal",
    "tax",
    "total",
    "payment_method",
    "language",
)
NUMERIC_FIELDS: Final[frozenset[str]] = frozenset({"subtotal", "tax", "total"})
ITEM_FIELDS: Final[tuple[str, ...]] = ("name", "quantity", "unit_price", "total")
ITEM_NUMERIC_FIELDS: Final[frozenset[str]] = frozenset({"quantity", "unit_price", "total"})
# Order of keys in serialized targets (schema order, CLAUDE.md §4.5).
FIELD_ORDER: Final[tuple[str, ...]] = (
    "merchant",
    "date",
    "time",
    "currency",
    "items",
    "subtotal",
    "tax",
    "total",
    "payment_method",
    "language",
)
ALL_FIELDS: Final[tuple[str, ...]] = FIELD_ORDER

PAYMENT_METHODS: Final[tuple[str, ...]] = ("cash", "card", "mobile", "voucher", "other")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_LANG_RE = re.compile(r"^[a-z]{2}$")


def _strict_number(v: Any) -> Any:
    """Accept JSON numbers only (int/float, not bool, not str, finite)."""
    if v is None:
        return v
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"expected a JSON number, got {type(v).__name__}")
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("number must be finite")
    return float(v)


def _strict_str(v: Any) -> Any:
    if v is None:
        return v
    if not isinstance(v, str):
        raise ValueError(f"expected a string, got {type(v).__name__}")
    return v


Number = Annotated[float | None, BeforeValidator(_strict_number)]
OptStr = Annotated[str | None, BeforeValidator(_strict_str)]


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, BeforeValidator(_strict_str)]
    quantity: Number = None
    unit_price: Number = None
    total: Number = None


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant: OptStr = None
    date: OptStr = None  # ISO 8601 YYYY-MM-DD
    time: OptStr = None  # HH:MM
    currency: OptStr = None  # ISO 4217, e.g. AED, USD, RUB
    items: list[Item] = Field(default_factory=list)
    subtotal: Number = None
    tax: Number = None
    total: Number = None
    payment_method: OptStr = None
    language: OptStr = None  # ISO 639-1 of the receipt text

    @field_validator("date")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _DATE_RE.match(v):
            raise ValueError("date must be YYYY-MM-DD")
        dt.date.fromisoformat(v)  # raises on e.g. 2023-02-30
        return v

    @field_validator("time")
    @classmethod
    def _check_time(cls, v: str | None) -> str | None:
        if v is not None and not _TIME_RE.match(v):
            raise ValueError("time must be HH:MM (24h)")
        return v

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, v: str | None) -> str | None:
        if v is not None and not _CURRENCY_RE.match(v):
            raise ValueError("currency must be an ISO 4217 code such as USD")
        return v

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str | None) -> str | None:
        if v is not None and not _LANG_RE.match(v):
            raise ValueError("language must be an ISO 639-1 code such as en")
        return v

    def to_target_json(self) -> str:
        """Compact JSON in schema key order: the exact string the model is trained to emit."""
        return dump_target(self.model_dump(mode="json"))


def _fmt_number(x: float) -> int | float:
    """2.0 -> 2 so targets read like real receipts and use fewer tokens."""
    return int(x) if float(x).is_integer() and abs(x) < 1e15 else round(x, 4)


def canonical_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Schema-ordered dict with numbers in compact form (ints where integral)."""
    out: dict[str, Any] = {}
    for key in FIELD_ORDER:
        val = data.get(key)
        if key == "items":
            items = []
            for it in val or []:
                row: dict[str, Any] = {}
                for k in ITEM_FIELDS:
                    v = it.get(k)
                    row[k] = _fmt_number(v) if k in ITEM_NUMERIC_FIELDS and v is not None else v
                items.append(row)
            out[key] = items
        elif key in NUMERIC_FIELDS and val is not None:
            out[key] = _fmt_number(val)
        else:
            out[key] = val
    return out


def dump_target(data: dict[str, Any]) -> str:
    import json

    return json.dumps(canonical_dict(data), ensure_ascii=False, separators=(",", ":"))


def json_schema() -> dict[str, Any]:
    return Receipt.model_json_schema()
