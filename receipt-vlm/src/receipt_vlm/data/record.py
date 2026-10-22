"""Dataset record types shared by converters, the builder, training and evaluation.

``labelled`` lists which fields a sample's source annotates *exhaustively*. Only those are
scored (CLAUDE.md §4.2): a null gold value in a labelled field means "not printed on the
receipt" (so predicting a value is a false positive); an unlabelled field is never scored.
Item attributes are tracked separately as ``items.name``, ``items.quantity``,
``items.unit_price``, ``items.total``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from receipt_vlm.schema import ITEM_FIELDS, SCALAR_FIELDS, Receipt

Source = Literal["cord", "sroie", "wildreceipt", "synthetic"]
Split = Literal["train", "val", "test"]

ITEM_LABEL_KEYS: tuple[str, ...] = tuple(f"items.{k}" for k in ITEM_FIELDS)
LABEL_KEYS: tuple[str, ...] = (*SCALAR_FIELDS, *ITEM_LABEL_KEYS)


def items_labelled(labelled: list[str] | set[str]) -> bool:
    return "items.name" in labelled


class Sample(BaseModel):
    """One processed example, one JSON line in ``data/processed/{split}.jsonl``."""

    id: str
    source: Source
    split: Split
    image: str  # path relative to the project root
    image_sha256: str
    phash: str
    width: int
    height: int
    language: str  # stratification language (dataset-level; may be "mixed"/"id"/...)
    target: Receipt
    labelled: list[str]
    # "nospace": source transcripts have whitespace removed (WildReceipt), so string fields
    # are compared with all whitespace stripped on both sides.
    string_compare: Literal["default", "nospace"] = "default"
    meta: dict[str, Any] = Field(default_factory=dict)

    def gold_dict(self) -> dict[str, Any]:
        return self.target.model_dump(mode="json")


@dataclass
class Converted:
    """Converter output before images are copied, hashed and split."""

    source_id: str  # unique within the source
    source: Source
    image_path: Path | None  # either a file on disk ...
    image_bytes: bytes | None  # ... or encoded bytes (HF datasets)
    target: Receipt
    labelled: list[str]
    language: str
    official_split: Split | None = None
    string_compare: Literal["default", "nospace"] = "default"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.labelled) - set(LABEL_KEYS)
        if unknown:
            raise ValueError(f"unknown label keys {unknown}")
        if (self.image_path is None) == (self.image_bytes is None):
            raise ValueError("exactly one of image_path / image_bytes must be set")
