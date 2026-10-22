"""SROIE (ICDAR 2019 task 3) -> unified schema.

Source format (``data/key/<id>.json`` in the zzzDavid mirror)::

    {"company": "...", "date": "25/12/2018", "address": "...", "total": "9.00"}

SROIE labels exactly four keys for every receipt; ``address`` is not part of our schema, so
the labelled fields are merchant, date and total. Malaysian receipts print day-first dates.
A key that is missing or that our parser cannot read is marked *unlabelled* for that sample
(rather than null), so it is never scored against a guess.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from receipt_vlm.data.record import Converted
from receipt_vlm.schema import Receipt
from receipt_vlm.utils.logging import get_logger
from receipt_vlm.values import parse_amount, parse_date

log = get_logger(__name__)


def convert_key(key: dict[str, Any]) -> tuple[Receipt, list[str]]:
    """One SROIE key dict -> (target, labelled fields)."""
    labelled: list[str] = []
    merchant = " ".join(str(key.get("company") or "").split()) or None
    if merchant:
        labelled.append("merchant")
    date = parse_date(key.get("date"), order="DMY")
    if date:
        labelled.append("date")
    total = parse_amount(key.get("total"))
    if total is not None:
        labelled.append("total")
    return Receipt(merchant=merchant, date=date, total=total), labelled


def iter_sroie(root: str | Path) -> Iterator[Converted]:
    """Yield converted samples from a checkout of the SROIE mirror (``root/data/{img,key}``)."""
    root = Path(root)
    key_dir, img_dir = root / "data" / "key", root / "data" / "img"
    if not key_dir.is_dir():
        raise FileNotFoundError(f"SROIE keys not found under {key_dir}")
    n_bad = 0
    for key_path in sorted(key_dir.glob("*.json")):
        img_path = img_dir / f"{key_path.stem}.jpg"
        if not img_path.exists():
            n_bad += 1
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        target, labelled = convert_key(key)
        yield Converted(
            source_id=key_path.stem,
            source="sroie",
            image_path=img_path,
            image_bytes=None,
            target=target,
            labelled=labelled,
            language="en",
            meta={"raw_date": key.get("date"), "raw_total": key.get("total")},
        )
    if n_bad:
        log.warning("SROIE: %d keys without an image were skipped", n_bad)
