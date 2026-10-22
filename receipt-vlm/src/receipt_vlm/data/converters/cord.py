"""CORD v2 (naver-clova-ix/cord-v2) -> unified schema.

Each HF row has ``image`` and ``ground_truth`` (a JSON string) whose ``gt_parse`` looks like::

    {"menu": [{"nm": "CINNAMON SUGAR", "unitprice": "17,000", "cnt": "1 x", "price": "17,000"},
              ...] | {single item},
     "sub_total": {"subtotal_price": "17,000", "tax_price": "1,700", "service_price": ...},
     "total": {"total_price": "17,000", "cashprice": "20,000", "changeprice": "3,000", ...}}

CORD annotates every text line of the ``menu``, ``sub_total`` and ``total`` super-classes, so
items, subtotal, tax and total are labelled exhaustively (a missing key = not printed).
``store_info`` and ``payment_info`` were removed from the public release, so merchant, date
and time are *unlabelled*. payment_method is labelled only when exactly one of
cash / credit card / e-money amounts is printed. Prices are Indonesian Rupiah with ``.`` or
``,`` as thousands separators. Sub-menu lines (``sub``) are add-ons of a parent item and are
not emitted as separate items (DECISIONS.md).
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from receipt_vlm.data.record import Converted, Split
from receipt_vlm.schema import Item, Receipt
from receipt_vlm.values import parse_amount

SPLIT_MAP: dict[str, Split] = {"train": "train", "validation": "val", "test": "test"}
_PAYMENT_KEYS = {"cashprice": "cash", "creditcardprice": "card", "emoneyprice": "mobile"}


def _text(v: Any) -> str | None:
    """CORD values are strings, or lists of strings when a field spans several boxes."""
    if v is None:
        return None
    if isinstance(v, list):
        parts = [_text(x) for x in v]
        joined = " ".join(p for p in parts if p)
        return joined or None
    if isinstance(v, dict):
        return None
    s = " ".join(str(v).split())
    return s or None


def _money(v: Any) -> float | None:
    s = _text(v)
    if s is None:
        return None
    return parse_amount(s.replace("@", ""), thousands_dot=True)


def _quantity(v: Any) -> float | None:
    s = _text(v)
    if s is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    return parse_amount(m.group(0), thousands_dot=False)


def _as_list(v: Any) -> list[dict[str, Any]]:
    if v is None:
        return []
    if isinstance(v, dict):
        return [v]
    return [x for x in v if isinstance(x, dict)]


def convert_gt_parse(gt_parse: dict[str, Any]) -> tuple[Receipt, list[str]]:
    items: list[Item] = []
    for entry in _as_list(gt_parse.get("menu")):
        name = _text(entry.get("nm"))
        if not name:
            continue
        items.append(
            Item(
                name=name,
                quantity=_quantity(entry.get("cnt")),
                unit_price=_money(entry.get("unitprice")),
                total=_money(entry.get("price")),
            )
        )
    sub = gt_parse.get("sub_total") or gt_parse.get("subtotal") or {}
    tot = gt_parse.get("total") or {}
    if isinstance(sub, list):
        sub = sub[0] if sub else {}
    if isinstance(tot, list):
        tot = tot[0] if tot else {}

    labelled = ["items.name", "items.quantity", "items.unit_price", "items.total"]
    labelled += ["subtotal", "tax", "total"]
    paid = [m for k, m in _PAYMENT_KEYS.items() if _text(tot.get(k))]
    payment = None
    if len(paid) == 1:
        payment = paid[0]
        labelled.append("payment_method")
    receipt = Receipt(
        items=items,
        subtotal=_money(sub.get("subtotal_price")),
        tax=_money(sub.get("tax_price")),
        total=_money(tot.get("total_price")),
        payment_method=payment,
    )
    return receipt, labelled


def convert_row(row: dict[str, Any], split: str, index: int) -> Converted:
    gt = json.loads(row["ground_truth"])
    receipt, labelled = convert_gt_parse(gt["gt_parse"])
    meta = gt.get("meta") or {}
    image = row["image"]
    if isinstance(image, dict) and image.get("bytes") is not None:  # undecoded HF Image
        img_bytes = bytes(image["bytes"])
    else:  # PIL image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()
    image_id = meta.get("image_id", index)
    return Converted(
        source_id=f"{split}_{image_id}",
        source="cord",
        image_path=None,
        image_bytes=img_bytes,
        target=receipt,
        labelled=labelled,
        language="id",
        official_split=SPLIT_MAP[split],
        meta={"cord_image_id": image_id, "cord_split": split},
    )


def iter_cord(rows_by_split: dict[str, Iterable[dict[str, Any]]]) -> Iterator[Converted]:
    for split, rows in rows_by_split.items():
        for i, row in enumerate(rows):
            yield convert_row(row, split, i)


def load_cord_from_hub(dataset_id: str = "naver-clova-ix/cord-v2") -> Iterator[Converted]:
    """Download CORD v2 with ``datasets`` (needs huggingface.co access)."""
    from datasets import Image, load_dataset

    ds = load_dataset(dataset_id)
    rows = {}
    for split in ("train", "validation", "test"):
        part = ds[split].cast_column("image", Image(decode=False))
        rows[split] = part
    yield from iter_cord(rows)
