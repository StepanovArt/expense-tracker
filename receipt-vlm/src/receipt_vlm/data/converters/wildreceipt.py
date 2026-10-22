"""WildReceipt (MMOCR close-set KIE annotations) -> unified schema.

Each line of ``train.txt`` / ``test.txt`` is a JSON object::

    {"file_name": "image_files/Image_16/11/receipt_0.jpeg", "height": 348, "width": 348,
     "annotations": [{"box": [x1,y1,...,x4,y4], "text": "$13.99", "label": 15}, ...]}

with ``label`` indexing ``class_list.txt`` (1 Store_name_value, 7 Date_value, 9 Time_value,
11 Prod_item_value, 13 Prod_quantity_value, 15 Prod_price_value, 17 Subtotal_value,
19 Tax_value, 23 Total_value, ...). Every text box carries a class, so a missing value class
means "not printed" (labelled null). Two properties of the labels shape the conversion:

* transcripts have their spaces removed ("KOREANRESTAURANT"), so samples are tagged
  ``string_compare="nospace"`` and eval compares strings with whitespace stripped;
* items are boxes, not rows. We pair item names with prices only when the pairing is
  unambiguous (same count, row-aligned); otherwise items are unlabelled for that receipt.
  Quantity boxes are too inconsistent ("4.00xITEMS" is a receipt-level count) to use.
Receipts are mostly US, so numeric dates are read month-first unless that is impossible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from receipt_vlm.data.record import Converted, Split
from receipt_vlm.schema import Item, Receipt
from receipt_vlm.values import parse_amount, parse_date, parse_time

STORE_NAME, DATE, TIME, ITEM, PRICE = 1, 7, 9, 11, 15
SUBTOTAL, TAX, TOTAL = 17, 19, 23


def _yspan(box: list[float]) -> tuple[float, float]:
    ys = box[1::2]
    return min(ys), max(ys)


def _xmin(box: list[float]) -> float:
    return min(box[0::2])


def _reading_order(anns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(a: dict[str, Any]) -> tuple[float, float]:
        y0, y1 = _yspan(a["box"])
        return (round((y0 + y1) / 2 / max(y1 - y0, 1.0)), _xmin(a["box"]))

    return sorted(anns, key=key)


def _single_amount(anns: list[dict[str, Any]]) -> tuple[float | None, bool]:
    """(value, labelled). No box -> (None, True); conflicting/unreadable -> (None, False)."""
    if not anns:
        return None, True
    values = {parse_amount(a["text"]) for a in anns}
    if None in values or len(values) != 1:
        return None, False
    return values.pop(), True


def _y_overlap(a: list[float], b: list[float]) -> float:
    a0, a1 = _yspan(a)
    b0, b1 = _yspan(b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(min(a1 - a0, b1 - b0), 1.0)


def _pair_items(items: list[dict[str, Any]], prices: list[dict[str, Any]]) -> list[Item] | None:
    if len(items) != len(prices):
        return None
    by_y = lambda a: sum(_yspan(a["box"])) / 2  # noqa: E731
    items, prices = sorted(items, key=by_y), sorted(prices, key=by_y)
    out: list[Item] = []
    for it, pr in zip(items, prices, strict=True):
        if _y_overlap(it["box"], pr["box"]) < 0.5 or _xmin(pr["box"]) <= _xmin(it["box"]):
            return None
        amount = parse_amount(pr["text"])
        name = it["text"].strip()
        if amount is None or not name:
            return None
        out.append(Item(name=name, total=amount))
    return out


def convert_annotation(record: dict[str, Any]) -> tuple[Receipt, list[str]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for a in record["annotations"]:
        if a.get("text"):
            by_label.setdefault(int(a["label"]), []).append(a)

    labelled = ["merchant"]
    names = [a["text"].strip() for a in _reading_order(by_label.get(STORE_NAME, []))]
    merchant = " ".join(n for n in names if n) or None

    date = None
    if DATE in by_label:
        date = parse_date(" ".join(a["text"] for a in by_label[DATE]), order="MDY")
        if date:
            labelled.append("date")
    else:
        labelled.append("date")

    time = None
    if TIME in by_label:
        time = parse_time(" ".join(a["text"] for a in by_label[TIME]))
        if time:
            labelled.append("time")
    else:
        labelled.append("time")

    nums: dict[str, float | None] = {}
    for field, cls in (("subtotal", SUBTOTAL), ("tax", TAX), ("total", TOTAL)):
        value, ok = _single_amount(by_label.get(cls, []))
        nums[field] = value
        if ok:
            labelled.append(field)

    items = _pair_items(by_label.get(ITEM, []), by_label.get(PRICE, []))
    if items is not None:
        labelled += ["items.name", "items.total"]
    receipt = Receipt(
        merchant=merchant,
        date=date,
        time=time,
        items=items or [],
        subtotal=nums["subtotal"],
        tax=nums["tax"],
        total=nums["total"],
    )
    return receipt, labelled


def iter_wildreceipt(root: str | Path) -> Iterator[Converted]:
    """Yield samples from an extracted ``wildreceipt/`` directory (train.txt + test.txt)."""
    root = Path(root)
    for ann_file, split in (("train.txt", "train"), ("test.txt", "test")):
        path = root / ann_file
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            receipt, labelled = convert_annotation(record)
            file_name = record["file_name"]
            official: Split = "train" if split == "train" else "test"
            yield Converted(
                source_id=file_name.removeprefix("image_files/").rsplit(".", 1)[0],
                source="wildreceipt",
                image_path=root / file_name,
                image_bytes=None,
                target=receipt,
                labelled=labelled,
                language="en",
                official_split=official,
                string_compare="nospace",
                meta={"file_name": file_name},
            )
