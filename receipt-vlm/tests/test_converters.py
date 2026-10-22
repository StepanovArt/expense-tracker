"""Converter tests on real annotation records (see tests/fixtures/README.md for provenance)."""

import json
from pathlib import Path

import pytest

from receipt_vlm.data.converters.cord import convert_gt_parse
from receipt_vlm.data.converters.sroie import convert_key
from receipt_vlm.data.converters.wildreceipt import convert_annotation
from receipt_vlm.schema import Receipt

FIX = Path(__file__).parent / "fixtures"
RAW = Path(__file__).parents[1] / "data" / "raw"


# ---------------------------------------------------------------- SROIE (verbatim records)
def _sroie(i: str) -> dict[str, str]:
    return json.loads((FIX / "sroie" / f"{i}.json").read_text())


def test_sroie_typical_record() -> None:
    r, labelled = convert_key(_sroie("000"))
    assert r.merchant == "BOOK TA .K (TAMAN DAYA) SDN BHD"
    assert r.date == "2018-12-25" and r.total == 9.0
    assert labelled == ["merchant", "date", "total"]
    assert r.items == [] and r.currency is None  # unlabelled, stays null


def test_sroie_compact_dates_and_rm_prefix() -> None:
    assert convert_key(_sroie("068"))[0].date == "2018-03-04"  # "20180304"
    r, _ = convert_key(_sroie("152"))  # "25032018", total "RM41.45"
    assert r.date == "2018-03-25" and r.total == 41.45


def test_sroie_empty_total_is_unlabelled_not_null_scored() -> None:
    r, labelled = convert_key(_sroie("033"))  # "total": ""
    assert r.total is None and "total" not in labelled
    assert r.date == "2018-03-10"  # "10 MAR 2018"


def test_sroie_negative_total_kept() -> None:
    # 347 is an adjustment note whose printed "Total Payable" really is -1.73
    assert convert_key(_sroie("347"))[0].total == -1.73


@pytest.mark.data
def test_sroie_real_data() -> None:
    root = RAW / "sroie"
    if not (root / "data" / "key").is_dir():
        pytest.skip("SROIE mirror not downloaded (make data)")
    from receipt_vlm.data.converters.sroie import iter_sroie

    rows = list(iter_sroie(root))
    assert len(rows) == 626
    share = {f: sum(f in r.labelled for r in rows) / len(rows) for f in ("date", "total")}
    assert share["date"] > 0.99 and share["total"] > 0.99


# ------------------------------------------------- WildReceipt (verbatim MMOCR toy records)
def _wild() -> list[dict[str, object]]:
    lines = (FIX / "wildreceipt" / "data.txt").read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def test_wildreceipt_clean_receipt() -> None:
    r, labelled = convert_annotation(_wild()[1])  # CHOEUN KOREAN RESTAURANT
    assert r.merchant == "CHOEUN KOREANRESTAURANT"
    assert r.date == "2016-12-30"  # "12/30/2016FRI"
    assert r.time == "19:19"
    assert (r.subtotal, r.tax, r.total) == (55.96, 4.48, 60.44)
    assert [i.name for i in r.items] == [
        "BIBIM.OCTOPUT1",
        "S-FOODP.CAKT1",
        "PORKDUMPLINT1",
        "LABEEFRIBT1",
    ]
    assert [i.total for i in r.items] == [13.99, 14.99, 8.99, 17.99]  # "￥17.99" label noise
    assert sum(i.total or 0 for i in r.items) == pytest.approx(r.subtotal)
    assert {"items.name", "items.total"} <= set(labelled)
    assert "items.quantity" not in labelled


def test_wildreceipt_ambiguous_items_are_unlabelled() -> None:
    r, labelled = convert_annotation(_wild()[0])  # SAFEWAY: 6 item boxes vs 9 price boxes
    assert r.merchant == "SAFEWAY"
    assert r.items == [] and "items.name" not in labelled
    # no date/time/total boxes in this crop -> labelled as not printed
    assert r.date is None and "date" in labelled
    assert r.total is None and "total" in labelled


# ---------------------------------------- CORD (structure-faithful, not verbatim: README)
def _cord(name: str) -> dict[str, object]:
    return json.loads((FIX / "cord" / f"{name}.json").read_text())["gt_parse"]


def test_cord_list_menu_thousands_and_subitems() -> None:
    r, labelled = convert_gt_parse(_cord("gt_parse_list_menu"))
    assert [(i.name, i.quantity, i.unit_price, i.total) for i in r.items] == [
        ("ICE BLACKCOFFEE", 2.0, 41000.0, 82000.0),
        ("CHOCO CROISSANT", 1.0, None, 25000.0),  # list-valued nm joined, sub-item dropped
    ]
    assert (r.subtotal, r.tax, r.total) == (107000.0, 10700.0, 123050.0)
    assert r.payment_method == "cash" and "payment_method" in labelled
    assert "merchant" not in labelled and "date" not in labelled


def test_cord_single_dict_menu_and_card_payment() -> None:
    r, labelled = convert_gt_parse(_cord("gt_parse_single_menu"))
    assert len(r.items) == 1 and r.items[0].total == 17000.0
    assert r.tax is None and "tax" in labelled  # no tax line printed -> labelled null
    assert r.payment_method == "card"


def test_cord_two_payment_methods_is_unlabelled() -> None:
    gt = {
        "menu": [],
        "total": {"total_price": "10.000", "cashprice": "5.000", "emoneyprice": "5.000"},
    }
    r, labelled = convert_gt_parse(gt)
    assert r.payment_method is None and "payment_method" not in labelled


@pytest.mark.data
def test_cord_real_data() -> None:
    """Runs on the real CORD v2 parquet export when `make data` could reach the Hub."""
    root = RAW / "cord"
    files = sorted(root.glob("*.parquet"))
    if not files:
        pytest.skip("CORD v2 not downloaded (needs huggingface.co)")
    import pandas as pd

    from receipt_vlm.data.converters.cord import convert_row

    n = 0
    for f in files:
        split = f.stem.split("-")[0]
        for i, row in pd.read_parquet(f).iterrows():
            c = convert_row(row.to_dict(), split, int(i))  # type: ignore[arg-type]
            Receipt.model_validate(c.target.model_dump())
            n += 1
    assert n == 1000
