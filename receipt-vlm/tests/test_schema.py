import json

import pytest
from pydantic import ValidationError

from receipt_vlm.schema import FIELD_ORDER, Item, Receipt, dump_target


def test_full_receipt_roundtrip_and_key_order() -> None:
    r = Receipt(
        merchant="Carrefour",
        date="2024-03-01",
        time="09:05",
        currency="AED",
        items=[Item(name="Milk", quantity=2, unit_price=3.5, total=7.0)],
        subtotal=7.0,
        tax=0.35,
        total=7.35,
        payment_method="card",
        language="en",
    )
    s = r.to_target_json()
    assert list(json.loads(s)) == list(FIELD_ORDER)
    # compact: no spaces, integral floats printed as ints
    assert '"quantity":2,' in s and '"total":7}' in s and '"subtotal":7,' in s and " " not in s
    assert Receipt.model_validate_json(s) == r


def test_empty_receipt_is_valid() -> None:
    r = Receipt.model_validate({})
    assert r.items == [] and r.total is None
    assert json.loads(r.to_target_json())["items"] == []


@pytest.mark.parametrize(
    "bad",
    [
        {"total": "12.50"},  # numbers must be numbers
        {"total": True},
        {"date": "01/03/2024"},
        {"date": "2023-02-30"},  # not a real day
        {"time": "9:05"},
        {"time": "24:00"},
        {"currency": "$"},
        {"currency": "usd"},
        {"language": "eng"},
        {"vendor": "x"},  # unknown key
        {"items": [{"quantity": 1}]},  # item name required
        {"items": [{"name": "a", "price": 1}]},  # unknown item key
        {"merchant": 5},
    ],
)
def test_strict_validation_rejects(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Receipt.model_validate(bad)


def test_int_accepted_for_float() -> None:
    assert Receipt.model_validate({"total": 12}).total == 12.0


def test_dump_target_orders_keys_from_partial_dict() -> None:
    s = dump_target({"total": 3.0, "merchant": "A"})
    assert s.startswith('{"merchant":"A","date":null')
    assert s.endswith('"total":3,"payment_method":null,"language":null}')


def test_non_ascii_is_kept_verbatim() -> None:
    r = Receipt(merchant="كارفور", items=[Item(name="Молоко", total=89.9)])
    s = r.to_target_json()
    assert "كارفور" in s and "Молоко" in s
