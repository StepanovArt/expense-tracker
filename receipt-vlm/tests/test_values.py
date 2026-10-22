import pytest

from receipt_vlm.values import currency_from_text, parse_amount, parse_date, parse_time


@pytest.mark.parametrize(
    ("text", "kw", "expected"),
    [
        ("9.00", {}, 9.0),
        ("RM 41.45", {}, 41.45),
        ("$13.99", {}, 13.99),
        ("￥17.99", {}, 17.99),
        ("1,234.56", {}, 1234.56),
        ("1.234,56", {}, 1234.56),
        ("1 234,56 ₽", {}, 1234.56),
        ("12,5", {}, 12.5),
        ("-1.73", {}, -1.73),
        ("1.80-", {}, -1.80),
        (".49", {}, 0.49),
        ("60.000", {"thousands_dot": True}, 60000.0),
        ("17,000", {"thousands_dot": True}, 17000.0),
        ("1.250.000", {}, 1250000.0),
        ("٣٤٫٥٠ د.إ", {}, 34.5),
        ("", {}, None),
        ("abc", {}, None),
        (None, {}, None),
    ],
)
def test_parse_amount(text: str | None, kw: dict[str, bool], expected: float | None) -> None:
    assert parse_amount(text, **kw) == expected


@pytest.mark.parametrize(
    ("text", "order", "expected"),
    [
        ("25/12/2018", "DMY", "2018-12-25"),
        ("11-03-18", "DMY", "2018-03-11"),
        ("11-03-18", "MDY", "2018-11-03"),
        ("12/30/2016FRI", "MDY", "2016-12-30"),
        ("12/30/2016", "DMY", "2016-12-30"),  # day-first impossible -> month-first
        ("2018-03-12", "DMY", "2018-03-12"),
        ("10 MAR 2018", "DMY", "2018-03-10"),
        ("25-DEC-18", "DMY", "2018-12-25"),
        ("OCT 9, 2017", "DMY", "2017-10-09"),
        ("(12/01/2018)", "DMY", "2018-01-12"),
        ("20180304", "DMY", "2018-03-04"),
        ("25032018", "DMY", "2018-03-25"),
        ("05 марта 2024", "DMY", "2024-03-05"),
        ("31/02/2018", "DMY", None),
        ("no date", "DMY", None),
    ],
)
def test_parse_date(text: str, order: str, expected: str | None) -> None:
    assert parse_date(text, order=order) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("19:19", "19:19"),
        ("2:05 PM", "14:05"),
        ("12:30 am", "00:30"),
        ("10:23:59", "10:23"),
        ("25:00", None),
        ("", None),
    ],
)
def test_parse_time(text: str, expected: str | None) -> None:
    assert parse_time(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Total: 34.50 د.إ", "AED"),
        ("ИТОГО 120 руб.", "RUB"),
        ("12,00 €", "EUR"),
        ("TOTAL RM 9.00", "MYR"),
        ("charm 5", None),  # 'rm' inside a word is not ringgit
    ],
)
def test_currency_from_text(text: str, expected: str | None) -> None:
    assert currency_from_text(text) == expected
