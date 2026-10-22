"""Text -> typed value parsing shared by the dataset converters and the lenient eval path.

One implementation, so "what counts as the same date/amount" is identical in data and eval.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Final, Literal

DateOrder = Literal["DMY", "MDY", "YMD"]

# Symbol / token -> ISO 4217. Used by converters only (CLAUDE.md §5.2: not in eval).
CURRENCY_SYMBOLS: Final[dict[str, str]] = {
    "د.إ": "AED",
    "درهم": "AED",
    "aed": "AED",
    "dhs": "AED",
    "€": "EUR",
    "eur": "EUR",
    "₽": "RUB",
    "руб": "RUB",
    "rub": "RUB",
    "р.": "RUB",
    "$": "USD",
    "usd": "USD",
    "rm": "MYR",
    "myr": "MYR",
    "rp": "IDR",
    "idr": "IDR",
}

_MONTHS: Final[dict[str, int]] = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january", "янв"),
            ("feb", "february", "фев"),
            ("mar", "march", "мар"),
            ("apr", "april", "апр"),
            ("may", "мая", "май"),
            ("jun", "june", "июн"),
            ("jul", "july", "июл"),
            ("aug", "august", "авг"),
            ("sep", "sept", "september", "сен"),
            ("oct", "october", "окт"),
            ("nov", "november", "ноя"),
            ("dec", "december", "дек"),
        ],
        start=1,
    )
    for m in names
}

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")


def to_ascii_digits(s: str) -> str:
    """Arabic-Indic / Persian digits and separators -> ASCII."""
    return s.translate(_ARABIC_INDIC)


def parse_amount(text: str | None, thousands_dot: bool = False) -> float | None:
    """Parse a printed money amount into a float.

    ``thousands_dot=True`` means a lone ``.`` or ``,`` followed by exactly three digits is a
    thousands separator (Indonesian ``60.000`` = 60000); otherwise it is a decimal point.
    Returns None if no number can be read.
    """
    if text is None:
        return None
    s = to_ascii_digits(str(text)).strip()
    if not s:
        return None
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")")) or s.endswith("-")
    # keep digits and separators only
    s = re.sub(r"[^\d.,\s']", "", s)
    s = re.sub(r"(?<=\d)[\s'](?=\d{3}\b)", "", s)  # "1 234" / "1'234" -> "1234"
    s = s.replace(" ", "").replace("'", "")
    s = s.rstrip(".,")
    if s[:1] in (".", ","):
        s = "0" + s  # ".49" -> "0.49"
    if not s or not re.search(r"\d", s):
        return None
    if not re.fullmatch(r"[\d.,]+", s):
        return None
    dots, commas = s.count("."), s.count(",")
    if dots and commas:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thou = "," if dec == "." else "."
        s = s.replace(thou, "").replace(dec, ".")
    elif dots + commas >= 1:
        sep = "." if dots else ","
        parts = s.split(sep)
        if len(parts) > 2:
            if all(len(p) == 3 for p in parts[1:]):
                s = "".join(parts)  # 1.234.567
            else:
                return None
        else:
            frac = parts[1]
            if len(frac) == 3 and thousands_dot:
                s = "".join(parts)
            else:
                s = parts[0] + "." + frac if frac else parts[0]
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def _mk_date(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(text: str | None, order: DateOrder = "DMY") -> str | None:
    """Parse a printed date into ISO ``YYYY-MM-DD``.

    Numeric dates are read in ``order`` unless that is impossible (e.g. a 13+ in the month
    slot), in which case the other day/month order is tried. Returns None when unreadable.
    """
    if not text:
        return None
    s = to_ascii_digits(str(text)).strip().lower()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _mk_date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"\D*(\d{8})\D*", s)  # compact 20180304 / 25032018
    if m:
        d8 = m[1]
        if d8[:2] in ("19", "20"):
            out = _mk_date(int(d8[:4]), int(d8[4:6]), int(d8[6:]))
            if out:
                return out
        s = f"{d8[:2]}/{d8[2:4]}/{d8[4:]}"
    m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)", s)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        first = (b, a) if order == "DMY" else (a, b)  # (month, day)
        for mo, d in (first, first[::-1]):
            out = _mk_date(y, mo, d)
            if out:
                return out
        return None
    # textual months: "25 Dec 2018", "Dec 25, 2018", "25-DEC-18", "25 декабря 2018"
    m = re.search(r"(\d{1,2})[\s\-./]*([a-zа-я]{3,9})\.?[\s\-./,]*(\d{4}|\d{2})(?!\d)", s)
    if m and m[2][:3] in _MONTHS:
        return _mk_date(int(m[3]), _MONTHS[m[2][:3]], int(m[1]))
    m = re.search(r"([a-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4}|\d{2})(?!\d)", s)
    if m and m[1][:3] in _MONTHS:
        return _mk_date(int(m[3]), _MONTHS[m[1][:3]], int(m[2]))
    return None


def parse_time(text: str | None) -> str | None:
    """Parse a printed time into 24h ``HH:MM`` (seconds dropped)."""
    if not text:
        return None
    s = to_ascii_digits(str(text)).strip().lower()
    m = re.search(r"(\d{1,2})[:.](\d{2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)?", s)
    if not m:
        return None
    h, mi = int(m[1]), int(m[2])
    ampm = (m[3] or "").replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return f"{h:02d}:{mi:02d}"


def currency_from_text(text: str | None) -> str | None:
    """Map a currency symbol/word found in text to ISO 4217 (converters only)."""
    if not text:
        return None
    s = text.lower()
    for sym in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
        if sym.isalpha():
            if re.search(rf"(?<![a-zа-я]){re.escape(sym)}(?![a-zа-я])", s):
                return CURRENCY_SYMBOLS[sym]
        elif sym in s:
            return CURRENCY_SYMBOLS[sym]
    return None
