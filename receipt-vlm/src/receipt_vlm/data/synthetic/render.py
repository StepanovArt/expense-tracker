"""Draw a ReceiptSpec onto a clean paper image.

Arabic text is shaped with ``arabic-reshaper`` and put into visual (right-to-left) order with
``python-bidi``; the layout is mirrored (labels and item names on the right, amounts on the
left). Text is drawn with Pillow's BASIC layout engine so the already-shaped glyphs are not
shaped a second time when libraqm happens to be installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import arabic_reshaper
import numpy as np
from bidi import get_display
from PIL import Image, ImageDraw

from receipt_vlm.data.synthetic.fonts import FontChoice, available, covers, pick_font
from receipt_vlm.data.synthetic.templates import ItemLine, ReceiptSpec

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_RESHAPER = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": False})

Kind = Literal["center", "kv", "start", "sep", "blank", "barcode"]


@dataclass
class Line:
    kind: Kind
    a: str = ""
    b: str = ""
    scale: float = 1.0
    bold: bool = False


def visual(text: str, rtl: bool) -> str:
    """Logical -> visual order. Arabic is reshaped (contextual forms) and reordered RTL."""
    if not text:
        return text
    has_arabic = bool(_ARABIC.search(text))
    if not has_arabic and not rtl:
        return text
    shaped = _RESHAPER.reshape(text) if has_arabic else text
    return str(get_display(shaped, base_dir="R" if rtl else "L"))


_WEIGHT_UNIT = {"en": "kg", "ru": "кг", "ar": "كغ"}
_QTY_ONLY = {
    "en": ["x{q}", "{q} pcs", "QTY {q}"],
    "ru": ["{q} шт", "x{q}"],
    "ar": ["{q} ×", "عدد {q}"],
}


def _qty(it: ItemLine, spec: ReceiptSpec) -> str:
    q = f"{it.quantity:.3f}" if it.by_weight else f"{int(it.quantity)}"
    if it.by_weight and spec.style.decimal == ",":
        q = q.replace(".", ",")
    return spec.style.number(q)


def _item_lines(spec: ReceiptSpec, rng: np.random.Generator) -> list[Line]:
    fmt_by_layout = {
        "supermarket": ["two_line", "detail_below"],
        "restaurant": ["qty_prefix", "two_line"],
        "pharmacy": ["detail_below", "two_line"],
    }
    fmt = fmt_by_layout[spec.layout][int(rng.integers(2))]
    times = "×" if spec.lang == "ar" or rng.random() < 0.3 else "x"
    qty_only = _QTY_ONLY[spec.lang][int(rng.integers(len(_QTY_ONLY[spec.lang])))]
    out: list[Line] = []
    for it in spec.items:
        tot = spec.style.amount(it.total)
        q = _qty(it, spec)
        unit = f" {_WEIGHT_UNIT[spec.lang]}" if it.by_weight else ""
        detail = ""
        if it.print_unit_price:
            detail = f"{q}{unit} {times} {spec.style.amount(it.unit_price)}"
        elif it.print_quantity:
            detail = qty_only.replace("{q}", q) if not it.by_weight else f"{q}{unit}"
        if fmt == "qty_prefix" and it.print_quantity and not it.by_weight:
            out.append(Line("kv", f"{q}  {it.name}", tot))
            if it.print_unit_price:
                out.append(Line("start", f"    @ {spec.style.amount(it.unit_price)}"))
        elif fmt == "detail_below" and detail:
            out.append(Line("start", it.name))
            out.append(Line("kv", f"    {detail}", tot))
        else:
            out.append(Line("kv", it.name, tot))
            if detail:
                out.append(Line("start", f"    {detail}"))
    return out


def build_lines(spec: ReceiptSpec, rng: np.random.Generator) -> list[Line]:
    st, lab = spec.style, spec.labels
    sep = ["-", "=", "*", "."][int(rng.integers(4))]
    lines = [Line("center", spec.merchant, scale=float(rng.uniform(1.25, 1.6)), bold=True)]
    lines.append(Line("center", spec.address))
    if spec.tax_id:
        lines.append(Line("center", spec.tax_id))
    if spec.header:
        lines += [Line("blank"), Line("center", spec.header, bold=True)]
    lines.append(Line("sep", sep))

    if rng.random() < 0.5 or spec.time_text is None:
        lines.append(Line("kv", lab["date"] or " ", spec.date_text))
        if spec.time_text is not None:
            lines.append(Line("kv", lab["time"] or " ", spec.time_text))
    else:
        lines.append(Line("start", f"{spec.date_text}   {spec.time_text}"))
    lines.append(Line("start", spec.receipt_no))
    if rng.random() < 0.7:
        lines.append(Line("start", spec.cashier))
    lines.append(Line("sep", sep))
    if rng.random() < 0.5:
        lines.append(Line("kv", lab["qty_hdr"], lab["amount_hdr"], bold=True))
    lines += _item_lines(spec, rng)
    lines.append(Line("sep", sep))

    eq = "=" if spec.lang == "ru" and rng.random() < 0.4 else ""
    if spec.print_subtotal:
        lines.append(Line("kv", lab["subtotal"], st.amount(spec.subtotal)))
    if spec.print_tax:
        lines.append(Line("kv", lab["tax"], st.amount(spec.tax)))
    lines.append(
        Line(
            "kv", lab["total"], eq + st.amount(spec.total, with_currency=True), scale=1.2, bold=True
        )
    )
    if spec.payment_text is not None:
        if rng.random() < 0.5:
            lines.append(Line("kv", spec.labels["payment"], spec.payment_text))
        else:
            lines.append(Line("start", spec.payment_text))
        if spec.tendered is not None:
            lines.append(Line("kv", lab["tendered"], st.amount(spec.tendered)))
            lines.append(Line("kv", lab["change"], st.amount(round(spec.tendered - spec.total, 2))))
    lines += [Line("sep", sep), Line("center", spec.footer)]
    if rng.random() < 0.5:
        lines.append(Line("barcode"))
    return lines


def _texts(lines: list[Line], rtl: bool) -> list[str]:
    return [visual(t, rtl) for ln in lines for t in (ln.a, ln.b) if t and ln.kind != "sep"]


def choose_font(
    spec: ReceiptSpec, lines: list[Line], rng: np.random.Generator
) -> tuple[str, FontChoice]:
    names = available(spec.lang)
    order = rng.permutation(len(names))
    texts = "".join(_texts(lines, spec.lang == "ar"))
    for i in order:
        fc = pick_font(names[int(i)])
        if covers(fc.regular, texts) and covers(fc.bold, texts):
            return names[int(i)], fc
    fallback = "DejaVuSans.ttf" if spec.lang == "ar" else "DejaVuSansMono.ttf"
    return fallback, pick_font(fallback)


def render(spec: ReceiptSpec, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    rtl = spec.lang == "ar"
    lines = build_lines(spec, rng)
    font_name, fc = choose_font(spec, lines, rng)
    width = int(rng.integers(500, 660))
    margin = int(rng.integers(16, 30))
    size = int(rng.integers(18, 25))
    gap = 12

    def line_width(ln: Line, sz: int) -> float:
        f = fc.get(max(8, round(sz * ln.scale)), ln.bold)
        if ln.kind == "kv":
            return f.getlength(visual(ln.a, rtl)) + gap + f.getlength(visual(ln.b, rtl))
        if ln.kind in ("center", "start"):
            return f.getlength(visual(ln.a, rtl))
        return 0.0

    inner = width - 2 * margin
    while size > 12 and max(line_width(ln, size) for ln in lines) > inner:
        size -= 1
    need = max(line_width(ln, size) for ln in lines)
    if need > inner:  # still too wide at the minimum size: widen the paper instead
        width = int(need + 2 * margin + 1)
        inner = width - 2 * margin

    heights = [
        0.6 * size
        if ln.kind == "blank"
        else 70
        if ln.kind == "barcode"
        else round(size * ln.scale * 1.38)
        for ln in lines
    ]
    height = int(sum(heights) + 2 * margin + rng.integers(10, 60))
    # thermal paper: near-white, at most slightly warm (never pink/green/blue)
    base = int(rng.integers(228, 253))
    paper = (min(255, base + int(rng.integers(0, 4))), base, base - int(rng.integers(0, 10)))
    ink_level = int(rng.integers(0, 80))
    ink = (ink_level, ink_level, ink_level + int(rng.integers(0, 20)))
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)

    y = float(margin)
    for ln, h in zip(lines, heights, strict=True):
        f = fc.get(max(8, round(size * ln.scale)), ln.bold)
        a, b = visual(ln.a, rtl), visual(ln.b, rtl)
        if ln.kind == "center":
            draw.text(((width - f.getlength(a)) / 2, y), a, font=f, fill=ink)
        elif ln.kind == "start":
            x = width - margin - f.getlength(a) if rtl else margin
            draw.text((x, y), a, font=f, fill=ink)
        elif ln.kind == "kv":
            if rtl:  # label right, value left
                draw.text((width - margin - f.getlength(a), y), a, font=f, fill=ink)
                draw.text((margin, y), b, font=f, fill=ink)
            else:
                draw.text((margin, y), a, font=f, fill=ink)
                draw.text((width - margin - f.getlength(b), y), b, font=f, fill=ink)
        elif ln.kind == "sep":
            n = int(inner / max(f.getlength(ln.a), 1))
            draw.text((margin, y), ln.a * n, font=f, fill=ink)
        elif ln.kind == "barcode":
            x = margin + inner * 0.15
            while x < width - margin - inner * 0.15:
                w = float(rng.choice([1.5, 2.5, 3.5]))
                if rng.random() < 0.55:
                    draw.rectangle([x, y + 5, x + w, y + 60], fill=ink)
                x += w + float(rng.choice([1.5, 2.5]))
        y += h
    meta = {"font": font_name, "font_size": size, "paper_width": width}
    return img, meta
