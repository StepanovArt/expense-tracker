"""Synthetic generator: exact, schema-valid, arithmetically consistent, correctly shaped Arabic."""

import hashlib
import io

import numpy as np
import pytest

from receipt_vlm.data.synthetic.generate import LANGS, LAYOUTS, is_consistent, spec_for
from receipt_vlm.data.synthetic.render import build_lines, render, visual
from receipt_vlm.schema import Item, Receipt

N_SPECS = 600


@pytest.fixture(scope="module")
def specs():  # type: ignore[no-untyped-def]
    return [spec_for(i, seed=99)[0] for i in range(N_SPECS)]


def test_ground_truth_always_validates(specs) -> None:  # type: ignore[no-untyped-def]
    for s in specs:
        gold = s.gold()
        assert Receipt.model_validate_json(gold.to_target_json()) == gold


def test_arithmetic_always_consistent(specs) -> None:  # type: ignore[no-untyped-def]
    for s in specs:
        assert round(sum(i.total for i in s.items), 2) == pytest.approx(s.subtotal, abs=0.005)
        assert round(s.subtotal + s.tax, 2) == pytest.approx(s.total, abs=0.005)
        assert is_consistent(s.gold())


def test_grid_is_balanced(specs) -> None:  # type: ignore[no-untyped-def]
    cells = {(s.lang, s.layout) for s in specs}
    assert cells == {(lang, layout) for lang in LANGS for layout in LAYOUTS}


def test_currency_and_coverage(specs) -> None:  # type: ignore[no-untyped-def]
    currencies = {s.currency for s in specs}
    assert currencies == {"AED", "USD", "RUB", "EUR"}
    golds = [s.gold() for s in specs]
    # null-by-design fields appear in both states so the model must read, not guess
    assert {g.time is None for g in golds} == {True, False}
    assert {g.subtotal is None for g in golds} == {True, False}
    assert {g.payment_method for g in golds} >= {"cash", "card", "mobile", "voucher", None}
    assert any(
        i.quantity is not None and not float(i.quantity).is_integer()
        for g in golds
        for i in g.items
    )


def test_is_consistent_detects_errors() -> None:
    ok = Receipt(
        items=[Item(name="a", quantity=2, unit_price=1.5, total=3.0)],
        subtotal=3.0,
        tax=0.15,
        total=3.15,
    )
    assert is_consistent(ok)
    assert not is_consistent(ok.model_copy(update={"total": 3.20}))
    assert not is_consistent(ok.model_copy(update={"subtotal": 2.0}))
    bad_item = ok.model_copy(
        update={"items": [Item(name="a", quantity=2, unit_price=1.5, total=4.0)]}
    )
    assert not is_consistent(bad_item)


def _printed(i: int):  # type: ignore[no-untyped-def]
    spec, g = spec_for(i, seed=7)
    text = "\n".join(f"{ln.a} {ln.b}" for ln in build_lines(spec, g))
    return spec, text


@pytest.mark.parametrize("i", range(18))
def test_every_gold_value_is_printed(i: int) -> None:
    """Gold JSON == what is on the paper: each non-null value appears in the printed text."""
    spec, text = _printed(i)
    st = spec.style
    gold = spec.gold()
    assert spec.merchant in text
    assert spec.date_text in text
    assert st.amount(spec.total) in text
    assert st.currency_token in text
    if gold.time is not None:
        assert spec.time_text in text
    if gold.subtotal is not None:
        assert st.amount(spec.subtotal) in text
    if gold.tax is not None:
        assert st.amount(spec.tax) in text
    if gold.payment_method is not None:
        assert spec.payment_text in text
    for it, git in zip(spec.items, gold.items, strict=True):
        assert it.name in text and st.amount(it.total) in text
        if git.unit_price is not None:
            assert st.amount(it.unit_price) in text


def test_arabic_is_shaped_and_right_to_left() -> None:
    logical = "حليب طازج"
    vis = visual(logical, rtl=True)
    # contextual presentation forms (U+FE70-U+FEFF), no bare isolated letters left
    assert all(0xFE70 <= ord(c) <= 0xFEFF or c == " " for c in vis)
    # the first logical word ends up at the visual end (right-to-left order)
    words = vis.split(" ")
    assert len(words) == 2 and len(words[1]) == 4 and len(words[0]) == 4  # "حليب" -> 4 glyphs
    assert visual("Total 12.50", rtl=False) == "Total 12.50"


@pytest.mark.parametrize("lang_index", [0, 1, 2])
def test_render_is_deterministic_and_uses_covering_font(lang_index: int) -> None:
    from receipt_vlm.data.synthetic.fonts import covers

    spec_a, g_a = spec_for(lang_index, seed=5)
    spec_b, g_b = spec_for(lang_index, seed=5)
    img_a, meta = render(spec_a, g_a)
    img_b, _ = render(spec_b, g_b)

    def digest(im) -> str:  # type: ignore[no-untyped-def]
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    assert digest(img_a) == digest(img_b)
    assert np.asarray(img_a).std() > 10  # something was drawn
    from receipt_vlm.data.synthetic.fonts import font_path

    path = font_path(str(meta["font"]))
    assert path is not None
    assert covers(
        path,
        "".join(
            visual(t, spec_a.lang == "ar")
            for t in [spec_a.merchant] + [i.name for i in spec_a.items]
        ),
    )
