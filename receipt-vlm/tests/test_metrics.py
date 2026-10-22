"""Metric tests with hand-computed expected values."""

import json

import numpy as np
import pytest

from receipt_vlm.eval.bootstrap import bootstrap_ci, paired_bootstrap, resample_weights
from receipt_vlm.eval.items_matching import match_items
from receipt_vlm.eval.metrics import (
    ScoreTable,
    compute,
    count_instance,
    is_arith_consistent,
    micro_f1,
    score_sample,
)
from receipt_vlm.eval.normalize import field_equal, norm_str, str_similarity
from receipt_vlm.eval.significance import mcnemar


def _sample(target: dict, labelled: list[str], sid: str = "s1", nospace: bool = False) -> dict:
    base = {
        "merchant": None,
        "date": None,
        "time": None,
        "currency": None,
        "items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
        "payment_method": None,
        "language": None,
    }
    return {
        "id": sid,
        "source": "test",
        "language": "en",
        "target": {**base, **target},
        "labelled": labelled,
        "string_compare": "nospace" if nospace else "default",
    }


GOLD = _sample(
    {
        "merchant": "Green Valley Market",
        "date": "2024-03-05",
        "currency": "USD",
        "items": [
            {"name": "Milk", "quantity": 2, "unit_price": 1.5, "total": 3.0},
            {"name": "Bread", "quantity": 1, "unit_price": None, "total": 2.5},
        ],
        "subtotal": 5.5,
        "total": 5.94,
    },
    [
        "merchant",
        "date",
        "time",
        "currency",
        "subtotal",
        "tax",
        "total",
        "items.name",
        "items.quantity",
        "items.unit_price",
        "items.total",
    ],
)


def test_hand_computed_micro_f1() -> None:
    """Prediction vs GOLD, field by field (only labelled fields count):

    scalar fields
      merchant  "GREEN VALLEY MARKET." vs "Green Valley Market"  -> equal after norm  TP
      date      "2024-03-05" == gold                                                TP
      time      pred null, gold null                                                (nothing)
      currency  pred "EUR" vs "USD"                                                 FP + FN
      subtotal  pred 5.5 == gold                                                    TP
      tax       pred 0.44, gold null (not printed)                                  FP
      total     pred 5.945 vs 5.94 -> |diff| = 0.005 <= 0.01                        TP
    items (Hungarian: Milk<->Milk, Bred<->Bread by similarity 0.8; pred "Eggs" unmatched)
      Milk:  name TP, quantity 2 == 2 TP, unit_price 1.5 TP, total 3.0 TP           4 TP
      Bred:  name "bred" != "bread" FP+FN, quantity 1 TP,
             unit_price pred null & gold null (nothing), total 2.5 TP               2 TP, 1 FP, 1 FN
      Eggs (unmatched prediction): name FP, total FP                                2 FP

    TP = 4 (scalars) + 4 + 2 = 10
    FP = 2 (currency, tax) + 1 + 2 = 5
    FN = 1 (currency) + 1 = 2
    F1 = 2*10 / (2*10 + 5 + 2) = 20 / 27 = 0.7407...
    """
    pred = {
        "merchant": "GREEN VALLEY MARKET.",
        "date": "2024-03-05",
        "time": None,
        "currency": "EUR",
        "items": [
            {"name": "Milk", "quantity": 2, "unit_price": 1.5, "total": 3.0},
            {"name": "Bred", "quantity": 1, "unit_price": None, "total": 2.5},
            {"name": "Eggs", "quantity": None, "unit_price": None, "total": 4.0},
        ],
        "subtotal": 5.5,
        "tax": 0.44,
        "total": 5.945,
        "payment_method": None,
        "language": None,
    }
    s = score_sample(GOLD, json.dumps(pred))
    assert s.status == "ok"
    assert s.f1 == pytest.approx(20 / 27)
    assert s.counts["currency"] == (0, 1, 1)
    assert s.counts["tax"] == (0, 1, 0)
    assert s.counts["items.name"] == (1, 2, 1)
    assert s.total_correct == 1
    # item-level: 1 fully correct (Milk) of 3 predicted and 2 gold
    assert (s.items_tp, s.items_pred, s.items_gold) == (1, 3, 2)
    t = ScoreTable.from_scores([s])
    m = compute(t, ["overall_f1", "item_precision", "item_recall", "item_f1", "total_em"])
    assert m["overall_f1"] == pytest.approx(20 / 27)
    assert m["item_precision"] == pytest.approx(1 / 3)
    assert m["item_recall"] == pytest.approx(1 / 2)
    assert m["item_f1"] == pytest.approx(2 * 1 / (3 + 2))
    assert m["total_em"] == 1.0


def test_unlabelled_fields_never_count() -> None:
    sroie = _sample(
        {"merchant": "ABC SDN BHD", "date": "2018-12-25", "total": 9.0},
        ["merchant", "date", "total"],
    )
    pred = {
        "merchant": "ABC SDN BHD",
        "date": "2018-12-25",
        "total": 9.0,
        "currency": "MYR",
        "items": [{"name": "Tea", "total": 9.0}],
        "subtotal": 9.0,
    }
    s = score_sample(sroie, json.dumps(pred))
    assert s.f1 == 1.0  # currency/items/subtotal are unlabelled for SROIE: no FPs
    assert set(s.counts) == {"merchant", "date", "total"}


def test_failures_score_as_empty_prediction() -> None:
    s = score_sample(GOLD, "Sorry, I cannot read this receipt.")
    assert s.status == "parse_fail" and s.f1 == 0.0
    # schema_fail (total as string) -> strict: empty; lenient: total still recovered
    s = score_sample(GOLD, '```json\n{"total": "5.94", "merchant": "Green Valley Market"}\n```')
    assert s.status == "schema_fail"
    assert s.counts["total"] == (0, 0, 1)
    assert s.lenient_counts["total"] == (1, 0, 0)
    assert s.lenient_counts["merchant"] == (1, 0, 0)


def test_all_null_prediction_scores_zero_on_labelled_sample() -> None:
    empty = {
        k: None
        for k in (
            "merchant",
            "date",
            "time",
            "currency",
            "subtotal",
            "tax",
            "total",
            "payment_method",
            "language",
        )
    }
    s = score_sample(GOLD, json.dumps({**empty, "items": []}))
    assert s.status == "ok" and s.f1 == 0.0
    # ...while correctly abstaining on a receipt where nothing labelled is printed scores 1
    nothing = _sample({}, ["tax", "time"])
    assert score_sample(nothing, json.dumps({**empty, "items": []})).f1 == 1.0


def test_count_instance_rules() -> None:
    assert count_instance("total", None, None) == (0, 0, 0)
    assert count_instance("total", 1.0, None) == (0, 1, 0)
    assert count_instance("total", None, 1.0) == (0, 0, 1)
    assert count_instance("total", 1.009, 1.0) == (1, 0, 0)
    assert count_instance("total", 1.02, 1.0) == (0, 1, 1)
    assert micro_f1({}) == 1.0


def test_normalisation() -> None:
    assert norm_str("  «Магазин  Берёзка»! ") == "магазин  берёзка".replace("  ", " ")
    assert norm_str("CHOEUN KOREAN RESTAURANT", nospace=True) == "choeunkoreanrestaurant"
    assert field_equal(
        "merchant", "Cho Eun Korean Restaurant", "CHOEUN KOREANRESTAURANT", nospace=True
    )
    assert not field_equal("merchant", "Cho Eun Korean Restaurant", "CHOEUN KOREANRESTAURANT")
    assert field_equal("payment_method", "VISA", "card")
    assert field_equal("currency", "usd", "USD")
    assert not field_equal("total", "5.94", 5.94)  # strings are not numbers
    assert str_similarity("bred", "bread") == pytest.approx(0.8)
    assert field_equal("merchant", "ＡＢＣ", "abc")  # NFKC folds full-width letters


def test_hungarian_prefers_global_optimum_and_rejects_unrelated() -> None:
    gold = [{"name": "Chicken Burger", "total": 10.0}, {"name": "Chicken Wings", "total": 8.0}]
    # names point to the swapped pairing (score 0.6 + 0.6 = 1.2); names that are similar
    # (sim ~0.57) *and* matching totals score 0.74 + 0.74 = 1.48, so the global optimum pairs
    # by price -- the assignment is global, not greedy per item.
    pred = [{"name": "Chicken Wings", "total": 10.0}, {"name": "Chicken Burger", "total": 8.0}]
    assert match_items(pred, gold) == [(0, 0), (1, 1)]
    # with clearly different names, name identity wins over price
    pred = [{"name": "Wings", "total": 10.0}, {"name": "Burger", "total": 8.0}]
    gold = [{"name": "Burger", "total": 10.0}, {"name": "Wings", "total": 8.0}]
    assert match_items(pred, gold) == [(1, 0), (0, 1)]
    # an unrelated item with the same price is not matched
    assert (
        match_items(
            [{"name": "Toothpaste", "total": 10.0}], [{"name": "Chicken Burger", "total": 10.0}]
        )
        == []
    )


def test_arith_consistency() -> None:
    assert is_arith_consistent({"items": [{"total": 1.0}, {"total": 2.0}], "subtotal": 3.0}) is True
    assert is_arith_consistent({"items": [{"total": 1.0}], "subtotal": 3.0}) is False
    assert is_arith_consistent({"items": [], "subtotal": 3.0}) is None


def _table(f1_like: list[int], prefix: str = "m") -> ScoreTable:
    scores = []
    for i, ok in enumerate(f1_like):
        g = _sample({"total": 10.0}, ["total"], sid=f"s{i}")
        scores.append(score_sample(g, json.dumps({"total": 10.0 if ok else 11.0})))
    return ScoreTable.from_scores(scores)


def test_bootstrap_ci_contains_point_and_is_reproducible() -> None:
    t = _table([1] * 70 + [0] * 30)
    a = bootstrap_ci(t, ["total_em", "overall_f1"], b=2000, seed=3)
    b = bootstrap_ci(t, ["total_em", "overall_f1"], b=2000, seed=3)
    assert a == b
    ci = a["total_em"]
    assert ci.point == pytest.approx(0.7)
    assert ci.lo < 0.7 < ci.hi
    # binomial SE = sqrt(.7*.3/100) = 0.0458 -> 95% CI half-width ~ 0.09
    assert (ci.hi - ci.lo) / 2 == pytest.approx(0.09, abs=0.015)


def test_resample_weights_rows_sum_to_n() -> None:
    w = resample_weights(50, b=100, seed=1)
    assert w.shape == (100, 50) and np.all(w.sum(1) == 50)


def test_paired_bootstrap_detects_clear_improvement_and_null_case() -> None:
    a = _table([1] * 90 + [0] * 10)
    b = _table([1] * 60 + [0] * 40)
    r = paired_bootstrap(a, b, ["total_em"], b=2000, seed=0)["total_em"]
    assert r.diff == pytest.approx(0.3) and r.significant and r.p_le_0 < 0.01
    same = paired_bootstrap(a, a, ["total_em"], b=500, seed=0)["total_em"]
    assert same.diff == 0 and not same.significant and same.p_le_0 == 1.0


def test_mcnemar_hand_computed() -> None:
    """A right & B wrong on 10 images, the reverse on 2, both right/wrong elsewhere.
    Exact two-sided binomial p = 2 * P(X <= 2 | n=12, p=.5) = 2 * (1+12+66)/4096 = 0.03857.
    """
    a = [1] * 10 + [0] * 2 + [1] * 50 + [0] * 5
    b = [0] * 10 + [1] * 2 + [1] * 50 + [0] * 5
    r = mcnemar(a, b)
    assert (r.a_only, r.b_only, r.exact) == (10, 2, True)
    assert r.p_value == pytest.approx(2 * 79 / 4096)
    assert mcnemar([1, 0, None], [1, 0, 1]).n == 2
