"""GRPO reward tests, including adversarial outputs (CLAUDE.md §6.2)."""

import json

import pytest

from receipt_vlm.rewards.grpo_rewards import completion_text, compute_reward, make_trl_reward_funcs

KEYS = (
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
ALL_LABELS = [*KEYS, "items.name", "items.quantity", "items.unit_price", "items.total"]


def _sample(target: dict, labelled: list[str]) -> dict:
    return {
        "id": "x",
        "target": {**dict.fromkeys(KEYS), "items": [], **target},
        "labelled": labelled,
    }


SYN = _sample(
    {
        "merchant": "FRESHMART MARKET",
        "date": "2024-03-05",
        "currency": "USD",
        "language": "en",
        "items": [
            {"name": "Milk", "quantity": 2, "unit_price": 1.5, "total": 3.0},
            {"name": "Bread", "quantity": None, "unit_price": None, "total": 2.5},
        ],
        "subtotal": 5.5,
        "tax": 0.44,
        "total": 5.94,
        "payment_method": "card",
    },
    ALL_LABELS,
)


def _j(d: dict) -> str:
    return json.dumps({**dict.fromkeys(KEYS), "items": [], **d})


def test_perfect_prediction_gets_full_reward() -> None:
    r = compute_reward(json.dumps(SYN["target"]), SYN)
    assert (r.format, r.fields, r.total, r.consistency, r.reward) == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_all_null_prediction_scores_zero_fields() -> None:
    r = compute_reward(_j({}), SYN)
    assert r.format == 1.0 and r.fields == 0.0 and r.total == 0.0
    assert r.consistency == 0.0  # dropping the numbers is not "not applicable"
    assert r.reward == pytest.approx(0.2)


@pytest.mark.parametrize(
    "bad", ["", "{}", "null", "[]", "I can't read it", '{"total": 5.94', "```json\n```"]
)
def test_garbage_and_empty_outputs_score_low(bad: str) -> None:
    r = compute_reward(bad, SYN)
    assert r.reward <= 0.2 + 1e-9 and r.fields == 0.0


def test_empty_object_is_schema_valid_but_worthless() -> None:
    # "{}" validates (all keys default to null) -> only the format component
    r = compute_reward("{}", SYN)
    assert r.status == "ok" and r.reward == pytest.approx(0.2)


def test_garbage_gets_no_abstention_credit_on_all_null_gold() -> None:
    nothing = _sample({}, ["tax", "time"])  # nothing labelled is printed
    assert compute_reward("garbage", nothing).reward == 0.0
    assert compute_reward(_j({}), nothing).reward == 1.0  # correct abstention, valid JSON


def test_schema_invalid_gets_no_field_reward_even_if_values_right() -> None:
    bad = json.dumps({**SYN["target"], "total": "5.94"})  # number as string
    r = compute_reward(bad, SYN)
    assert r.status == "schema_fail" and r.format == 0.0 and r.fields == 0.0 and r.reward == 0.0


def test_partial_credit_is_monotone() -> None:
    wrong_total = {**SYN["target"], "total": 6.0}
    hallucinated = {**SYN["target"], "time": "12:00"}
    full = compute_reward(json.dumps(SYN["target"]), SYN).reward
    r1 = compute_reward(json.dumps(wrong_total), SYN)
    r2 = compute_reward(json.dumps(hallucinated), SYN)
    assert r1.total == 0.0 and r1.consistency == 0.0 and r1.reward < r2.reward < full


def test_total_not_applicable_redistributes_weight() -> None:
    s = _sample({"merchant": "A"}, ["merchant"])
    r = compute_reward(_j({"merchant": "A"}), s)
    assert r.total is None and r.consistency is None and r.reward == 1.0
    r = compute_reward(_j({}), s)
    # format 0.2 + fields 0 over active weight 0.7
    assert r.reward == pytest.approx(0.2 / 0.7)


def test_inconsistent_gold_does_not_reward_consistency() -> None:
    cord_like = _sample(
        {"items": [{"name": "A", "total": 10.0}], "subtotal": 10.0, "total": 11.5},
        ["items.name", "items.total", "subtotal", "tax", "total"],
    )  # service charge
    r = compute_reward(json.dumps(cord_like["target"]), cord_like)
    assert r.consistency is None and r.reward == 1.0


def test_trl_wrapper_and_conversational_completions() -> None:
    (fn,) = make_trl_reward_funcs()
    comps = [[{"role": "assistant", "content": json.dumps(SYN["target"])}], "not json"]
    out = fn(
        prompts=[None, None],
        completions=comps,
        id=["x", "x"],
        target=[SYN["target"]] * 2,
        labelled=[SYN["labelled"]] * 2,
    )
    assert out == [1.0, 0.0]
    assert (
        completion_text([{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}])
        == "hi"
    )
