"""GRPO reward = 0.2 r_format + 0.5 r_fields + 0.2 r_total + 0.1 r_consistency (CLAUDE.md §6.2).

Every component reuses the evaluation code (parse_prediction, score_sample, micro_f1,
is_fully_consistent), so the reward optimises exactly what we report. Guards against
reward hacking:

* r_fields is 0 for anything that is not schema-valid (a garbage string must not collect
  "correct abstention" credit on a sample whose labelled fields are all null);
* r_fields of an all-null prediction on a sample with labelled non-null gold is 0
  (micro F1 with TP = 0);
* r_total: when total is labelled, null-vs-value is scored like any field; when total is not
  labelled the component is *not applicable* and its weight is redistributed (no free points);
* r_consistency only counts when the gold receipt is itself consistent (real receipts with
  service charges / discounts are not); a prediction that omits the numbers the gold has gets
  0, not "not applicable" - so dropping fields never earns consistency reward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from receipt_vlm.eval.metrics import is_fully_consistent, micro_f1, score_sample
from receipt_vlm.eval.normalize import num_equal
from receipt_vlm.eval.parse import ParseResult, parse_prediction

WEIGHTS: Final[dict[str, float]] = {"format": 0.2, "fields": 0.5, "total": 0.2, "consistency": 0.1}


@dataclass(frozen=True)
class RewardBreakdown:
    format: float
    fields: float
    total: float | None  # None = not applicable
    consistency: float | None
    reward: float
    status: str


def _total_component(pr: ParseResult, sample: Mapping[str, Any]) -> float | None:
    if "total" not in sample["labelled"]:
        return None
    if pr.status != "ok":
        return 0.0
    gold = sample["target"].get("total")
    pred = pr.strict_dict.get("total")
    if gold is None:
        return 1.0 if pred is None else 0.0
    return 1.0 if num_equal(pred, gold) else 0.0


def _consistency_component(pr: ParseResult, sample: Mapping[str, Any]) -> float | None:
    gold = sample["target"]
    gold_c = is_fully_consistent(gold)
    if gold_c is not True:  # gold not checkable or genuinely inconsistent -> do not reward it
        return None
    if pr.status != "ok":
        return 0.0
    pred_c = is_fully_consistent(pr.strict_dict)
    if pred_c is None:  # the prediction dropped the numbers the gold receipt has
        return 0.0
    return 1.0 if pred_c else 0.0


def compute_reward(
    completion: str, sample: Mapping[str, Any], weights: Mapping[str, float] = WEIGHTS
) -> RewardBreakdown:
    pr = parse_prediction(completion)
    r_format = 1.0 if pr.status == "ok" else 0.0
    r_fields = micro_f1(score_sample(sample, pr).counts) if pr.status == "ok" else 0.0
    r_total = _total_component(pr, sample)
    r_cons = _consistency_component(pr, sample)
    parts = {"format": r_format, "fields": r_fields, "total": r_total, "consistency": r_cons}
    active = {k: v for k, v in parts.items() if v is not None}
    wsum = sum(weights[k] for k in active)
    reward = sum(weights[k] * v for k, v in active.items()) / wsum
    return RewardBreakdown(r_format, r_fields, r_total, r_cons, float(reward), pr.status)


def completion_text(completion: Any) -> str:
    """TRL passes plain strings, or (conversational) a list of message dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion and isinstance(completion[-1], Mapping):
        content = completion[-1].get("content", "")
        if isinstance(content, list):
            return "".join(str(c.get("text", "")) for c in content if isinstance(c, Mapping))
        return str(content)
    return str(completion)


def make_trl_reward_funcs(weights: Mapping[str, float] = WEIGHTS) -> list[Any]:
    """Reward functions in TRL's ``reward_funcs`` signature.

    Returns the combined reward plus per-component logging functions with zero weight are
    *not* used; instead the trainer logs components itself (see train/grpo.py). The dataset
    must carry ``target``, ``labelled``, ``id`` columns, which TRL forwards as kwargs.
    """

    def receipt_reward(prompts: list[Any], completions: list[Any], **kw: Any) -> list[float]:
        out = []
        for i, c in enumerate(completions):
            sample = {
                "id": kw["id"][i],
                "target": kw["target"][i],
                "labelled": kw["labelled"][i],
                "string_compare": (kw.get("string_compare") or ["default"] * len(completions))[i],
            }
            out.append(compute_reward(completion_text(c), sample, weights).reward)
        return out

    receipt_reward.__name__ = "receipt_reward"
    return [receipt_reward]
