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

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    pred_null: dict[str, int] = field(default_factory=dict)  # labelled fields predicted null
    n_chars: int = 0


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
    scored = score_sample(sample, pr)
    r_fields = micro_f1(scored.counts) if pr.status == "ok" else 0.0
    r_total = _total_component(pr, sample)
    r_cons = _consistency_component(pr, sample)
    parts = {"format": r_format, "fields": r_fields, "total": r_total, "consistency": r_cons}
    active = {k: v for k, v in parts.items() if v is not None}
    wsum = sum(weights[k] for k in active)
    reward = sum(weights[k] * v for k, v in active.items()) / wsum
    return RewardBreakdown(
        r_format,
        r_fields,
        r_total,
        r_cons,
        float(reward),
        pr.status,
        scored.pred_null,
        len(completion),
    )


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


def _decode(v: Any) -> Any:
    """Dataset columns may hold JSON strings (avoids Arrow's nested-type inference)."""
    return json.loads(v) if isinstance(v, str) else v


def make_trl_reward_funcs(weights: Mapping[str, float] = WEIGHTS) -> list[Any]:
    """Reward function in TRL's ``reward_funcs`` signature.

    The dataset must carry ``id``, ``target`` and ``labelled`` columns (dicts/lists or JSON
    strings), which TRL forwards as kwargs. When TRL passes ``log_metric``, the mean of each
    reward component, the schema-valid rate, the mean output length and the per-field rate of
    null predictions (reward-hacking monitor, CLAUDE.md §6.2) are logged with every batch.
    """

    def receipt_reward(prompts: list[Any], completions: list[Any], **kw: Any) -> list[float]:
        out: list[RewardBreakdown] = []
        n = len(completions)
        compare = kw.get("string_compare") or ["default"] * n
        for i, c in enumerate(completions):
            sample = {
                "id": kw["id"][i],
                "target": _decode(kw["target"][i]),
                "labelled": _decode(kw["labelled"][i]),
                "string_compare": compare[i],
            }
            out.append(compute_reward(completion_text(c), sample, weights))
        log = kw.get("log_metric")
        if callable(log) and out:
            log("reward/format", sum(b.format for b in out) / n)
            log("reward/fields", sum(b.fields for b in out) / n)
            tot = [b.total for b in out if b.total is not None]
            if tot:
                log("reward/total", sum(tot) / len(tot))
            cons = [b.consistency for b in out if b.consistency is not None]
            if cons:
                log("reward/consistency", sum(cons) / len(cons))
            log("schema_valid_rate", sum(b.status == "ok" for b in out) / n)
            log("completion_chars", sum(b.n_chars for b in out) / n)
            keys = {k for b in out for k in b.pred_null}
            for k in sorted(keys):
                vals = [b.pred_null[k] for b in out if k in b.pred_null]
                log(f"null_rate/{k}", sum(vals) / len(vals))
            all_null = [v for b in out for v in b.pred_null.values()]
            if all_null:
                log("null_rate/all", sum(all_null) / len(all_null))
        return [b.reward for b in out]

    receipt_reward.__name__ = "receipt_reward"
    return [receipt_reward]
