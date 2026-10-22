"""Per-sample scoring and corpus metrics (CLAUDE.md §5.3).

Counting rule for one labelled field instance with gold ``g`` and prediction ``p``:

====================  =====================
case                  counts
====================  =====================
g null, p null        nothing (correct abstention)
g null, p set         FP
g set,  p null        FN
g set,  p == g        TP
g set,  p != g        FP + FN
====================  =====================

Item attributes (``items.name``, ``items.quantity``, ``items.unit_price``, ``items.total``)
are counted the same way over Hungarian-matched pairs; attributes of unmatched gold items
are FNs and of unmatched predicted items FPs. Unlabelled fields are skipped entirely.

Every sample is reduced to a vector of additive counts, so each corpus metric is a function
of column sums. That makes bootstrap resampling a single matrix product.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from receipt_vlm.eval.items_matching import match_items
from receipt_vlm.eval.normalize import field_equal, str_equal, str_similarity
from receipt_vlm.eval.parse import ParseResult, parse_prediction
from receipt_vlm.schema import ITEM_FIELDS, SCALAR_FIELDS

ITEM_KEYS: tuple[str, ...] = tuple(f"items.{a}" for a in ITEM_FIELDS)
COUNT_KEYS: tuple[str, ...] = (*SCALAR_FIELDS, *ITEM_KEYS)
Counts = dict[str, tuple[int, int, int]]
FloatArray = npt.NDArray[np.float64]


def count_instance(key: str, pred: Any, gold: Any, nospace: bool = False) -> tuple[int, int, int]:
    """(tp, fp, fn) for one field instance."""
    if gold is None:
        return (0, 1, 0) if pred is not None else (0, 0, 0)
    if pred is None:
        return (0, 0, 1)
    return (1, 0, 0) if field_equal(key, pred, gold, nospace) else (0, 1, 1)


@dataclass
class ItemResult:
    counts: Counts
    pairs: list[tuple[int, int]]
    fully_correct: int
    attr_correct: dict[str, int]
    attr_compared: dict[str, int]


def score_items(
    pred_items: Sequence[Mapping[str, Any]],
    gold_items: Sequence[Mapping[str, Any]],
    labelled: Iterable[str],
    nospace: bool = False,
) -> ItemResult:
    attrs = [a for a in ITEM_FIELDS if f"items.{a}" in set(labelled)]
    counts: dict[str, list[int]] = {f"items.{a}": [0, 0, 0] for a in attrs}
    pairs = match_items([dict(p) for p in pred_items], [dict(g) for g in gold_items], nospace)
    attr_correct = dict.fromkeys(attrs, 0)
    attr_compared = dict.fromkeys(attrs, 0)
    fully = 0
    matched_p = {p for p, _ in pairs}
    matched_g = {g for _, g in pairs}
    for pi, gi in pairs:
        p, g = pred_items[pi], gold_items[gi]
        for a in attrs:
            c = count_instance(f"items.{a}", p.get(a), g.get(a), nospace)
            for k in range(3):
                counts[f"items.{a}"][k] += c[k]
            if g.get(a) is not None:
                attr_compared[a] += 1
                attr_correct[a] += c[0]
        name_ok = str_equal(p.get("name"), g.get("name"), nospace)
        total_ok = (
            g.get("total") is None
            or "total" not in attrs
            or count_instance("items.total", p.get("total"), g.get("total")) == (1, 0, 0)
        )
        fully += int(name_ok and total_ok)
    for gi, g in enumerate(gold_items):
        if gi not in matched_g:
            for a in attrs:
                if g.get(a) is not None:
                    counts[f"items.{a}"][2] += 1
    for pi, p in enumerate(pred_items):
        if pi not in matched_p:
            for a in attrs:
                if p.get(a) is not None:
                    counts[f"items.{a}"][1] += 1
    return ItemResult(
        {k: (v[0], v[1], v[2]) for k, v in counts.items()},
        pairs,
        fully,
        attr_correct,
        attr_compared,
    )


def score_fields(
    pred: Mapping[str, Any], gold: Mapping[str, Any], labelled: Iterable[str], nospace: bool = False
) -> tuple[Counts, ItemResult | None]:
    lab = set(labelled)
    counts: Counts = {}
    for f in SCALAR_FIELDS:
        if f in lab:
            counts[f] = count_instance(f, pred.get(f), gold.get(f), nospace)
    items = None
    if "items.name" in lab:
        items = score_items(pred.get("items") or [], gold.get("items") or [], lab, nospace)
        counts.update(items.counts)
    return counts, items


def micro_f1(counts: Mapping[str, tuple[int, int, int]]) -> float:
    """Micro F1 over all field instances of one sample; 1.0 when there is nothing to find
    and nothing was predicted (correct abstention)."""
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def is_arith_consistent(pred: Mapping[str, Any], tol: float = 0.01) -> bool | None:
    """Σ item totals == subtotal (when both present). None if not applicable."""
    items = pred.get("items") or []
    sub = pred.get("subtotal")
    if not items or sub is None or any(it.get("total") is None for it in items):
        return None
    return abs(sum(float(it["total"]) for it in items) - float(sub)) <= tol + 1e-9


def is_fully_consistent(pred: Mapping[str, Any], tol: float = 0.01) -> bool | None:
    """Items vs subtotal and subtotal + tax vs total; None if neither check applies."""
    checks: list[bool] = []
    a = is_arith_consistent(pred, tol)
    if a is not None:
        checks.append(a)
    sub, tax, total = pred.get("subtotal"), pred.get("tax"), pred.get("total")
    if sub is not None and total is not None:
        checks.append(abs(float(sub) + float(tax or 0.0) - float(total)) <= tol + 1e-9)
    return all(checks) if checks else None


@dataclass
class SampleScore:
    id: str
    source: str
    language: str
    status: str
    counts: Counts
    lenient_counts: Counts
    total_correct: int | None
    items_labelled: bool
    items_gold: int = 0
    items_pred: int = 0
    items_tp: int = 0
    attr_correct: dict[str, int] = field(default_factory=dict)
    attr_compared: dict[str, int] = field(default_factory=dict)
    merchant_sim: float | None = None
    consistent: int | None = None
    pred_null: dict[str, int] = field(default_factory=dict)
    extra_text: bool = False
    latency_s: float | None = None
    error: str | None = None

    @property
    def f1(self) -> float:
        return micro_f1(self.counts)


def score_sample(
    sample: Mapping[str, Any], pred: str | ParseResult, latency_s: float | None = None
) -> SampleScore:
    """Score one prediction against one dataset sample (a row of data/processed/*.jsonl)."""
    pr = pred if isinstance(pred, ParseResult) else parse_prediction(pred)
    gold = sample["target"]
    labelled = set(sample["labelled"])
    nospace = sample.get("string_compare") == "nospace"
    strict = pr.strict_dict
    counts, items = score_fields(strict, gold, labelled, nospace)
    lenient_counts, _ = score_fields(pr.lenient, gold, labelled, nospace)

    total_correct = None
    if "total" in labelled and gold.get("total") is not None:
        total_correct = counts["total"][0]
    merchant_sim = None
    if "merchant" in labelled and gold.get("merchant") is not None:
        merchant_sim = str_similarity(strict.get("merchant"), gold["merchant"], nospace)
    cons = is_arith_consistent(strict) if pr.status == "ok" else None
    pred_null = {f: int(strict.get(f) is None) for f in SCALAR_FIELDS if f in labelled}
    if "items.name" in labelled:
        pred_null["items"] = int(not strict.get("items"))
    return SampleScore(
        id=str(sample["id"]),
        source=str(sample.get("source", "")),
        language=str(sample.get("language", "")),
        status=pr.status,
        counts=counts,
        lenient_counts=lenient_counts,
        total_correct=total_correct,
        items_labelled=items is not None,
        items_gold=len(gold.get("items") or []) if items is not None else 0,
        items_pred=len(strict.get("items") or []) if items is not None else 0,
        items_tp=items.fully_correct if items is not None else 0,
        attr_correct=items.attr_correct if items is not None else {},
        attr_compared=items.attr_compared if items is not None else {},
        merchant_sim=merchant_sim,
        consistent=None if cons is None else int(cons),
        pred_null=pred_null,
        extra_text=pr.extra_text,
        latency_s=latency_s,
        error=pr.error,
    )


# --------------------------------------------------------------------- count table
def _columns() -> list[str]:
    cols = ["n", "parse_ok", "schema_ok", "extra_text"]
    for prefix in ("", "L:"):
        for k in COUNT_KEYS:
            cols += [f"{prefix}{k}.tp", f"{prefix}{k}.fp", f"{prefix}{k}.fn"]
    cols += ["total_correct", "total_n", "items_tp", "items_pred", "items_gold"]
    for a in ITEM_FIELDS:
        cols += [f"attr.{a}.correct", f"attr.{a}.n"]
    cols += ["merchant_sim_sum", "merchant_sim_n", "consistent", "consistency_n"]
    return cols


COLUMNS: tuple[str, ...] = tuple(_columns())
_COL_INDEX = {c: i for i, c in enumerate(COLUMNS)}


def sample_vector(s: SampleScore) -> FloatArray:
    v = np.zeros(len(COLUMNS), dtype=np.float64)

    def put(col: str, val: float) -> None:
        v[_COL_INDEX[col]] += val

    put("n", 1)
    put("parse_ok", float(s.status != "parse_fail"))
    put("schema_ok", float(s.status == "ok"))
    put("extra_text", float(s.extra_text))
    for prefix, counts in (("", s.counts), ("L:", s.lenient_counts)):
        for k, (tp, fp, fn) in counts.items():
            put(f"{prefix}{k}.tp", tp)
            put(f"{prefix}{k}.fp", fp)
            put(f"{prefix}{k}.fn", fn)
    if s.total_correct is not None:
        put("total_correct", s.total_correct)
        put("total_n", 1)
    put("items_tp", s.items_tp)
    put("items_pred", s.items_pred)
    put("items_gold", s.items_gold)
    for a, c in s.attr_correct.items():
        put(f"attr.{a}.correct", c)
        put(f"attr.{a}.n", s.attr_compared.get(a, 0))
    if s.merchant_sim is not None:
        put("merchant_sim_sum", s.merchant_sim)
        put("merchant_sim_n", 1)
    if s.consistent is not None:
        put("consistent", s.consistent)
        put("consistency_n", 1)
    return v


@dataclass
class ScoreTable:
    ids: list[str]
    sources: list[str]
    languages: list[str]
    matrix: FloatArray  # (n_samples, n_columns)

    @classmethod
    def from_scores(cls, scores: Sequence[SampleScore]) -> ScoreTable:
        m = np.stack([sample_vector(s) for s in scores]) if scores else np.zeros((0, len(COLUMNS)))
        return cls(
            [s.id for s in scores], [s.source for s in scores], [s.language for s in scores], m
        )

    def subset(self, mask: Sequence[bool] | npt.NDArray[np.bool_]) -> ScoreTable:
        mk = np.asarray(mask, dtype=bool)
        return ScoreTable(
            [i for i, k in zip(self.ids, mk, strict=True) if k],
            [i for i, k in zip(self.sources, mk, strict=True) if k],
            [i for i, k in zip(self.languages, mk, strict=True) if k],
            self.matrix[mk],
        )

    def align(self, ids: Sequence[str]) -> ScoreTable:
        index = {i: k for k, i in enumerate(self.ids)}
        missing = [i for i in ids if i not in index]
        if missing:
            raise KeyError(f"{len(missing)} ids missing from score table, e.g. {missing[:3]}")
        order = [index[i] for i in ids]
        return ScoreTable(
            list(ids),
            [self.sources[k] for k in order],
            [self.languages[k] for k in order],
            self.matrix[order],
        )

    def __len__(self) -> int:
        return len(self.ids)


# --------------------------------------------------------------------- metrics on sums
Sums = Mapping[str, FloatArray]
MetricFn = Callable[[Sums], FloatArray]


def sums_of(matrix: FloatArray) -> dict[str, FloatArray]:
    """Column sums; ``matrix`` is (n, C) or already-summed (B, C) bootstrap sums."""
    s = matrix if matrix.ndim == 2 else matrix[None, :]
    return {c: s[:, i] for i, c in enumerate(COLUMNS)}


def _div(a: FloatArray, b: FloatArray) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _prf(
    s: Sums, keys: Iterable[str], prefix: str = ""
) -> tuple[FloatArray, FloatArray, FloatArray]:
    keys = list(keys)
    tp = sum(s[f"{prefix}{k}.tp"] for k in keys)
    fp = sum(s[f"{prefix}{k}.fp"] for k in keys)
    fn = sum(s[f"{prefix}{k}.fn"] for k in keys)
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    return _div(tp, tp + fp), _div(tp, tp + fn), _div(2 * tp, 2 * tp + fp + fn)


def _make_metrics() -> dict[str, MetricFn]:
    m: dict[str, MetricFn] = {
        "overall_f1": lambda s: _prf(s, COUNT_KEYS)[2],
        "overall_precision": lambda s: _prf(s, COUNT_KEYS)[0],
        "overall_recall": lambda s: _prf(s, COUNT_KEYS)[1],
        "lenient_f1": lambda s: _prf(s, COUNT_KEYS, "L:")[2],
        "scalar_f1": lambda s: _prf(s, SCALAR_FIELDS)[2],
        "items_attr_f1": lambda s: _prf(s, ITEM_KEYS)[2],
        "parse_rate": lambda s: _div(s["parse_ok"], s["n"]),
        "schema_valid_rate": lambda s: _div(s["schema_ok"], s["n"]),
        "total_em": lambda s: _div(s["total_correct"], s["total_n"]),
        "item_precision": lambda s: _div(s["items_tp"], s["items_pred"]),
        "item_recall": lambda s: _div(s["items_tp"], s["items_gold"]),
        "item_f1": lambda s: _div(2 * s["items_tp"], s["items_pred"] + s["items_gold"]),
        "merchant_similarity": lambda s: _div(s["merchant_sim_sum"], s["merchant_sim_n"]),
        "arith_consistency": lambda s: _div(s["consistent"], s["consistency_n"]),
    }
    for k in COUNT_KEYS:
        m[f"f1.{k}"] = lambda s, k=k: _prf(s, [k])[2]  # type: ignore[misc]
        m[f"precision.{k}"] = lambda s, k=k: _prf(s, [k])[0]  # type: ignore[misc]
        m[f"recall.{k}"] = lambda s, k=k: _prf(s, [k])[1]  # type: ignore[misc]
    for a in ITEM_FIELDS:
        m[f"attr_acc.{a}"] = lambda s, a=a: _div(s[f"attr.{a}.correct"], s[f"attr.{a}.n"])  # type: ignore[misc]
    return m


METRICS: dict[str, MetricFn] = _make_metrics()
HEADLINE: tuple[str, ...] = (
    "overall_f1",
    "lenient_f1",
    "parse_rate",
    "schema_valid_rate",
    "total_em",
    "item_f1",
    "merchant_similarity",
    "arith_consistency",
)


def field_support(table: ScoreTable) -> dict[str, int]:
    """Number of gold non-null instances per count key (to hide unsupported fields)."""
    s = sums_of(table.matrix.sum(0))
    return {k: int((s[f"{k}.tp"] + s[f"{k}.fn"])[0]) for k in COUNT_KEYS}


def compute(table: ScoreTable, names: Iterable[str] = HEADLINE) -> dict[str, float]:
    s = sums_of(table.matrix.sum(0))
    return {n: float(METRICS[n](s)[0]) for n in names}
