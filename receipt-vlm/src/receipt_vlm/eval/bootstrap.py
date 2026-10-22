"""Percentile bootstrap over test images, single-model and paired (CLAUDE.md §5.4).

Resampling weights ``W`` (B x N, multinomial counts) are drawn once; because every metric is
a function of additive per-sample counts, the B resampled corpora are just ``W @ M``.
For a paired comparison the *same* W is applied to both models' aligned count matrices, so
the per-image correlation between models is preserved.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from receipt_vlm.eval.metrics import METRICS, FloatArray, ScoreTable, sums_of

DEFAULT_B = 2000


def resample_weights(n: int, b: int = DEFAULT_B, seed: int = 0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    if n == 0:
        return np.zeros((b, 0))
    idx = rng.integers(0, n, size=(b, n))
    w = np.zeros((b, n), dtype=np.float64)
    np.add.at(w, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
    return w


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    n: int

    def fmt(self, pct: bool = True, digits: int = 1) -> str:
        if np.isnan(self.point):
            return "–"
        k = 100.0 if pct else 1.0
        return f"{k * self.point:.{digits}f} [{k * self.lo:.{digits}f}, {k * self.hi:.{digits}f}]"


def _percentiles(x: FloatArray, alpha: float) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(
    table: ScoreTable, names: Iterable[str], b: int = DEFAULT_B, seed: int = 0, alpha: float = 0.05
) -> dict[str, CI]:
    names = list(names)
    point = sums_of(table.matrix.sum(0))
    w = resample_weights(len(table), b, seed)
    boot = sums_of(w @ table.matrix)
    out: dict[str, CI] = {}
    for n in names:
        lo, hi = _percentiles(METRICS[n](boot), alpha)
        out[n] = CI(float(METRICS[n](point)[0]), lo, hi, len(table))
    return out


@dataclass(frozen=True)
class PairedResult:
    diff: float  # metric(A) - metric(B) on the full sample
    lo: float
    hi: float
    p_le_0: float  # share of resamples where A - B <= 0 (one-sided "A is not better")
    n: int

    @property
    def significant(self) -> bool:
        """95% CI of the difference excludes zero."""
        return bool(self.lo > 0 or self.hi < 0)


def paired_bootstrap(
    a: ScoreTable,
    b_table: ScoreTable,
    names: Iterable[str],
    b: int = DEFAULT_B,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, PairedResult]:
    """Compare model A against model B on the same images (aligned by id)."""
    b_table = b_table.align(a.ids)
    w = resample_weights(len(a), b, seed)
    pa, pb = sums_of(a.matrix.sum(0)), sums_of(b_table.matrix.sum(0))
    ba, bb = sums_of(w @ a.matrix), sums_of(w @ b_table.matrix)
    out: dict[str, PairedResult] = {}
    for n in names:
        d = METRICS[n](ba) - METRICS[n](bb)
        lo, hi = _percentiles(d, alpha)
        valid = d[~np.isnan(d)]
        p = float(np.mean(valid <= 0)) if valid.size else float("nan")
        out[n] = PairedResult(float(METRICS[n](pa)[0] - METRICS[n](pb)[0]), lo, hi, p, len(a))
    return out
