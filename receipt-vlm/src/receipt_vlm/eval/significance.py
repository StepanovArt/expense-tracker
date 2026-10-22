"""McNemar's test for paired binary outcomes (total exact match of two models, same images)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import binomtest, chi2


@dataclass(frozen=True)
class McNemarResult:
    n: int  # pairs where both outcomes are defined
    a_only: int  # A correct, B wrong
    b_only: int  # A wrong, B correct
    statistic: float  # chi-square with continuity correction
    p_value: float  # exact binomial if a_only + b_only < 25, else chi-square
    exact: bool


def mcnemar(a_correct: Sequence[int | None], b_correct: Sequence[int | None]) -> McNemarResult:
    if len(a_correct) != len(b_correct):
        raise ValueError("outcome lists must be aligned (same images, same order)")
    pairs = [
        (a, b) for a, b in zip(a_correct, b_correct, strict=True) if a is not None and b is not None
    ]
    a_only = sum(1 for a, b in pairs if a and not b)
    b_only = sum(1 for a, b in pairs if b and not a)
    disc = a_only + b_only
    if disc == 0:
        return McNemarResult(len(pairs), 0, 0, 0.0, 1.0, True)
    stat = (abs(a_only - b_only) - 1) ** 2 / disc
    if disc < 25:
        p = float(binomtest(min(a_only, b_only), disc, 0.5).pvalue)
        return McNemarResult(len(pairs), a_only, b_only, float(stat), p, True)
    return McNemarResult(len(pairs), a_only, b_only, float(stat), float(chi2.sf(stat, 1)), False)
