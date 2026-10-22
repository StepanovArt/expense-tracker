"""Optimal one-to-one matching of predicted to gold line items (Hungarian algorithm).

Pair score = 0.6 * name similarity + 0.4 * [totals equal within 0.01]. A pair is admissible
only if the names are at least loosely similar (similarity >= 0.5), or the totals match and
the names are not unrelated (similarity >= 0.3). Among admissible pairs we maximise the total
score; inadmissible pairs are never matched, so a hallucinated item does not "steal" credit
from an unrelated gold item that happens to cost the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from scipy.optimize import linear_sum_assignment

from receipt_vlm.eval.normalize import num_equal, str_similarity

W_NAME: Final[float] = 0.6
W_TOTAL: Final[float] = 0.4
MIN_SIM: Final[float] = 0.5
MIN_SIM_WITH_TOTAL: Final[float] = 0.3


def pair_score(pred: dict[str, Any], gold: dict[str, Any], nospace: bool = False) -> float | None:
    """Score of pairing ``pred`` with ``gold``; None if the pair is inadmissible."""
    sim = str_similarity(pred.get("name"), gold.get("name"), nospace)
    tm = num_equal(pred.get("total"), gold.get("total"))
    if sim >= MIN_SIM or (tm and sim >= MIN_SIM_WITH_TOTAL):
        return W_NAME * sim + W_TOTAL * float(tm)
    return None


def match_items(
    pred: Sequence[dict[str, Any]], gold: Sequence[dict[str, Any]], nospace: bool = False
) -> list[tuple[int, int]]:
    """Return matched (pred_index, gold_index) pairs, sorted by gold index."""
    if not pred or not gold:
        return []
    big = 1e6
    cost = np.full((len(pred), len(gold)), big)
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            s = pair_score(p, g, nospace)
            if s is not None:
                cost[i, j] = -s
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] < big]
    return sorted(pairs, key=lambda rc: rc[1])
