"""Train/test leakage checks on the committed split manifests (ids + image hashes).

These run without any images: the manifests in data/splits/ are written by `make data`.
"""

from pathlib import Path

import numpy as np
import pytest

from receipt_vlm.data.dedup import hamming_matrix
from receipt_vlm.utils.config import load_config
from receipt_vlm.utils.io import read_jsonl

ROOT = Path(__file__).parents[1]
SPLITS = ROOT / "data" / "splits"


def _manifest(split: str) -> list[dict[str, str]]:
    p = SPLITS / f"{split}_manifest.jsonl"
    if not p.exists():
        pytest.skip(f"{p.name} not built yet (make data)")
    return read_jsonl(p)


@pytest.mark.parametrize("other", ["train", "val"])
def test_no_exact_hash_shared_with_test(other: str) -> None:
    test = {r["image_sha256"] for r in _manifest("test")}
    rows = _manifest(other)
    shared = [r["id"] for r in rows if r["image_sha256"] in test]
    assert not shared, f"{other} images also in test: {shared[:10]}"


@pytest.mark.parametrize("other", ["train", "val"])
def test_no_near_duplicate_shared_with_test(other: str) -> None:
    max_d = int(load_config(ROOT / "configs" / "data.yaml")["split"]["phash_max_distance"])
    test, rows = _manifest("test"), _manifest(other)
    d = hamming_matrix([r["phash"] for r in test] + [r["phash"] for r in rows])
    cross = d[: len(test), len(test) :]
    i, j = np.nonzero(cross <= max_d)
    pairs = [(test[a]["id"], rows[b]["id"], int(cross[a, b])) for a, b in zip(i, j, strict=True)]
    assert not pairs, f"near-duplicates across {other}/test: {pairs[:10]}"


def test_ids_unique_across_splits() -> None:
    ids = [r["id"] for s in ("train", "val", "test") for r in _manifest(s)]
    assert len(ids) == len(set(ids))
