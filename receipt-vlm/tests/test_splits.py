from pathlib import Path

import pytest

from receipt_vlm.data.dedup import duplicate_groups, hamming
from receipt_vlm.data.splits import (
    ManifestMismatchError,
    SplitItem,
    assign_splits,
    check_or_freeze_manifest,
)

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
H0 = "0" * 64  # 256-bit hashes as hex
H_FAR = "f" * 64


def _flip(h: str, n_bits: int) -> str:
    """Flip the first n_bits of a hex hash."""
    v = int(h, 16) ^ (((1 << n_bits) - 1) << (256 - n_bits))
    return f"{v:064x}"


def test_hamming_and_groups() -> None:
    assert hamming(H0, _flip(H0, 5)) == 5
    shas = ["a", "b", "a", "c"]
    phs = [H0, _flip(H0, 3), H_FAR, _flip(H_FAR, 40)]
    g = duplicate_groups(shas, phs, max_distance=6)
    assert g[0] == g[1] == g[2]  # near-dup (0,1) and exact dup (0,2)
    assert g[3] != g[0]


def _items(n: int, source: str = "s", lang: str = "en", group_offset: int = 0) -> list[SplitItem]:
    return [
        SplitItem(f"{source}/{i}", source, lang, None, group_offset + i, f"sha{source}{i}")
        for i in range(n)
    ]


def test_ratio_split_is_deterministic_and_close_to_ratios() -> None:
    items = _items(1000)
    a = assign_splits(items, RATIOS, seed=1).assignment
    b = assign_splits(items, RATIOS, seed=1).assignment
    c = assign_splits(items, RATIOS, seed=2).assignment
    assert a == b and a != c
    counts = {s: sum(v == s for v in a.values()) for s in RATIOS}
    assert counts == {"train": 800, "val": 100, "test": 100}


def test_adding_a_source_does_not_move_existing_assignments() -> None:
    base = assign_splits(_items(200, "sroie"), RATIOS, seed=3).assignment
    more = assign_splits(
        _items(200, "sroie") + _items(50, "synthetic", "ar", 10_000), RATIOS, seed=3
    ).assignment
    assert all(more[k] == v for k, v in base.items())


def test_groups_are_atomic_and_official_test_is_protected() -> None:
    items = [
        SplitItem("cord/test_1", "cord", "id", "test", 7, "x"),
        SplitItem("cord/train_9", "cord", "id", "train", 7, "y"),  # re-scan of a test receipt
        SplitItem("sroie/1", "sroie", "en", None, 7, "z"),  # cross-source near dup
        SplitItem("cord/train_2", "cord", "id", "train", 8, "w"),
        SplitItem("cord/train_2b", "cord", "id", "train", 8, "w"),  # exact dup
    ]
    res = assign_splits(items, RATIOS, seed=0)
    assert res.assignment["cord/test_1"] == "test"
    assert res.assignment["sroie/1"] == "test"  # inherits the protected split
    assert res.dropped["cord/train_9"] == "near_duplicate_of_test"
    assert res.dropped["cord/train_2b"] == "exact_duplicate"
    assert res.assignment["cord/train_2"] == "train"


def test_val_carved_from_official_train() -> None:
    items = [SplitItem(f"w/{i}", "w", "en", "train", i, f"h{i}") for i in range(90)]
    items += [SplitItem(f"w/t{i}", "w", "en", "test", 100 + i, f"t{i}") for i in range(10)]
    a = assign_splits(items, RATIOS, seed=0, carve_val_sources=["w"]).assignment
    assert sum(v == "val" for v in a.values()) == 10
    assert sum(v == "test" for v in a.values()) == 10


def _row(i: str, src: str = "sroie", sha: str = "h") -> dict[str, str]:
    return {"id": i, "source": src, "image_sha256": sha + i, "phash": H0}


def test_manifest_freeze_check_and_append(tmp_path: Path) -> None:
    p = tmp_path / "test_manifest.jsonl"
    check_or_freeze_manifest(p, [_row("sroie/1"), _row("sroie/2")], {"sroie"})
    # identical rebuild passes; a new source is appended
    rows = check_or_freeze_manifest(
        p,
        [_row("sroie/1"), _row("sroie/2"), _row("synthetic/1", "synthetic")],
        {"sroie", "synthetic"},
    )
    assert [r["id"] for r in rows] == ["sroie/1", "sroie/2", "synthetic/1"]
    # a frozen sample leaving test is an error
    with pytest.raises(ManifestMismatchError):
        check_or_freeze_manifest(
            p, [_row("sroie/1"), _row("synthetic/1", "synthetic")], {"sroie", "synthetic"}
        )
    # a frozen source that failed to download this time is left untouched
    rows = check_or_freeze_manifest(p, [_row("synthetic/1", "synthetic")], {"synthetic"})
    assert len(rows) == 3
