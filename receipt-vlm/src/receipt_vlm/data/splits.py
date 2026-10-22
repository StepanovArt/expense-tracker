"""Split assignment and the frozen test manifest (CLAUDE.md §4.3).

Rules, in priority order:

1. Duplicate groups (exact or near-duplicate images) are atomic: one group, one split.
2. Official splits are respected. If a group contains members from different official splits,
   the group takes the most protected one (test > val > train) and members whose official
   split differs are *dropped* (a re-scan of a test receipt must not stay in train).
3. Sources without official splits are split 80/10/10 per (source, language) stratum by a
   seeded permutation of groups, so the assignment of one source never depends on another.
   Sources with official train/test but no val get val carved out of their official train.
4. Within a group only one sample per (source, sha256) is kept (exact duplicates removed).

The test manifest (ids + image hashes) is frozen on first build. Later builds must reproduce
every frozen entry; new sources may append their test samples. ``refreeze=True`` overrides.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from receipt_vlm.data.record import Split
from receipt_vlm.utils.io import read_jsonl, write_jsonl
from receipt_vlm.utils.seed import rng

PROTECTION: dict[str, int] = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class SplitItem:
    id: str
    source: str
    language: str
    official_split: Split | None
    group: int
    sha256: str


@dataclass
class SplitResult:
    assignment: dict[str, Split]  # kept ids -> split
    dropped: dict[str, str]  # id -> reason


def _ratio_split(n_groups_sizes: list[int], ratios: dict[str, float]) -> list[Split]:
    """Assign groups (already shuffled) to splits by cumulative sample share."""
    total = sum(n_groups_sizes)
    bounds = (ratios["train"] * total, (ratios["train"] + ratios["val"]) * total)
    out: list[Split] = []
    seen = 0.0
    for size in n_groups_sizes:
        mid = seen + size / 2  # assign by the group's midpoint -> unbiased for big groups
        out.append("train" if mid < bounds[0] else "val" if mid < bounds[1] else "test")
        seen += size
    return out


def assign_splits(
    items: Sequence[SplitItem],
    ratios: dict[str, float],
    seed: int,
    carve_val_sources: Sequence[str] = (),
) -> SplitResult:
    by_group: dict[int, list[SplitItem]] = defaultdict(list)
    for it in items:
        by_group[it.group].append(it)

    assignment: dict[str, Split] = {}
    dropped: dict[str, str] = {}
    free_groups: dict[tuple[str, str], list[list[SplitItem]]] = defaultdict(list)
    carve: dict[str, list[list[SplitItem]]] = defaultdict(list)

    for _gid, members in sorted(by_group.items()):
        # exact duplicates: keep the first per (source, sha256), prefer protected official split
        members = sorted(
            members,
            key=lambda m: (-PROTECTION.get(m.official_split or "train", 0), m.id),
        )
        seen_hash: set[tuple[str, str]] = set()
        uniq: list[SplitItem] = []
        for m in members:
            key = (m.source, m.sha256)
            if key in seen_hash:
                dropped[m.id] = "exact_duplicate"
                continue
            seen_hash.add(key)
            uniq.append(m)

        official = [m.official_split for m in uniq if m.official_split]
        if official:
            target: Split = max(official, key=lambda s: PROTECTION[s])
            keep = []
            for m in uniq:
                if m.official_split and m.official_split != target:
                    dropped[m.id] = f"near_duplicate_of_{target}"
                else:
                    keep.append(m)
            if target == "train" and keep[0].source in carve_val_sources:
                carve[keep[0].source].append(keep)
            else:
                for m in keep:
                    assignment[m.id] = target
        else:
            anchor = uniq[0]
            free_groups[(anchor.source, anchor.language)].append(uniq)

    for (source, lang), groups in sorted(free_groups.items()):
        order = rng(seed, "split", source, lang).permutation(len(groups))
        shuffled = [groups[i] for i in order]
        for g, split in zip(
            shuffled, _ratio_split([len(g) for g in shuffled], ratios), strict=True
        ):
            for m in g:
                assignment[m.id] = split

    val_share = ratios["val"] / (ratios["train"] + ratios["val"])
    for source, groups in sorted(carve.items()):
        order = rng(seed, "carve_val", source).permutation(len(groups))
        shuffled = [groups[i] for i in order]
        sub = _ratio_split([len(g) for g in shuffled], {"train": 1 - val_share, "val": val_share})
        for g, split in zip(shuffled, sub, strict=True):
            for m in g:
                assignment[m.id] = split
    return SplitResult(assignment=assignment, dropped=dropped)


# ----------------------------------------------------------------------------- manifest
def manifest_rows(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": s["id"],
            "source": s["source"],
            "image_sha256": s["image_sha256"],
            "phash": s["phash"],
        }
        for s in sorted(samples, key=lambda s: s["id"])
    ]


class ManifestMismatchError(RuntimeError):
    pass


def check_or_freeze_manifest(
    path: Path,
    test_samples: Sequence[dict[str, Any]],
    present_sources: set[str],
    refreeze: bool = False,
) -> list[dict[str, Any]]:
    """Freeze the test manifest on first call; afterwards verify nothing frozen moved.

    ``present_sources`` are the sources built in this run. Frozen entries of a source that is
    absent (e.g. its download failed) are kept untouched. New sources are appended; any change
    to a frozen source that is present raises ManifestMismatchError.
    """
    rows = manifest_rows(test_samples)
    if refreeze or not path.exists():
        write_jsonl(path, rows)
        return rows
    frozen = read_jsonl(path)
    frozen_sources = {r["source"] for r in frozen}
    current = {r["id"]: r for r in rows}
    problems = []
    for r in frozen:
        cur = current.get(r["id"])
        if cur is None:
            if r["source"] in present_sources:
                problems.append(f"frozen test sample {r['id']} is no longer in test")
        elif cur["image_sha256"] != r["image_sha256"]:
            problems.append(f"image hash changed for frozen test sample {r['id']}")
    frozen_ids = {r["id"] for r in frozen}
    for r in rows:
        if r["source"] in frozen_sources and r["id"] not in frozen_ids:
            problems.append(f"new test sample {r['id']} in already-frozen source {r['source']}")
    if problems:
        raise ManifestMismatchError(
            "Test manifest mismatch (use --refreeze only if you really mean to change the test "
            "set):\n  " + "\n  ".join(problems[:20])
        )
    merged = frozen + [r for r in rows if r["source"] not in frozen_sources]
    merged.sort(key=lambda r: r["id"])
    write_jsonl(path, merged)
    return merged
