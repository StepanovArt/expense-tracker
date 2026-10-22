"""Build ``data/processed/{train,val,test}.jsonl``: download, convert, hash, dedup, split.

``make data`` runs the full build. ``make synth`` regenerates synthetic receipts and then
re-runs only the merge/split step (``--merge-only``) on the cached real records.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from receipt_vlm.data.dedup import duplicate_groups, phash_hex
from receipt_vlm.data.record import Converted, Sample
from receipt_vlm.data.sources import SourceUnavailableError, iter_source
from receipt_vlm.data.splits import (
    SplitItem,
    assign_splits,
    check_or_freeze_manifest,
    manifest_rows,
)
from receipt_vlm.utils.io import read_jsonl, resolve, sha256_bytes, write_json, write_jsonl
from receipt_vlm.utils.logging import get_logger
from receipt_vlm.utils.seed import rng

log = get_logger(__name__)

REAL_CACHE = "_real_records.jsonl"


def _safe(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _process_one(args: tuple[Converted, str, int, bool]) -> dict[str, Any]:
    """Decode, normalise orientation, cap size, hash. Runs in a worker process."""
    conv, out_dir, max_side, reference_original = args
    raw = conv.image_bytes if conv.image_bytes is not None else Path(conv.image_path).read_bytes()  # type: ignore[arg-type]
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if reference_original and max(img.size) <= max_side and conv.image_path is not None:
        image_rel = str(Path(conv.image_path).resolve().relative_to(resolve(".")))
    else:
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        dest = resolve(out_dir) / conv.source / f"{_safe(conv.source_id)}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="JPEG", quality=92)
        image_rel = str(dest.relative_to(resolve(".")))
    return {
        "id": f"{conv.source}/{conv.source_id}",
        "source": conv.source,
        "official_split": conv.official_split,
        "image": image_rel,
        "image_sha256": sha256_bytes(raw),
        "phash": phash_hex(img),
        "width": img.size[0],
        "height": img.size[1],
        "language": conv.language,
        "target": conv.target.model_dump(mode="json"),
        "labelled": conv.labelled,
        "string_compare": conv.string_compare,
        "meta": conv.meta,
    }


def process_converted(
    convs: list[Converted], out_dir: str, max_side: int, workers: int, reference_original: bool
) -> list[dict[str, Any]]:
    jobs = [(c, out_dir, max_side, reference_original) for c in convs]
    if workers <= 1 or len(jobs) < 32:
        return [_process_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_process_one, jobs, chunksize=8))


def load_real_records(
    cfg: dict[str, Any], workers: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    raw = resolve(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    out_dir = str(Path(cfg["paths"]["processed"]) / "images")
    records: list[dict[str, Any]] = []
    gaps: dict[str, str] = {}
    for name, scfg in cfg["sources"].items():
        if not scfg.get("enabled", True):
            gaps[name] = "disabled in config"
            continue
        try:
            convs = list(iter_source(name, raw, scfg))
        except (SourceUnavailableError, FileNotFoundError) as e:
            log.warning("source %s unavailable: %s", name, e)
            gaps[name] = str(e).splitlines()[0][:300]
            continue
        log.info("source %s: %d converted samples, processing images", name, len(convs))
        records += process_converted(convs, out_dir, cfg["images"]["max_side"], workers, False)
    return records, gaps


def load_synthetic_records(cfg: dict[str, Any], workers: int) -> list[dict[str, Any]]:
    path = resolve(cfg["paths"]["synthetic"]) / "samples.jsonl"
    if not path.exists():
        return []
    from receipt_vlm.schema import Receipt

    convs = [
        Converted(
            source_id=r["source_id"],
            source="synthetic",
            image_path=resolve(r["image"]),
            image_bytes=None,
            target=Receipt.model_validate(r["target"]),
            labelled=r["labelled"],
            language=r["language"],
            meta=r.get("meta", {}),
        )
        for r in read_jsonl(path)
    ]
    out_dir = str(Path(cfg["paths"]["processed"]) / "images")
    return process_converted(convs, out_dir, cfg["images"]["max_side"], workers, True)


def cap_synthetic_train(
    records: list[dict[str, Any]], max_share: float, seed: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop synthetic *train* samples so they are at most ``max_share`` of train."""
    train = [r for r in records if r["split"] == "train"]
    synth = [r for r in train if r["source"] == "synthetic"]
    real_n = len(train) - len(synth)
    if not synth or max_share >= 1:
        return records, 0
    allowed = (
        len(synth) if real_n == 0 else min(len(synth), int(max_share / (1 - max_share) * real_n))
    )
    if real_n == 0:
        log.warning("no real train data: synthetic share cap cannot be applied")
    if allowed >= len(synth):
        return records, 0
    keep_idx = set(rng(seed, "synthetic_cap").permutation(len(synth))[:allowed].tolist())
    drop_ids = {s["id"] for i, s in enumerate(synth) if i not in keep_idx}
    return [r for r in records if r["id"] not in drop_ids], len(drop_ids)


def build(
    cfg: dict[str, Any], merge_only: bool = False, refreeze: bool = False, workers: int = 4
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    processed = resolve(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)
    cache = processed / REAL_CACHE
    info_path = processed / "build_info.json"
    if merge_only and cache.exists():
        real = read_jsonl(cache)
        gaps = json.loads(info_path.read_text())["gaps"] if info_path.exists() else {}
    else:
        real, gaps = load_real_records(cfg, workers)
        write_jsonl(cache, real)
    synthetic = load_synthetic_records(cfg, workers)
    records = real + synthetic
    if not records:
        raise RuntimeError(f"No data could be built. Gaps: {gaps}")

    groups = duplicate_groups(
        [r["image_sha256"] for r in records],
        [r["phash"] for r in records],
        int(cfg["split"]["phash_max_distance"]),
    )
    items = [
        SplitItem(r["id"], r["source"], r["language"], r["official_split"], g, r["image_sha256"])
        for r, g in zip(records, groups, strict=True)
    ]
    carve = [s for s in ("wildreceipt",) if s in {r["source"] for r in records}]
    res = assign_splits(items, cfg["split"]["ratios"], seed, carve_val_sources=carve)
    kept = []
    for r in records:
        if r["id"] in res.assignment:
            r = {**r, "split": res.assignment[r["id"]]}
            kept.append(r)
    kept, n_capped = cap_synthetic_train(kept, float(cfg["synthetic"]["max_train_share"]), seed)

    samples = {s: [] for s in ("train", "val", "test")}  # type: dict[str, list[dict[str, Any]]]
    for r in kept:
        sample = Sample.model_validate({k: v for k, v in r.items() if k != "official_split"})
        samples[sample.split].append(sample.model_dump(mode="json"))
    for split, rows in samples.items():
        rows.sort(key=lambda x: x["id"])
        write_jsonl(processed / f"{split}.jsonl", rows)

    splits_dir = resolve(cfg["paths"]["splits"])
    present = {r["source"] for r in kept}
    check_or_freeze_manifest(splits_dir / "test_manifest.jsonl", samples["test"], present, refreeze)
    # train/val manifests (ids + hashes only) are committed too, so the leakage test runs in CI
    for split in ("train", "val"):
        write_jsonl(splits_dir / f"{split}_manifest.jsonl", manifest_rows(samples[split]))
    info = {
        "gaps": gaps,
        "counts": {k: len(v) for k, v in samples.items()},
        "dropped": dict(Counter(res.dropped.values())),
        "dropped_ids": res.dropped,
        "synthetic_train_capped": n_capped,
        "seed": seed,
    }
    write_json(info_path, info)
    from receipt_vlm.data.stats import write_stats_report

    write_stats_report(samples, info, resolve(cfg["paths"]["stats_report"]))
    log.info("built dataset: %s (dropped %s, gaps %s)", info["counts"], info["dropped"], list(gaps))
    return info
