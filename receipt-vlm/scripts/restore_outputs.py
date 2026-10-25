#!/usr/bin/env python
"""Restore a packed run (the archive of the val run) into outputs/ and check it is usable.

    uv run python scripts/restore_outputs.py --archive /workspace/results_val.tgz

The archive may hold ``outputs/...`` at any depth. Everything under that ``outputs`` folder is
copied into place, then the files the ablation needs (the SFT adapter, the val predictions
it is compared with) are checked, so a bad archive stops the run before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

from receipt_vlm.utils.io import resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger("restore_outputs")

REQUIRED = (
    "sft/best/adapter_config.json",
    "preds/B1/val.jsonl",
    "preds/M1/val.jsonl",
    "preds/M2/val.jsonl",
)
EXPECTED = ("preds/B0/val.jsonl", "preds/B2/val.jsonl", "preds/B3/val.jsonl")


def outputs_root(tree: Path) -> Path:
    """The folder that holds ``sft/best`` inside an extracted archive."""
    hits = sorted(tree.rglob("sft/best/adapter_config.json"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit("the archive has no sft/best/adapter_config.json (the SFT adapter)")
    return hits[0].parents[2]


def restore(archive: Path, dest: Path, required: Sequence[str] = REQUIRED) -> list[str]:
    """Copy the archive's outputs folder into ``dest``; returns the top-level names copied."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive) as tf:
            tf.extractall(tmp, filter="data")
        root = outputs_root(Path(tmp))
        copied = []
        for item in sorted(root.iterdir()):
            if item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest / item.name)
            copied.append(item.name)
    missing = [r for r in required if not (dest / r).exists()]
    if missing:
        raise SystemExit(f"the archive lacks {missing}: the ablation cannot be compared")
    for e in EXPECTED:
        if not (dest / e).exists():
            log.warning("%s is missing: that baseline will be listed as not run", e)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--dest", default="outputs")
    args = ap.parse_args()
    if not args.archive.exists():
        raise SystemExit(f"{args.archive} not found: upload results_val.tgz to the server first")
    copied = restore(args.archive, resolve(args.dest))
    log.info("restored %s into %s", ", ".join(copied), args.dest)


if __name__ == "__main__":
    main()
