#!/usr/bin/env python
"""Download sources, convert, dedup, split, freeze the test manifest, write data_stats.md."""

from __future__ import annotations

import argparse
import os

from receipt_vlm.data.build_dataset import build
from receipt_vlm.utils.config import load_config
from receipt_vlm.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--merge-only", action="store_true", help="reuse cached real records")
    ap.add_argument(
        "--refreeze", action="store_true", help="DANGER: rewrite the frozen test manifest"
    )
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg["seed"]))
    build(cfg, merge_only=args.merge_only, refreeze=args.refreeze, workers=args.workers)


if __name__ == "__main__":
    main()
