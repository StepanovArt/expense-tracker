#!/usr/bin/env python
"""Generate synthetic EN/AR/RU receipts (exact ground truth) + the sample grid figure."""

from __future__ import annotations

import argparse
import os

from receipt_vlm.data.synthetic.generate import generate, sample_grid
from receipt_vlm.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=None, help="override synthetic.n_samples")
    ap.add_argument("--out", default=None, help="override paths.synthetic")
    ap.add_argument("--grid", default="reports/figures/synthetic_samples.png")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = args.n or int(cfg["synthetic"]["n_samples"])
    out = args.out or cfg["paths"]["synthetic"]
    rows = generate(n, int(cfg["seed"]), out, workers=args.workers)
    if args.grid:
        sample_grid(rows, args.grid, n=int(cfg["synthetic"]["grid_samples"]), seed=int(cfg["seed"]))


if __name__ == "__main__":
    main()
