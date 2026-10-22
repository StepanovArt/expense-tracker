#!/usr/bin/env python
"""GRPO: `uv run python scripts/train_grpo.py --config configs/grpo/gpu16.yaml [key=value ...]`"""

from __future__ import annotations

import argparse

from receipt_vlm.train.grpo import train_grpo
from receipt_vlm.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/grpo/gpu16.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    train_grpo(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
