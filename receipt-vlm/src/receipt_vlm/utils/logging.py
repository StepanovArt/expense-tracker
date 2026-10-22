"""Logging: stdlib logger + a metrics sink that is W&B when configured, else CSV/JSONL."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from receipt_vlm.utils.io import append_jsonl, resolve


def get_logger(name: str = "receipt_vlm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


class MetricsLogger:
    """Writes every logged dict to ``<run_dir>/metrics.jsonl``; mirrors to W&B if available.

    W&B is used only when WANDB_API_KEY is set and wandb is importable, so CI and
    smoke runs never touch the network.
    """

    def __init__(self, run_dir: str | Path, run_name: str, config: dict[str, Any] | None = None):
        self.run_dir = resolve(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self._wandb: Any = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self._wandb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "receipt-vlm"),
                    name=run_name,
                    config=config or {},
                    dir=str(self.run_dir),
                )
            except Exception as e:  # tracking must never kill a run
                get_logger().warning("W&B init failed (%s); logging to %s only", e, self.path)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        row = {"time": time.time(), "step": step, **metrics}
        append_jsonl(self.path, row)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
