#!/usr/bin/env python
"""Run models on a split, save raw predictions + a metrics JSON (with bootstrap CIs) per model.

uv run python scripts/run_eval.py --config configs/eval/baselines.yaml --split val --models B1 B2
"""

from __future__ import annotations

import argparse
import traceback

from receipt_vlm.eval.report import metrics_payload
from receipt_vlm.eval.runner import evaluate_model
from receipt_vlm.utils.config import load_config
from receipt_vlm.utils.io import resolve, write_json
from receipt_vlm.utils.logging import get_logger
from receipt_vlm.utils.seed import seed_everything

log = get_logger("run_eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval/baselines.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg["seed"]))
    if args.split == "test":
        log.warning(
            "TEST split: final evaluation only (CLAUDE.md §4.3) - never tune on these numbers"
        )
    failed = []
    for mid in args.models:
        spec = cfg["models"].get(mid)
        if spec is None:
            raise SystemExit(f"model {mid!r} not in {args.config}")
        try:
            scores, table = evaluate_model(mid, spec, cfg, args.split)
        except Exception as e:  # keep going with the other models; record why this one failed
            log.error("%s failed: %s", mid, e)
            err = resolve(cfg["metrics_dir"]) / mid / f"{args.split}.error.txt"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(traceback.format_exc())
            failed.append(mid)
            continue
        payload = metrics_payload(
            scores, table, int(cfg["bootstrap"]["resamples"]), int(cfg["seed"])
        )
        payload["model"] = {"id": mid, **spec}
        write_json(resolve(cfg["metrics_dir"]) / mid / f"{args.split}.json", payload)
        h = payload["headline"]
        log.info(
            "%s %s n=%d overall_f1=%.3f [%.3f, %.3f] schema_valid=%.3f total_em=%.3f",
            mid,
            args.split,
            payload["n"],
            h["overall_f1"]["point"],
            h["overall_f1"]["lo"],
            h["overall_f1"]["hi"],
            h["schema_valid_rate"]["point"],
            h["total_em"]["point"],
        )
    if failed:
        raise SystemExit(f"failed models: {failed} (see {cfg['metrics_dir']}/<model>/*.error.txt)")


if __name__ == "__main__":
    main()
