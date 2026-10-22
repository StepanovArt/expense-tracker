#!/usr/bin/env python
"""`make smoke`: the whole pipeline on tiny data with a tiny random Qwen3-VL, CPU, < 5 min.

Runs the real CLI scripts (data build, SFT, eval, GRPO, report, error analysis, merge) plus an
API request, all under outputs/smoke/. It proves the pipeline runs end to end; the model is
random, so every number it produces is meaningless and nothing is written to reports/.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "outputs" / "smoke"
PY = [sys.executable]


def run(name: str, args: list[str], timings: list[tuple[str, float]]) -> None:
    t0 = time.time()
    print(f"\n=== {name}: {' '.join(args)}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.run(PY + args, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], proc.stderr[-6000:], sep="\n")
        raise SystemExit(f"smoke step '{name}' failed (exit {proc.returncode})")
    tail = [ln for ln in proc.stderr.splitlines() if " INFO " in ln][-2:]
    print("\n".join(tail))
    timings.append((name, time.time() - t0))


def main() -> None:
    t_start = time.time()
    shutil.rmtree(SMOKE, ignore_errors=True)
    timings: list[tuple[str, float]] = []

    run(
        "tiny model",
        [
            "-c",
            "from receipt_vlm.models.tiny import build_tiny_qwen3_vl as b; b('outputs/smoke/tiny-qwen3vl')",
        ],
        timings,
    )
    run(
        "synthetic data",
        [
            "scripts/generate_synthetic.py",
            "--config",
            "configs/data_smoke.yaml",
            "--grid",
            "outputs/smoke/synthetic_grid.png",
        ],
        timings,
    )
    run(
        "dataset build",
        ["scripts/build_data.py", "--config", "configs/data_smoke.yaml", "--merge-only"],
        timings,
    )
    run("SFT (LoRA)", ["scripts/train_sft.py", "--config", "configs/sft/smoke.yaml"], timings)
    run(
        "eval TINY + TINY_SFT (val)",
        [
            "scripts/run_eval.py",
            "--config",
            "configs/eval/smoke.yaml",
            "--split",
            "val",
            "--models",
            "TINY",
            "TINY_SFT",
        ],
        timings,
    )
    run("GRPO", ["scripts/train_grpo.py", "--config", "configs/grpo/smoke.yaml"], timings)
    run(
        "report",
        [
            "scripts/make_report.py",
            "--config",
            "configs/eval/smoke.yaml",
            "--out",
            "outputs/smoke/RESULTS_smoke.md",
            "--fig-dir",
            "outputs/smoke/figures",
        ],
        timings,
    )
    run(
        "error analysis",
        [
            "scripts/error_analysis.py",
            "--config",
            "configs/eval/smoke.yaml",
            "--model",
            "TINY_SFT",
            "--split",
            "val",
            "--out",
            "outputs/smoke/RESULTS_smoke.md",
            "--fig-dir",
            "outputs/smoke/figures",
        ],
        timings,
    )
    run(
        "merge LoRA",
        [
            "scripts/merge_lora.py",
            "--base",
            "outputs/smoke/tiny-qwen3vl",
            "--adapter",
            "outputs/smoke/sft/best",
            "--out",
            "outputs/smoke/merged",
            "--dtype",
            "float32",
        ],
        timings,
    )
    run(
        "API /extract",
        [
            "-c",
            (
                "import glob, os\n"
                "os.environ.update(RECEIPT_VLM_BACKEND='hf', RECEIPT_VLM_MODEL='outputs/smoke/merged',"
                " RECEIPT_VLM_MAX_PIXELS='65536', RECEIPT_VLM_MAX_NEW_TOKENS='32', RECEIPT_VLM_DTYPE='float32')\n"
                "from fastapi.testclient import TestClient\n"
                "from receipt_vlm.serve.api import app\n"
                "img = sorted(glob.glob('outputs/smoke/data/synthetic/images/*.jpg'))[0]\n"
                "with TestClient(app) as c:\n"
                "    r = c.post('/extract', files={'image': ('r.jpg', open(img, 'rb').read(), 'image/jpeg')})\n"
                "assert r.status_code in (200, 422), r.text\n"
                "import logging; logging.getLogger('smoke').warning('INFO /extract -> HTTP %s (random tiny model: 422 expected)', r.status_code)\n"
            ),
        ],
        timings,
    )

    total = time.time() - t_start
    print("\n=== smoke summary (CPU, tiny random model - numbers are meaningless)")
    for name, sec in timings:
        print(f"  {name:<28} {sec:6.1f} s")
    print(f"  {'TOTAL':<28} {total:6.1f} s  ({'OK' if total < 300 else 'SLOWER THAN 5 MIN'})")
    print(
        "artifacts: outputs/smoke/{data,sft,grpo,preds,metrics,merged}, outputs/smoke/RESULTS_smoke.md"
    )


if __name__ == "__main__":
    main()
