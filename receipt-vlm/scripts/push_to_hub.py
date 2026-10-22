#!/usr/bin/env python
"""Push the LoRA adapter and the merged model to the Hugging Face Hub with a model card.

    uv run python scripts/push_to_hub.py --adapter outputs/sft/best --merged outputs/merged \\
        --repo <user>/qwen3-vl-2b-receipts [--model-id M2] [--private]

Only uploads when HF_TOKEN is set; otherwise writes the card to outputs/model_card.md and
exits. Metrics in the card are read from outputs/metrics/<model-id>/test.json (written by
run_eval.py); if that file does not exist the card says the model is not evaluated yet.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from receipt_vlm.utils.io import resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger("push_to_hub")

HEADLINE_ROWS = [
    ("overall_f1", "Overall field F1 (strict, primary)"),
    ("lenient_f1", "Lenient field F1"),
    ("schema_valid_rate", "Schema-valid JSON rate"),
    ("total_em", "Total exact match (±0.01)"),
    ("item_f1", "Line-item F1"),
]


def _pct(ci: dict[str, float]) -> str:
    return f"{100 * ci['point']:.1f} [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]"


def render_model_card(
    repo: str, metrics: dict[str, Any] | None, data_info: dict[str, Any] | None, base_model: str
) -> str:
    if metrics:
        rows = "\n".join(
            f"| {label} | {_pct(metrics['headline'][k])} |"
            for k, label in HEADLINE_ROWS
            if k in metrics["headline"]
        )
        lat = metrics.get("latency_s", {})
        by_lang = "\n".join(
            f"| {lg} | {v['n']} | {_pct(v['overall_f1'])} |"
            + (" low-n |" if v.get("low_confidence") else " |")
            for lg, v in sorted(metrics.get("breakdown", {}).get("language", {}).items())
        )
        evaluation = (
            f"Frozen test split, n = {metrics['n']} receipts, greedy decoding, 95 % bootstrap CIs "
            "(2,000 resamples over images). Full tables, baselines and paired significance tests: "
            "`reports/RESULTS.md` in the source repository.\n\n"
            f"| Metric | Value (95 % CI) |\n|---|---|\n{rows}\n\n"
            f"| Language | n | Overall F1 (95 % CI) | |\n|---|---|---|---|\n{by_lang}\n\n"
            f"Latency p50 / p95: {lat.get('p50', float('nan')):.2f} / {lat.get('p95', float('nan')):.2f} s per image "
            "(backend and GPU as recorded in the repository)."
        )
    else:
        evaluation = (
            "**Not evaluated yet.** No test-set metrics were available when this card was written."
        )
    counts = (data_info or {}).get("counts", {})
    gaps = (data_info or {}).get("gaps", {})
    data_line = (
        f"train / val / test = {counts.get('train', '?')} / {counts.get('val', '?')} / {counts.get('test', '?')} receipts"
        + (f"; sources missing from this build: {', '.join(gaps)}" if gaps else "")
    )
    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: transformers
pipeline_tag: image-text-to-text
language: [en, ar, ru, id]
tags: [receipts, document-understanding, information-extraction, qwen3-vl, lora, grpo]
---

# {repo.split("/")[-1]}

Qwen3-VL-2B-Instruct fine-tuned (LoRA SFT, optionally + GRPO) to turn a **photo of a shop
receipt into one strict, schema-valid JSON object**: merchant, ISO date, time, ISO 4217
currency, line items (name, quantity, unit price, total), subtotal, tax, total, payment method
and receipt language. English, Arabic and Russian receipts (plus Indonesian/Malaysian real
receipts in the training data).

## Output schema

```json
{{"merchant": str|null, "date": "YYYY-MM-DD"|null, "time": "HH:MM"|null, "currency": "USD"|null,
 "items": [{{"name": str, "quantity": num|null, "unit_price": num|null, "total": num|null}}],
 "subtotal": num|null, "tax": num|null, "total": num|null,
 "payment_method": "cash|card|mobile|voucher|other"|null, "language": "en"|null}}
```

## How to use

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
model = AutoModelForImageTextToText.from_pretrained("{repo}", dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("{repo}")
# use the exact prompt from receipt_vlm.prompts (SYSTEM_PROMPT + INSTRUCTION), greedy decoding
```

Serve with vLLM (`vllm serve {repo}`) behind the validating FastAPI service from the source
repository, which retries once and rejects non-schema output with HTTP 422.

## Training

* Data: {data_line}. Sources: CORD v2 (CC BY 4.0), SROIE (ICDAR 2019, research use),
  WildReceipt (MMOCR), and synthetic EN/AR/RU receipts with exact ground truth (≤ 50 % of train).
  Fields a source does not annotate are excluded from the loss and from evaluation.
* SFT: LoRA r=16, α=32 on the language model's attention + MLP projections, vision tower
  frozen, loss on the assistant JSON only; checkpoint selected by generation-based
  validation F1.
* GRPO (if applied): reward = 0.2·format + 0.5·field-F1 + 0.2·total + 0.1·arithmetic
  consistency, KL to the SFT policy, null-rate monitoring against reward hacking.

## Evaluation

{evaluation}

## Intended use and limitations

* Intended: digitising receipts for expense tracking / bookkeeping assistance **with a human
  in the loop**; always check totals before booking.
* Not intended: fully automatic accounting, fraud detection, or any decision about people.
* Known limitations: trained on few real receipts; synthetic receipts are stylised; quality
  drops on blurred, crumpled or partial photos and on long receipts (images are downscaled to
  ≈ 1 MP); Arabic and Russian performance is measured mostly on synthetic receipts; values the
  receipt does not print should be null, but the model can hallucinate them.
* Licence note: base model and code are Apache-2.0, but part of the training data (SROIE,
  WildReceipt) is distributed for research purposes; check the data licences before
  commercial use.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", default="outputs/sft/best")
    ap.add_argument("--merged", default="outputs/merged")
    ap.add_argument("--repo", required=True, help="<user>/<name>; the adapter goes to <name>-lora")
    ap.add_argument(
        "--model-id", default="M1", help="metrics id for the card (outputs/metrics/<id>/test.json)"
    )
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    mpath = resolve(f"outputs/metrics/{args.model_id}/test.json")
    metrics = json.loads(mpath.read_text()) if mpath.exists() else None
    ipath = resolve("data/processed/build_info.json")
    info = json.loads(ipath.read_text()) if ipath.exists() else None
    card = render_model_card(args.repo, metrics, info, args.base_model)
    out = resolve("outputs/model_card.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    log.info("model card written to %s (metrics: %s)", out, "yes" if metrics else "NOT EVALUATED")

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set: nothing uploaded")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for folder, repo in ((args.merged, args.repo), (args.adapter, f"{args.repo}-lora")):
        if not Path(resolve(folder)).exists():
            log.warning("skip %s: %s does not exist", repo, folder)
            continue
        api.create_repo(repo, private=args.private, exist_ok=True)
        (resolve(folder) / "README.md").write_text(card, encoding="utf-8")
        api.upload_folder(
            repo_id=repo, folder_path=str(resolve(folder)), commit_message="receipt-vlm release"
        )
        log.info("uploaded %s -> https://huggingface.co/%s", folder, repo)


if __name__ == "__main__":
    main()
