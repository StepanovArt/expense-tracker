#!/usr/bin/env python
"""Merge LoRA adapter(s) into the base weights and save a standalone model + processor.

    uv run python scripts/merge_lora.py --adapter outputs/sft/best --out outputs/merged
    # M2: GRPO adapter on top of the merged SFT model
    uv run python scripts/merge_lora.py --base outputs/sft/merged --adapter outputs/grpo/best --out outputs/merged

Several ``--adapter`` values are merged in order. The merged directory is what vLLM /
llama.cpp conversion / the Hub upload use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receipt_vlm.utils.io import resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger("merge_lora")


def base_of(adapter: str) -> str:
    cfg = json.loads((resolve(adapter) / "adapter_config.json").read_text())
    return str(cfg["base_model_name_or_path"])


def merge(base: str | None, adapters: list[str], out: str, dtype: str = "auto") -> Path:
    from peft import PeftModel
    from transformers import AutoProcessor

    from receipt_vlm.models.load import load_vlm

    base = base or base_of(adapters[0])
    model = load_vlm(base, None, False, dtype, device_map=None)
    for a in adapters:
        model = PeftModel.from_pretrained(model, str(resolve(a))).merge_and_unload()
        log.info("merged %s", a)
    out_dir = resolve(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    # the processor (chat template, image settings) of the last adapter wins, else the base
    proc_src = next(
        (
            str(resolve(a))
            for a in reversed(adapters)
            if (resolve(a) / "processor_config.json").exists()
        ),
        base,
    )
    AutoProcessor.from_pretrained(proc_src).save_pretrained(out_dir)
    (out_dir / "merge_info.json").write_text(
        json.dumps({"base": base, "adapters": adapters}, indent=2)
    )
    log.info("saved merged model to %s", out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base", default=None, help="base model (default: read from the first adapter)"
    )
    ap.add_argument("--adapter", action="append", required=True)
    ap.add_argument("--out", default="outputs/merged")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    merge(args.base, args.adapter, args.out, args.dtype)


if __name__ == "__main__":
    main()
