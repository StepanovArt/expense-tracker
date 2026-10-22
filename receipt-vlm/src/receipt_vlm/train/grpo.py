"""GRPO on top of the best SFT checkpoint (CLAUDE.md §6.2), with TRL's GRPOTrainer.

* The SFT adapter is **merged** into the base weights first and GRPO trains a fresh LoRA on
  top, so TRL's reference policy (the same model with the adapter disabled) is the SFT
  policy: the logged ``kl`` is KL(policy || SFT), as §6.2 asks. The merged SFT model is saved
  once (``model.merged_sft_dir``) so M2 = merged SFT + GRPO adapter at evaluation time.
* Reward = eval-code reward (``rewards/grpo_rewards.py``); per-component means, schema-valid
  rate, output length and per-field null rates are logged every step.
* A monitor flags reward hacking: if the null rate rises while the reward rises over the
  recent window, it writes ``reward_hacking_warning.json`` and logs a warning.
* Validation F1 (greedy, fixed subset) is computed every ``eval.every_steps``; the best
  adapter by val F1 is kept in ``output_dir/best``.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from receipt_vlm.utils.io import read_jsonl, resolve, write_json
from receipt_vlm.utils.logging import MetricsLogger, get_logger

log = get_logger(__name__)


def build_policy(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """(merged SFT model, processor)."""
    from receipt_vlm.models.load import ImageCap, load_processor, load_vlm

    mcfg = cfg["model"]
    cap = ImageCap(**cfg["image"])
    merged_dir = resolve(mcfg["merged_sft_dir"])
    if (merged_dir / "config.json").exists():
        log.info("using merged SFT model %s", merged_dir)
        return load_vlm(str(merged_dir), None, False, mcfg.get("dtype", "auto")), load_processor(
            str(merged_dir), cap
        )
    base = load_vlm(mcfg["name"], mcfg.get("sft_adapter"), False, mcfg.get("dtype", "auto"))
    if mcfg.get("sft_adapter"):
        base = base.merge_and_unload()
        log.info("merged SFT adapter %s into %s", mcfg["sft_adapter"], mcfg["name"])
    processor = load_processor(mcfg.get("sft_adapter") or mcfg["name"], cap)
    merged_dir.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    return base, processor


def build_dataset(cfg: dict[str, Any]) -> Any:
    """Conversational prompts with an image placeholder + an ``image`` column (TRL fills it)."""
    from datasets import Dataset, Image

    from receipt_vlm.eval.runner import select_subset
    from receipt_vlm.prompts import INSTRUCTION, SYSTEM_PROMPT

    d = cfg["data"]
    rows = read_jsonl(resolve(d["data_dir"]) / "train.jsonl")
    if d.get("sources"):
        rows = [r for r in rows if r["source"] in set(d["sources"])]
    rows = select_subset(rows, d.get("max_train_samples"), int(cfg["seed"]))
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
    ]
    records = [
        {
            "prompt": prompt,
            "image": str(resolve(r["image"])),
            "id": r["id"],
            "target": json.dumps(r["target"], ensure_ascii=False),
            "labelled": json.dumps(r["labelled"]),
            "string_compare": r.get("string_compare", "default"),
        }
        for r in rows
    ]
    return Dataset.from_list(records).cast_column("image", Image())


class HackingMonitor:
    """Flags 'null rate up while reward up' over a sliding window of logged steps."""

    def __init__(self, window: int = 10, min_rise: float = 0.05) -> None:
        self.window = window
        self.min_rise = min_rise
        self.rewards: list[float] = []
        self.nulls: list[float] = []

    def update(self, logs: dict[str, Any]) -> dict[str, float] | None:
        r = logs.get("reward")
        n = next((logs[k] for k in logs if k.endswith("null_rate/all")), None)
        if r is None or n is None:
            return None
        self.rewards.append(float(r))
        self.nulls.append(float(n))
        if len(self.rewards) < 2 * self.window:
            return None
        w = self.window
        dr = sum(self.rewards[-w:]) / w - sum(self.rewards[-2 * w : -w]) / w
        dn = sum(self.nulls[-w:]) / w - sum(self.nulls[-2 * w : -w]) / w
        if dr > 0 and dn > self.min_rise:
            return {"reward_rise": dr, "null_rate_rise": dn}
        return None


def make_callbacks(
    cfg: dict[str, Any],
    processor: Any,
    val: list[dict[str, Any]],
    out_dir: Path,
    mlog: MetricsLogger,
) -> list[Any]:
    from transformers import TrainerCallback

    from receipt_vlm.train.sft import generation_eval

    every = int(cfg["eval"]["every_steps"])
    monitor = HackingMonitor(int(cfg["monitor"]["window"]), float(cfg["monitor"]["min_null_rise"]))

    class GRPOCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best = -math.inf
            self.history: list[dict[str, Any]] = []
            self.last = -1

        def _evaluate(self, model: Any, step: int) -> None:
            if step == self.last:
                return
            self.last = step
            tag = f"step_{step}"
            t0 = time.time()
            m = generation_eval(model, processor, val, cfg, out_dir / "val_preds" / f"{tag}.jsonl")
            ckpt = out_dir / "checkpoints" / tag
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            row = {
                "tag": tag,
                "step": step,
                "eval_s": round(time.time() - t0, 1),
                **{f"val_{k}": v for k, v in m.items()},
            }
            self.history.append(row)
            mlog.log(row, step=step)
            log.info("val %s: %s", tag, json.dumps(row))
            if not math.isnan(m["overall_f1"]) and m["overall_f1"] > self.best:
                self.best = m["overall_f1"]
                best = out_dir / "best"
                shutil.rmtree(best, ignore_errors=True)
                shutil.copytree(ckpt, best)
                write_json(
                    best / "selection.json", {"selected_by": "val overall_f1 (generation)", **row}
                )
            write_json(out_dir / "val_history.json", self.history)

        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any
        ) -> None:
            if not logs:
                return
            mlog.log(logs, step=state.global_step)
            flag = monitor.update(logs)
            if flag:
                log.warning("possible reward hacking at step %d: %s", state.global_step, flag)
                write_json(
                    out_dir / "reward_hacking_warning.json", {"step": state.global_step, **flag}
                )

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if state.global_step % every == 0:
                self._evaluate(kw["model"], state.global_step)

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            self._evaluate(kw["model"], state.global_step)

    return [GRPOCallback()]


def train_grpo(cfg: dict[str, Any]) -> Path:
    import torch
    from peft import LoraConfig
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer

    from receipt_vlm.eval.runner import load_split, select_subset
    from receipt_vlm.rewards.grpo_rewards import WEIGHTS, make_trl_reward_funcs
    from receipt_vlm.train.sft import LM_LORA_TARGETS
    from receipt_vlm.train.trl_compat import install_cpu_logprob_fallback
    from receipt_vlm.utils.seed import seed_everything

    seed = int(cfg["seed"])
    seed_everything(seed)
    install_cpu_logprob_fallback()  # no-op on GPU
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", cfg)
    mlog = MetricsLogger(out_dir, cfg.get("run_name", "grpo"), cfg)

    model, processor = build_policy(cfg)
    if cfg["train"].get("gradient_checkpointing", True):
        model.config.use_cache = False
    ds = build_dataset(cfg)
    val = select_subset(
        load_split(cfg["data"]["data_dir"], "val"), int(cfg["eval"]["val_subset"]), seed
    )
    log.info("GRPO train prompts: %d, val subset: %d", len(ds), len(val))

    g, t = cfg["grpo"], cfg["train"]
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    args = GRPOConfig(
        output_dir=str(out_dir / "trainer"),
        learning_rate=float(t["lr"]),
        per_device_train_batch_size=int(t["per_device_batch_size"]),
        gradient_accumulation_steps=int(t["grad_accum"]),
        max_steps=int(t["max_steps"]),
        lr_scheduler_type=t.get("scheduler", "constant_with_warmup"),
        warmup_steps=int(t.get("warmup_steps", 0)),
        logging_steps=int(t.get("logging_steps", 1)),
        num_generations=int(g["num_generations"]),
        max_completion_length=int(g["max_completion_length"]),
        temperature=float(g["temperature"]),
        beta=float(g["beta"]),
        loss_type=g.get("loss_type", "dapo"),
        mask_truncated_completions=bool(g.get("mask_truncated_completions", True)),
        use_vllm=bool(g.get("use_vllm", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)) and cuda,
        bf16=bf16,
        fp16=cuda and not bf16,
        use_cpu=not cuda,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        log_completions=False,
    )
    lora = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=LM_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    weights = cfg.get("reward_weights") or WEIGHTS
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_trl_reward_funcs(weights),
        args=args,
        train_dataset=ds,
        processing_class=processor,
        peft_config=lora,
        callbacks=make_callbacks(cfg, processor, val, out_dir, mlog),
    )
    t0 = time.time()
    trainer.train()
    write_json(
        out_dir / "train_info.json",
        {
            "train_seconds": round(time.time() - t0, 1),
            "n_prompts": len(ds),
            "merged_sft_dir": cfg["model"]["merged_sft_dir"],
        },
    )
    mlog.finish()
    return out_dir / "best"
