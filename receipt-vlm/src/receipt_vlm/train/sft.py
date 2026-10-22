"""LoRA / QLoRA supervised fine-tuning of Qwen3-VL (CLAUDE.md §6.1).

* LoRA on the language model's attention + MLP projections only; the vision tower is frozen.
  The vision->LM projector (``visual.merger`` + deepstack mergers) is optionally trainable
  (``lora.train_projector``) - compare both on validation F1.
* Loss only on assistant tokens (+ partial-label masking, see ``data/collate.py``).
* Checkpoints are selected by **generation-based overall F1 on a fixed validation subset**,
  evaluated every epoch (or every ``eval.every_steps``), never by loss.
* Uses plain transformers ``Trainer`` + peft with our own collator (full control over label
  masking); ``model.use_unsloth: true`` switches to Unsloth's FastVisionModel on GPU.
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

LM_LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def projector_modules(model: Any) -> list[str]:
    """The vision->LM projector: ``visual.merger`` + each deepstack merger (peft cannot wrap
    the ModuleList itself, so its entries are listed individually)."""
    import re

    names = []
    for n, _ in model.named_modules():
        if n.endswith("visual.merger") or re.search(r"visual\.deepstack_merger_list\.\d+$", n):
            names.append("visual." + n.split("visual.", 1)[1])
    return sorted(set(names))


class ReceiptSFTDataset:
    """Lazily encodes samples (image load + processor + label masking) in __getitem__."""

    def __init__(
        self, samples: list[dict[str, Any]], processor: Any, mask_unlabelled: bool
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.mask_unlabelled = mask_unlabelled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        from receipt_vlm.data.collate import encode_example
        from receipt_vlm.models.inference import load_image

        s = self.samples[i]
        return encode_example(
            self.processor, load_image(s["image"]), s["target"], s["labelled"], self.mask_unlabelled
        )


def load_train_samples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    d = cfg["data"]
    rows = read_jsonl(resolve(d["data_dir"]) / f"{d.get('train_split', 'train')}.jsonl")
    if d.get("sources"):
        rows = [r for r in rows if r["source"] in set(d["sources"])]
    if d.get("max_train_samples"):
        from receipt_vlm.eval.runner import select_subset

        rows = select_subset(rows, int(d["max_train_samples"]), int(cfg["seed"]))
    return rows


def build_model(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """(peft model, processor)."""
    from receipt_vlm.models.load import ImageCap, load_processor

    mcfg, lcfg = cfg["model"], cfg["lora"]
    cap = ImageCap(**cfg["image"])
    if mcfg.get("use_unsloth"):
        try:
            return _build_unsloth(cfg, cap)
        except Exception as e:  # documented fallback (CLAUDE.md §0)
            log.warning("Unsloth path failed (%s); falling back to transformers + peft", e)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from receipt_vlm.models.load import load_vlm

    processor = load_processor(mcfg["name"], cap, padding_side="left")
    model = load_vlm(
        mcfg["name"],
        None,
        bool(mcfg.get("load_in_4bit", False)),
        mcfg.get("dtype", "auto"),
        attn_implementation=mcfg.get("attn_implementation"),
    )
    gc = bool(cfg["train"].get("gradient_checkpointing", True))
    if mcfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc)
    elif gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        target_modules=LM_LORA_TARGETS,
        modules_to_save=projector_modules(model) if lcfg.get("train_projector") else None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    _assert_vision_frozen(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable params: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)
    return model, processor


def _assert_vision_frozen(model: Any) -> None:
    bad = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and ".visual." in n and not any(m in n for m in ("merger",))
    ]
    if bad:
        raise RuntimeError(f"vision tower parameters are trainable: {bad[:5]}")


def _build_unsloth(cfg: dict[str, Any], cap: Any) -> tuple[Any, Any]:
    from unsloth import FastVisionModel  # GPU only

    from receipt_vlm.models.load import set_image_cap

    mcfg, lcfg = cfg["model"], cfg["lora"]
    model, processor = FastVisionModel.from_pretrained(
        mcfg["name"],
        load_in_4bit=bool(mcfg.get("load_in_4bit", False)),
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        bias="none",
        random_state=int(cfg["seed"]),
    )
    set_image_cap(processor, cap)
    processor.tokenizer.padding_side = "left"
    log.info("using Unsloth FastVisionModel")
    return model, processor


def generation_eval(
    model: Any, processor: Any, samples: list[dict[str, Any]], cfg: dict[str, Any], out_path: Path
) -> dict[str, float]:
    """Greedy-decode the fixed val subset with the current weights; return headline metrics."""
    from receipt_vlm.eval.metrics import ScoreTable, compute
    from receipt_vlm.eval.runner import run_predictions, score_predictions
    from receipt_vlm.models.inference import HFVisionGenerator
    from receipt_vlm.models.load import ImageCap

    was_training = model.training
    model.eval()
    use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    try:
        gen = HFVisionGenerator(
            "",
            max_new_tokens=int(cfg["eval"]["max_new_tokens"]),
            cap=ImageCap(**cfg["image"]),
            name="sft-eval",
            model_obj=model,
            processor_obj=processor,
        )
        out_path.unlink(missing_ok=True)
        run_predictions(gen, samples, out_path, int(cfg["eval"].get("batch_size", 4)))
        scores = score_predictions(samples, read_jsonl(out_path))
    finally:
        model.config.use_cache = use_cache
        if was_training:
            model.train()
    return compute(
        ScoreTable.from_scores(scores),
        ["overall_f1", "schema_valid_rate", "total_em", "parse_rate"],
    )


def make_callback(
    model_ref: dict[str, Any],
    processor: Any,
    val: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    mlog: MetricsLogger,
) -> Any:
    from transformers import TrainerCallback

    every_steps = cfg["eval"].get("every_steps")

    class GenEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best = -math.inf
            self.history: list[dict[str, Any]] = []
            self.last_eval_step = -1

        def _evaluate(self, state: Any, tag: str) -> None:
            if state.global_step == self.last_eval_step:
                return
            self.last_eval_step = state.global_step
            model = model_ref["model"]
            t0 = time.time()
            m = generation_eval(model, processor, val, cfg, out_dir / "val_preds" / f"{tag}.jsonl")
            ckpt = out_dir / "checkpoints" / tag
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            row = {
                "tag": tag,
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_s": round(time.time() - t0, 1),
                **{f"val_{k}": v for k, v in m.items()},
            }
            self.history.append(row)
            mlog.log(row, step=state.global_step)
            log.info("val %s: %s", tag, json.dumps(row))
            f1 = m["overall_f1"]
            if not math.isnan(f1) and f1 > self.best:
                self.best = f1
                best = out_dir / "best"
                shutil.rmtree(best, ignore_errors=True)
                shutil.copytree(ckpt, best)
                write_json(
                    best / "selection.json", {"selected_by": "val overall_f1 (generation)", **row}
                )
            write_json(out_dir / "val_history.json", self.history)

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if not every_steps:
                self._evaluate(state, f"epoch_{round(state.epoch or 0)}")

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if every_steps and state.global_step % int(every_steps) == 0:
                self._evaluate(state, f"step_{state.global_step}")

        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any
        ) -> None:
            if logs:
                mlog.log({f"train_{k}": v for k, v in logs.items()}, step=state.global_step)

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if not self.history or self.last_eval_step != state.global_step:
                self._evaluate(state, f"step_{state.global_step}")

    return GenEvalCallback()


def train_sft(cfg: dict[str, Any]) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments

    from receipt_vlm.data.collate import SFTCollator
    from receipt_vlm.eval.runner import load_split, select_subset
    from receipt_vlm.utils.seed import seed_everything

    seed = int(cfg["seed"])
    seed_everything(seed)
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", cfg)
    mlog = MetricsLogger(out_dir, cfg.get("run_name", out_dir.name), cfg)

    model, processor = build_model(cfg)
    train_rows = load_train_samples(cfg)
    val = select_subset(
        load_split(cfg["data"]["data_dir"], "val"), int(cfg["eval"]["val_subset"]), seed
    )
    log.info("train samples: %d, val subset: %d", len(train_rows), len(val))
    ds = ReceiptSFTDataset(train_rows, processor, bool(cfg["data"].get("mask_unlabelled", True)))

    t = cfg["train"]
    per_dev, accum = int(t["per_device_batch_size"]), int(t["grad_accum"])
    steps_per_epoch = math.ceil(len(ds) / (per_dev * accum))
    total_steps = int(t["max_steps"]) if t.get("max_steps") else steps_per_epoch * int(t["epochs"])
    cuda = torch.cuda.is_available()
    bf16 = (
        cuda and torch.cuda.is_bf16_supported() and t.get("precision", "auto") in ("auto", "bf16")
    )
    fp16 = cuda and not bf16 and t.get("precision", "auto") in ("auto", "fp16")
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=accum,
        learning_rate=float(t["lr"]),
        num_train_epochs=float(t.get("epochs", 1)),
        max_steps=int(t["max_steps"]) if t.get("max_steps") else -1,
        lr_scheduler_type=t.get("scheduler", "cosine"),
        warmup_steps=max(1, round(float(t.get("warmup_ratio", 0.03)) * total_steps)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_strategy="no",  # our callback saves adapters selected by val F1
        eval_strategy="no",
        bf16=bf16,
        fp16=fp16,
        use_cpu=not cuda,
        gradient_checkpointing=False,  # enabled on the model in build_model
        remove_unused_columns=False,
        dataloader_num_workers=int(t.get("dataloader_num_workers", 2 if cuda else 0)),
        report_to=[],
        seed=seed,
        data_seed=seed,
        optim=t.get("optim", "adamw_torch"),
    )
    pad_id = processor.tokenizer.pad_token_id
    model_ref = {"model": model}
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=SFTCollator(pad_id),
        callbacks=[make_callback(model_ref, processor, val, cfg, out_dir, mlog)],
    )
    t0 = time.time()
    trainer.train()
    write_json(
        out_dir / "train_info.json",
        {
            "train_seconds": round(time.time() - t0, 1),
            "total_steps": total_steps,
            "n_train": len(ds),
            "steps_per_epoch": steps_per_epoch,
        },
    )
    mlog.finish()
    best = out_dir / "best"
    log.info("best adapter (by val F1): %s", best)
    return best
