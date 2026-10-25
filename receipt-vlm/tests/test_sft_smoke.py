"""End-to-end SFT wiring check: the tiny model must memorise two receipts.

If labels were misaligned, the prompt/image were trained on, or the chat template differed
between training and generation, greedy decoding would not reproduce the targets exactly.
"""

from pathlib import Path

import pytest

from receipt_vlm.data.record import LABEL_KEYS
from receipt_vlm.data.synthetic.generate import generate_one, spec_for
from receipt_vlm.utils.config import load_config
from receipt_vlm.utils.io import read_json, write_jsonl

ROOT = Path(__file__).parents[1]


@pytest.mark.slow
def test_tiny_sft_memorises_then_grpo_updates(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from receipt_vlm.models.tiny import build_tiny_qwen3_vl
    from receipt_vlm.train.sft import train_sft

    idx = [i for i in range(0, 600, 3) if len(spec_for(i, 1234)[0].items) == 1][:2]
    rows = []
    for i in idx:
        r = generate_one((i, 1234, str(tmp_path / "syn")))
        rows.append(
            {
                "id": f"synthetic/{r['source_id']}",
                "source": "synthetic",
                "language": "en",
                "image": r["image"],
                "target": r["target"],
                "labelled": list(LABEL_KEYS),
            }
        )
    write_jsonl(tmp_path / "data" / "train.jsonl", rows)
    write_jsonl(tmp_path / "data" / "val.jsonl", rows)
    from receipt_vlm.schema import dump_target

    # the tiny BPE sees the targets so they tokenise compactly, as with a real tokenizer
    model_dir = build_tiny_qwen3_vl(
        tmp_path / "tiny", extra_text=[dump_target(r["target"]) for r in rows] * 20
    )

    cfg = load_config(
        ROOT / "configs" / "sft" / "smoke.yaml",
        [
            f"model.name={model_dir}",
            f"output_dir={tmp_path / 'run'}",
            f"data.data_dir={tmp_path / 'data'}",
            "data.max_train_samples=null",
            "eval.val_subset=2",
            "eval.every_steps=125",
            "eval.max_new_tokens=300",
            "train.max_steps=250",
            "train.lr=3e-3",
            "train.grad_accum=1",
            "train.per_device_batch_size=2",
            "lora.r=64",
            "lora.alpha=64",
            "lora.dropout=0.0",
        ],
    )
    best = train_sft(cfg)
    sel = read_json(best / "selection.json")
    assert sel["val_overall_f1"] == 1.0 and sel["val_schema_valid_rate"] == 1.0
    assert (best / "adapter_model.safetensors").exists()

    # GRPO from the memorised SFT policy: near-greedy sampling gives a mix of exact and broken
    # completions, so group rewards vary, gradients flow and the policy moves away from SFT.
    from safetensors.torch import load_file

    from receipt_vlm.train.grpo import train_grpo
    from receipt_vlm.utils.io import read_jsonl

    gcfg = load_config(
        ROOT / "configs" / "grpo" / "smoke.yaml",
        [
            f"model.name={model_dir}",
            f"model.sft_adapter={best}",
            f"model.merged_sft_dir={tmp_path / 'merged'}",
            f"output_dir={tmp_path / 'grpo'}",
            f"data.data_dir={tmp_path / 'data'}",
            "data.max_train_samples=null",
            "grpo.max_completion_length=300",
            "grpo.temperature=0.08",
            "grpo.num_generations=4",
            "train.per_device_batch_size=4",
            "train.max_steps=3",
            "train.lr=1e-3",
            "eval.max_new_tokens=300",
            "eval.val_subset=2",
            "eval.every_steps=3",
        ],
    )
    train_grpo(gcfg)
    logs = [r for r in read_jsonl(tmp_path / "grpo" / "metrics.jsonl") if "reward" in r]
    assert max(r["reward"] for r in logs) >= 0.5  # exact completions are rewarded inside TRL
    assert any(r["reward_std"] > 0 for r in logs) and any(r["grad_norm"] > 0 for r in logs)
    assert all("null_rate/all" in r and "kl" in r for r in logs)
    lora_b = [
        v
        for k, v in load_file(tmp_path / "grpo" / "best" / "adapter_model.safetensors").items()
        if "lora_B" in k
    ]
    assert lora_b and max(float(v.abs().max()) for v in lora_b) > 0


@pytest.mark.slow
def test_grpo_without_sft_adapter_starts_from_the_base_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForImageTextToText

    from receipt_vlm.models.tiny import build_tiny_qwen3_vl
    from receipt_vlm.train.grpo import build_policy

    model_dir = build_tiny_qwen3_vl(tmp_path / "tiny")
    cfg = load_config(
        ROOT / "configs" / "grpo" / "smoke.yaml",
        [f"model.name={model_dir}", "model.sft_adapter=null", "model.merged_sft_dir=null"],
    )
    model, _ = build_policy(cfg)
    base = AutoModelForImageTextToText.from_pretrained(model_dir)
    got, ref = dict(model.named_parameters()), dict(base.named_parameters())
    assert got.keys() == ref.keys()
    assert all(torch.equal(got[k], ref[k]) for k in ref)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tiny"]  # no merged copy written


@pytest.mark.slow
@pytest.mark.parametrize("train_projector", [False, True])
def test_lora_targets_language_model_only(tmp_path: Path, train_projector: bool) -> None:
    pytest.importorskip("torch")
    from receipt_vlm.models.tiny import build_tiny_qwen3_vl
    from receipt_vlm.train.sft import build_model

    model_dir = build_tiny_qwen3_vl(tmp_path / "tiny")
    cfg = load_config(
        ROOT / "configs" / "sft" / "smoke.yaml",
        [f"model.name={model_dir}", f"lora.train_projector={str(train_projector).lower()}"],
    )
    model, _ = build_model(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in n or "merger" in n for n in trainable)
    lora = [n for n in trainable if "lora_" in n]
    assert all(".language_model." in n for n in lora)  # never inside the vision tower
    assert {
        m
        for n in lora
        for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        if m in n
    } == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert not any(".visual.blocks." in n for n in trainable)
    assert any("merger" in n for n in trainable) == train_projector
