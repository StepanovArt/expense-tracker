"""scripts/merge_lora.py: merged weights must reproduce base + adapter(s)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.slow
def test_merge_matches_adapter_outputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, PeftModel, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from receipt_vlm.models.tiny import build_tiny_qwen3_vl
    from receipt_vlm.prompts import vlm_messages
    from receipt_vlm.train.sft import LM_LORA_TARGETS

    base = build_tiny_qwen3_vl(tmp_path / "tiny")
    adapters = []
    for k in range(2):  # two stacked adapters, like SFT then GRPO
        src = base if k == 0 else tmp_path / "m0"
        m = AutoModelForImageTextToText.from_pretrained(src)
        torch.manual_seed(k)
        pm = get_peft_model(
            m,
            LoraConfig(r=4, lora_alpha=8, target_modules=LM_LORA_TARGETS, init_lora_weights=False),
        )
        pm.save_pretrained(tmp_path / f"a{k}")
        AutoProcessor.from_pretrained(base).save_pretrained(tmp_path / f"a{k}")
        adapters.append(str(tmp_path / f"a{k}"))
        if k == 0:
            pm.merge_and_unload().save_pretrained(tmp_path / "m0")

    spec = importlib.util.spec_from_file_location("merge_lora", ROOT / "scripts" / "merge_lora.py")
    ml = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ml)  # type: ignore[union-attr]
    out = ml.merge(str(base), adapters, str(tmp_path / "merged"), dtype="float32")

    proc = AutoProcessor.from_pretrained(out)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (96, 64, 3), dtype=np.uint8))
    inputs = proc.apply_chat_template(
        vlm_messages(img),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    ref = AutoModelForImageTextToText.from_pretrained(base)
    for a in adapters:
        ref = PeftModel.from_pretrained(ref, a).merge_and_unload()  # reference path
    unmerged = PeftModel.from_pretrained(
        AutoModelForImageTextToText.from_pretrained(base), adapters[0]
    )
    merged = AutoModelForImageTextToText.from_pretrained(out)
    with torch.no_grad():
        a = merged(**inputs).logits
        b = ref(**inputs).logits
        c = unmerged.merge_and_unload()(**inputs).logits
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, c, atol=1e-3)  # the second adapter really changed the model
