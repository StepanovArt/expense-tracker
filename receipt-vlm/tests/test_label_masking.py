"""SFT label masking: loss only on the assistant JSON (CLAUDE.md §6.1), checked on real
tokenised examples from the tiny Qwen3-VL processor (same classes as Qwen3-VL-2B)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from receipt_vlm.data.collate import IGNORE_INDEX, serialize_with_spans
from receipt_vlm.schema import dump_target

SROIE_TARGET = {
    "merchant": "BOOK TA .K (TAMAN DAYA) SDN BHD",
    "date": "2018-12-25",
    "total": 9.0,
    "items": [],
}
SROIE_LABELLED = ["merchant", "date", "total"]
FULL_TARGET = {
    "merchant": "كارفور",
    "date": "2024-03-01",
    "time": "09:05",
    "currency": "AED",
    "items": [
        {"name": "Молоко", "quantity": 2, "unit_price": 3.5, "total": 7.0},
        {"name": "Bread", "quantity": None, "unit_price": None, "total": 2.25},
    ],
    "subtotal": 9.25,
    "tax": 0.46,
    "total": 9.71,
    "payment_method": "card",
    "language": "ar",
}


@pytest.mark.parametrize("target", [SROIE_TARGET, FULL_TARGET, {}])
def test_serialization_identical_to_target_json(target: dict) -> None:
    for labelled in ([], SROIE_LABELLED, ["items.name", "items.total", "total"]):
        assert serialize_with_spans(target, labelled)[0] == dump_target(target)


def test_unlabelled_spans_cover_exactly_unlabelled_values() -> None:
    text, spans = serialize_with_spans(SROIE_TARGET, SROIE_LABELLED)
    masked = [text[s:e] for s, e in spans]
    # time, currency, items, subtotal, tax, payment_method, language are not labelled by SROIE
    assert masked == ["null", "null", "[]", "null", "null", "null", "null"]
    text, spans = serialize_with_spans(FULL_TARGET, ["items.name", "items.total", "total"])
    masked = [text[s:e] for s, e in spans]
    # item quantity / unit_price masked per item; every scalar but total masked
    assert masked.count("2") == 1 and masked.count("3.5") == 1
    assert '"كارفور"' in masked and "9.71" not in masked and '"Молоко"' not in masked


@pytest.fixture(scope="module")
def processor(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    from transformers import AutoProcessor

    from receipt_vlm.models.load import ImageCap, set_image_cap
    from receipt_vlm.models.tiny import build_tiny_qwen3_vl

    path = build_tiny_qwen3_vl(
        tmp_path_factory.mktemp("tiny"), extra_text=[dump_target(FULL_TARGET)]
    )
    proc = AutoProcessor.from_pretrained(path)
    set_image_cap(proc, ImageCap(min_pixels=4096, max_pixels=65536))
    proc.model_path = path
    return proc


def _image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (200, 120, 3), dtype=np.uint8))


@pytest.mark.slow
def test_only_assistant_tokens_are_labelled(processor) -> None:  # type: ignore[no-untyped-def]
    from receipt_vlm.data.collate import encode_example

    ex = encode_example(processor, _image(), SROIE_TARGET, SROIE_LABELLED, mask_unlabelled=False)
    ids, labels = ex["input_ids"].tolist(), ex["labels"].tolist()
    tok = processor.tokenizer
    # locate the start of the assistant answer: right after the last "<|im_start|>assistant\n"
    prompt_len = len(
        processor.apply_chat_template(
            __import__("receipt_vlm.prompts", fromlist=["vlm_messages"]).vlm_messages(_image()),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"][0]
    )
    assert all(lab == IGNORE_INDEX for lab in labels[:prompt_len]), (
        "system/user/image tokens must be -100"
    )
    image_id = tok.convert_tokens_to_ids("<|image_pad|>")
    image_positions = [i for i, t in enumerate(ids) if t == image_id]
    assert image_positions and all(labels[i] == IGNORE_INDEX for i in image_positions)
    # every assistant token is trained, labels equal the inputs (shift happens inside the model)
    assert labels[prompt_len:] == ids[prompt_len:]
    assert tok.decode(ids[prompt_len:]) == dump_target(SROIE_TARGET) + "<|im_end|>"
    assert labels[-1] == tok.convert_tokens_to_ids("<|im_end|>")


@pytest.mark.slow
def test_unlabelled_values_are_masked_but_labelled_values_are_not(processor) -> None:  # type: ignore[no-untyped-def]
    from receipt_vlm.data.collate import encode_example

    ex = encode_example(processor, _image(), SROIE_TARGET, SROIE_LABELLED, mask_unlabelled=True)
    tok = processor.tokenizer
    ids, labels = ex["input_ids"].tolist(), ex["labels"].tolist()
    first = next(i for i, lab in enumerate(labels) if lab != IGNORE_INDEX)
    trained = tok.decode(
        [t for t, lab in zip(ids[first:], labels[first:], strict=True) if lab != IGNORE_INDEX]
    )
    masked = tok.decode(
        [t for t, lab in zip(ids[first:], labels[first:], strict=True) if lab == IGNORE_INDEX]
    )
    # labelled values are trained...
    for v in ('"BOOK TA .K (TAMAN DAYA) SDN BHD"', '"2018-12-25"', '"total":9,'):
        assert v in trained, v
    # ...the unlabelled values (7 x null/[] ) are not, and all keys are still trained
    assert masked.count("null") == 6 and "[]" in masked
    assert "null" not in trained and "[]" not in trained
    for key in (
        "merchant",
        "date",
        "time",
        "currency",
        "items",
        "subtotal",
        "tax",
        "total",
        "payment_method",
        "language",
    ):
        assert f'"{key}"' in trained


@pytest.mark.slow
def test_collator_and_forward_loss(processor) -> None:  # type: ignore[no-untyped-def]
    import torch
    from transformers import AutoModelForImageTextToText

    from receipt_vlm.data.collate import SFTCollator, encode_example

    a = encode_example(processor, _image(), SROIE_TARGET, SROIE_LABELLED)
    b = encode_example(
        processor, _image().resize((100, 300)), FULL_TARGET, ["items.name", "items.total", "total"]
    )
    batch = SFTCollator(processor.tokenizer.pad_token_id)([a, b])
    n = max(len(a["input_ids"]), len(b["input_ids"]))
    assert batch["input_ids"].shape == (2, n) == batch["labels"].shape
    shorter = 0 if len(a["input_ids"]) < len(b["input_ids"]) else 1
    assert (
        batch["labels"][shorter, -1] == IGNORE_INDEX and batch["attention_mask"][shorter, -1] == 0
    )
    assert batch["pixel_values"].shape[0] == a["pixel_values"].shape[0] + b["pixel_values"].shape[0]
    assert batch["image_grid_thw"].shape == (2, 3)
    model = AutoModelForImageTextToText.from_pretrained(Path(processor.model_path))
    out = model(**batch)
    assert torch.isfinite(out.loss)
