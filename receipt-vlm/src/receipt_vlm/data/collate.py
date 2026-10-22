"""Tokenisation + label masking for SFT, and the batch collator.

Loss is computed **only on the assistant JSON** (plus its closing ``<|im_end|>``): the system
prompt, the user instruction and every image token get label -100 (CLAUDE.md §6.1). On top of
that, with ``mask_unlabelled=True`` the *values* of fields a sample's source does not label
(e.g. ``"items":[]`` for SROIE, item quantities for WildReceipt) are also masked: they are
null in the target only because the source does not annotate them, and training on them would
teach the model to suppress fields it can actually see (DECISIONS.md). Keys, punctuation and
the values of labelled fields are always trained.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from receipt_vlm.schema import FIELD_ORDER, ITEM_FIELDS, canonical_dict

IGNORE_INDEX = -100
END_OF_TURN = "<|im_end|>"


def _dump(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def serialize_with_spans(
    target: dict[str, Any], labelled: Iterable[str]
) -> tuple[str, list[tuple[int, int]]]:
    """Target JSON (identical to ``schema.dump_target``) + char spans of unlabelled values."""
    lab = set(labelled)
    data = canonical_dict(target)
    parts: list[str] = ["{"]
    pos = 1
    spans: list[tuple[int, int]] = []

    def emit(s: str, masked: bool) -> None:
        nonlocal pos
        parts.append(s)
        if masked and s:
            spans.append((pos, pos + len(s)))
        pos += len(s)

    for i, key in enumerate(FIELD_ORDER):
        emit(("," if i else "") + _dump(key) + ":", False)
        val = data[key]
        if key != "items":
            emit(_dump(val), key not in lab)
            continue
        if "items.name" not in lab:
            emit(_dump(val), True)
            continue
        emit("[", False)
        for j, item in enumerate(val):
            emit(("," if j else "") + "{", False)
            for k, attr in enumerate(ITEM_FIELDS):
                emit(("," if k else "") + _dump(attr) + ":", False)
                emit(_dump(item[attr]), f"items.{attr}" not in lab)
            emit("}", False)
        emit("]", False)
    emit("}", False)
    return "".join(parts), spans


def _overlaps(a: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    return any(a[0] < e and s < a[1] for s, e in spans)


def encode_example(
    processor: Any,
    image: Any,
    target: dict[str, Any],
    labelled: Iterable[str],
    mask_unlabelled: bool = True,
) -> dict[str, Any]:
    """One training example -> tensors (batch dim 1 removed for sequence tensors)."""
    import torch

    from receipt_vlm.prompts import vlm_messages

    prompt = processor.apply_chat_template(
        vlm_messages(image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    text, spans = serialize_with_spans(target, labelled)
    completion = text + END_OF_TURN
    tok = processor.tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids: list[int] = list(tok["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in tok["offset_mapping"]]
    comp_labels = [
        IGNORE_INDEX if (mask_unlabelled and _overlaps(o, spans)) else t
        for t, o in zip(comp_ids, offsets, strict=True)
    ]
    p_ids = prompt["input_ids"][0]
    input_ids = torch.cat([p_ids, torch.tensor(comp_ids, dtype=p_ids.dtype)])
    labels = torch.cat(
        [torch.full_like(p_ids, IGNORE_INDEX), torch.tensor(comp_labels, dtype=p_ids.dtype)]
    )
    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "pixel_values": prompt["pixel_values"],
        "image_grid_thw": prompt["image_grid_thw"],
    }
    if "mm_token_type_ids" in prompt:
        out["mm_token_type_ids"] = torch.cat(
            [
                prompt["mm_token_type_ids"][0],
                torch.zeros(len(comp_ids), dtype=prompt["mm_token_type_ids"].dtype),
            ]
        )
    return out


class SFTCollator:
    """Right-pads sequences; concatenates image patches (Qwen-VL packs them along dim 0)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        n = max(int(f["input_ids"].shape[0]) for f in features)

        def pad(key: str, value: int) -> Any:
            rows = []
            for f in features:
                x = f[key]
                rows.append(torch.cat([x, torch.full((n - x.shape[0],), value, dtype=x.dtype)]))
            return torch.stack(rows)

        batch = {
            "input_ids": pad("input_ids", self.pad_token_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", IGNORE_INDEX),
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
        }
        if all("mm_token_type_ids" in f for f in features):
            batch["mm_token_type_ids"] = pad("mm_token_type_ids", 0)
        return batch


def check_serialization_matches(target: dict[str, Any]) -> bool:
    from receipt_vlm.schema import dump_target

    return serialize_with_spans(target, [])[0] == dump_target(target)
