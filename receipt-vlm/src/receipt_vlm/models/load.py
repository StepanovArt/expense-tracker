"""Load Qwen3-VL (or any HF image-text-to-text model) + processor with a fixed image cap.

Qwen3-VL resizes images so that height and width are multiples of 32 (patch 16 x merge 2)
and the pixel count lies in ``[min_pixels, max_pixels]`` (``size.shortest_edge`` /
``size.longest_edge`` of ``Qwen2VLImageProcessor``). The cap is set explicitly for every
model so all models in a comparison see the same resolution (CLAUDE.md §5.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from receipt_vlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ImageCap:
    min_pixels: int = 64 * 32 * 32
    max_pixels: int = 1024 * 32 * 32  # ~1M px -> at most 1,024 visual tokens after 2x2 merge


def set_image_cap(processor: Any, cap: ImageCap) -> None:
    ip = processor.image_processor
    ip.size = {"shortest_edge": cap.min_pixels, "longest_edge": cap.max_pixels}
    for attr, val in (("min_pixels", cap.min_pixels), ("max_pixels", cap.max_pixels)):
        if hasattr(ip, attr):
            setattr(ip, attr, val)


def torch_dtype(name: str | None) -> Any:
    import torch

    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return getattr(torch, name)


def load_processor(model_name_or_path: str, cap: ImageCap, padding_side: str = "left") -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    set_image_cap(processor, cap)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = padding_side
    return processor


def load_vlm(
    model_name_or_path: str,
    adapter: str | None = None,
    load_in_4bit: bool = False,
    dtype: str | None = "auto",
    device_map: str | dict[str, Any] | None = "auto",
    attn_implementation: str | None = None,
) -> Any:
    import torch
    from transformers import AutoModelForImageTextToText

    kwargs: dict[str, Any] = {"dtype": torch_dtype(dtype)}
    if torch.cuda.is_available() and device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=kwargs["dtype"],
        )
    model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)
    if adapter:
        from peft import PeftModel

        if not Path(adapter).exists():
            raise FileNotFoundError(f"adapter not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)  # type: ignore[assignment]
        log.info("loaded LoRA adapter %s", adapter)
    model.eval()
    return model


def load_text_llm(
    model_name_or_path: str, dtype: str | None = "auto", load_in_4bit: bool = False
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    tok.padding_side = "left"
    kwargs: dict[str, Any] = {"dtype": torch_dtype(dtype)}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return model, tok
