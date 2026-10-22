"""Batched inference backends. Every backend uses the same prompt (prompts.py), greedy decoding
and the same max-new-tokens / image cap from the eval config, and returns raw text + latency.

Backends
  hf        transformers ``generate`` (CPU/GPU, optional LoRA adapter / 4-bit)
  vllm      offline ``vllm.LLM.chat`` (GPU; fastest; used for latency/throughput numbers)
  openai    any OpenAI-compatible server (vLLM server, llama-server)
  ocr_llm   Tesseract OCR -> text LLM (baseline B0; the LLM is hf or openai)
  mock      fixed outputs, for tests
"""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

from receipt_vlm.models.load import ImageCap
from receipt_vlm.prompts import vlm_messages
from receipt_vlm.utils.io import resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class GenOutput:
    raw: str
    latency_s: float  # wall time for this sample (batch time / batch size when batched)
    n_output_tokens: int | None = None
    extra: dict[str, Any] | None = None


class Generator(Protocol):
    name: str

    def generate(self, images: Sequence[Image.Image]) -> list[GenOutput]: ...


def load_image(path: str) -> Image.Image:
    from PIL import ImageOps

    return ImageOps.exif_transpose(Image.open(resolve(path))).convert("RGB")


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class HFVisionGenerator:
    def __init__(
        self,
        model: str,
        adapter: str | None = None,
        load_in_4bit: bool = False,
        dtype: str | None = "auto",
        max_new_tokens: int = 1024,
        cap: ImageCap | None = None,
        name: str = "hf",
        model_obj: Any = None,
        processor_obj: Any = None,
    ) -> None:
        from receipt_vlm.models.load import load_processor, load_vlm

        self.name = name
        self.cap = cap or ImageCap()
        self.processor = processor_obj or load_processor(model, self.cap)
        self.model = model_obj or load_vlm(model, adapter, load_in_4bit, dtype)
        self.max_new_tokens = max_new_tokens

    def generate(self, images: Sequence[Image.Image]) -> list[GenOutput]:
        import torch

        convs = [vlm_messages(im) for im in images]
        inputs = self.processor.apply_chat_template(
            convs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(self.model.device)
        _sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1
            )
        _sync()
        dt = (time.perf_counter() - t0) / len(images)
        new = out[:, inputs["input_ids"].shape[1] :]
        texts = self.processor.batch_decode(new, skip_special_tokens=True)
        pad = self.processor.tokenizer.pad_token_id
        return [
            GenOutput(t, dt, int((row != pad).sum())) for t, row in zip(texts, new, strict=True)
        ]


class VLLMVisionGenerator:
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 1024,
        cap: ImageCap | None = None,
        name: str = "vllm",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        quantization: str | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.name = name
        cap = cap or ImageCap()
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs={"min_pixels": cap.min_pixels, "max_pixels": cap.max_pixels},
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            seed=0,
        )
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(self, images: Sequence[Image.Image]) -> list[GenOutput]:
        convs = [_openai_messages(im) for im in images]
        t0 = time.perf_counter()
        outs = self.llm.chat(convs, self.params, use_tqdm=False)
        dt = (time.perf_counter() - t0) / len(images)
        return [GenOutput(o.outputs[0].text, dt, len(o.outputs[0].token_ids)) for o in outs]


def _image_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _openai_messages(im: Image.Image) -> list[dict[str, Any]]:
    """prompts.vlm_messages in OpenAI chat format (image as a data URL)."""
    msgs = vlm_messages(im)
    out: list[dict[str, Any]] = []
    for m in msgs:
        parts = []
        for c in m["content"]:
            if c["type"] == "image":
                parts.append(
                    {"type": "image_url", "image_url": {"url": _image_data_url(c["image"])}}
                )
            else:
                parts.append({"type": "text", "text": c["text"]})
        out.append({"role": m["role"], "content": parts})
    return out


class OpenAIVisionGenerator:
    """Talks to an OpenAI-compatible /v1/chat/completions endpoint (vLLM or llama-server)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_new_tokens: int = 1024,
        name: str = "openai",
        timeout: float = 300.0,
        api_key: str = "EMPTY",
    ) -> None:
        import httpx

        self.name = name
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def chat(self, messages: list[dict[str, Any]]) -> GenOutput:
        t0 = time.perf_counter()
        r = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_new_tokens,
            },
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return GenOutput(
            data["choices"][0]["message"]["content"] or "",
            time.perf_counter() - t0,
            usage.get("completion_tokens"),
        )

    def generate(self, images: Sequence[Image.Image]) -> list[GenOutput]:
        return [self.chat(_openai_messages(im)) for im in images]


class HFTextLLM:
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 1024,
        dtype: str | None = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        from receipt_vlm.models.load import load_text_llm

        self.model, self.tok = load_text_llm(model, dtype, load_in_4bit)
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict[str, Any]]) -> tuple[str, int]:
        import torch

        # Qwen3 hybrid models: thinking off, so the output is the answer only (same as VLMs)
        ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, ids["input_ids"].shape[1] :]
        return self.tok.decode(new, skip_special_tokens=True), int(new.shape[0])


class MockGenerator:
    def __init__(self, outputs: Callable[[int], str] | Sequence[str], name: str = "mock") -> None:
        self.name = name
        self._outputs = outputs
        self._i = 0

    def generate(self, images: Sequence[Image.Image]) -> list[GenOutput]:
        res = []
        for _ in images:
            o = (
                self._outputs(self._i)
                if callable(self._outputs)
                else self._outputs[self._i % len(self._outputs)]
            )
            res.append(GenOutput(o, 0.0))
            self._i += 1
        return res


def build_generator(
    spec: dict[str, Any], gen_cfg: dict[str, Any], cap: ImageCap, name: str
) -> Generator:
    backend = spec.get("backend", "hf")
    max_new = int(gen_cfg.get("max_new_tokens", 1024))
    if backend == "hf":
        return HFVisionGenerator(
            spec["model"],
            spec.get("adapter"),
            bool(spec.get("load_in_4bit", False)),
            spec.get("dtype", "auto"),
            max_new,
            cap,
            name,
        )
    if backend == "vllm":
        return VLLMVisionGenerator(
            spec["model"],
            max_new,
            cap,
            name,
            int(spec.get("max_model_len", 8192)),
            float(spec.get("gpu_memory_utilization", 0.85)),
            spec.get("quantization"),
        )
    if backend == "openai":
        return OpenAIVisionGenerator(spec["base_url"], spec["model"], max_new, name)
    if backend == "ocr_llm":
        llm_spec = spec.get("llm", {})
        llm: Callable[[list[dict[str, Any]]], tuple[str, int]]
        if llm_spec.get("backend", "hf") == "openai":
            client = OpenAIVisionGenerator(llm_spec["base_url"], llm_spec["model"], max_new)

            def llm(m: list[dict[str, Any]]) -> tuple[str, int]:
                o = client.chat(m)
                return o.raw, o.n_output_tokens or 0
        else:
            llm = HFTextLLM(
                llm_spec["model"],
                max_new,
                llm_spec.get("dtype", "auto"),
                bool(llm_spec.get("load_in_4bit", False)),
            )
        from receipt_vlm.baselines.ocr_llm import OCRLLMGenerator

        return OCRLLMGenerator(llm, spec.get("ocr_langs", "eng+ara+rus"), name)
    raise ValueError(f"unknown backend {backend!r}")
