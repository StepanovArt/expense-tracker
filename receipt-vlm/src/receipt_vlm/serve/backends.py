"""Model backends for the API: one chat turn = image + extra messages -> raw text.

``openai``  any OpenAI-compatible server (vLLM ``vllm serve``, llama.cpp ``llama-server``)
``hf``      in-process transformers model (CPU or GPU; slow, but needs no extra server)
``mock``    scripted outputs (tests)
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Protocol

from PIL import Image

from receipt_vlm.prompts import vlm_messages


class Backend(Protocol):
    name: str

    def chat(self, image: Image.Image, history: Sequence[dict[str, Any]]) -> tuple[str, float]:
        """``history``: extra turns after the first user message (for the retry)."""
        ...


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        transport: Any = None,
    ) -> None:
        import httpx

        self.name = f"openai:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def chat(self, image: Image.Image, history: Sequence[dict[str, Any]]) -> tuple[str, float]:
        from receipt_vlm.models.inference import _openai_messages

        messages = _openai_messages(image) + [
            {
                "role": h["role"],
                "content": h["content"]
                if isinstance(h["content"], str)
                else h["content"][0]["text"],
            }
            for h in history
        ]
        t0 = time.perf_counter()
        r = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or "", time.perf_counter() - t0

    def served_model(self) -> str:
        r = self.client.get("/models")
        r.raise_for_status()
        return str(r.json()["data"][0]["id"])


class HFBackend:
    def __init__(
        self,
        model: str,
        adapter: str | None = None,
        max_new_tokens: int = 1024,
        max_pixels: int = 1048576,
        dtype: str = "auto",
    ) -> None:
        from receipt_vlm.models.load import ImageCap, load_processor, load_vlm

        self.name = f"hf:{model}" + (f"+{adapter}" if adapter else "")
        self.processor = load_processor(
            adapter or model, ImageCap(min_pixels=min(65536, max_pixels), max_pixels=max_pixels)
        )
        self.model = load_vlm(model, adapter, False, dtype)
        self.max_new_tokens = max_new_tokens

    def chat(self, image: Image.Image, history: Sequence[dict[str, Any]]) -> tuple[str, float]:
        import torch

        msgs = vlm_messages(image) + [
            {
                "role": h["role"],
                "content": [{"type": "text", "text": h["content"]}]
                if isinstance(h["content"], str)
                else h["content"],
            }
            for h in history
        ]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return str(text), time.perf_counter() - t0


class MockBackend:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.name = "mock"
        self.outputs = list(outputs)
        self.calls: list[Sequence[dict[str, Any]]] = []

    def chat(self, image: Image.Image, history: Sequence[dict[str, Any]]) -> tuple[str, float]:
        self.calls.append(list(history))
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)], 0.0


def backend_from_env() -> Backend:
    kind = os.environ.get("RECEIPT_VLM_BACKEND", "openai")
    max_tokens = int(os.environ.get("RECEIPT_VLM_MAX_NEW_TOKENS", "1024"))
    if kind == "openai":
        return OpenAIBackend(
            os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1"),
            os.environ.get("RECEIPT_VLM_MODEL", "receipt-vlm"),
            max_tokens,
        )
    if kind == "hf":
        return HFBackend(
            os.environ.get("RECEIPT_VLM_MODEL", "outputs/merged"),
            os.environ.get("RECEIPT_VLM_ADAPTER") or None,
            max_tokens,
            int(os.environ.get("RECEIPT_VLM_MAX_PIXELS", "1048576")),
            os.environ.get("RECEIPT_VLM_DTYPE", "auto"),
        )
    raise ValueError(f"unknown RECEIPT_VLM_BACKEND={kind!r} (openai|hf)")
