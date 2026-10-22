#!/usr/bin/env python
"""Gradio demo: receipt photo -> validated JSON, items table and arithmetic check.

Backends (same as the API, see ``serve/backends.py``):
  * ``--api URL``            call a running ``/extract`` service (default if EXTRACT_API_URL set)
  * ``RECEIPT_VLM_BACKEND``  ``openai`` (vLLM) or ``hf`` (in-process; what an HF Space uses)

The six examples are synthetic receipts from the frozen synthetic *test* subset, re-rendered
deterministically from their seed at start-up (licence-safe, nothing binary in git).
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Any

from PIL import Image

from receipt_vlm.utils.io import read_jsonl, resolve

EXAMPLE_SEED = 1234  # = configs/data.yaml seed used to generate the synthetic set
CACHE = "data/cache/demo_examples"


def example_images(n: int = 6) -> list[str]:
    """Two synthetic test receipts per language, regenerated from their generator seed."""
    from receipt_vlm.data.synthetic.generate import LANGS, generate_one

    manifest = resolve("data/splits/test_manifest.jsonl")
    ids = (
        [r["id"] for r in read_jsonl(manifest) if r["source"] == "synthetic"]
        if manifest.exists()
        else []
    )
    idx = sorted(int(i.rsplit("_", 1)[1]) for i in ids) or list(range(30))
    picks: list[int] = []
    for k in range(len(LANGS)):
        picks += [i for i in idx if i % len(LANGS) == k][: n // len(LANGS)]
    out = []
    for i in picks[:n]:
        path = resolve(CACHE) / "images" / f"syn_{i:05d}.jpg"
        if not path.exists():
            generate_one((i, EXAMPLE_SEED, CACHE))
        out.append(str(path))
    return out


class APIClient:
    def __init__(self, url: str) -> None:
        import httpx

        self.url = url
        self.client = httpx.Client(timeout=300)

    def __call__(self, image: Image.Image) -> tuple[int, dict[str, Any]]:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=95)
        r = self.client.post(
            self.url, files={"image": ("receipt.jpg", buf.getvalue(), "image/jpeg")}
        )
        return r.status_code, r.json()


class LocalClient:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(self, image: Image.Image) -> tuple[int, dict[str, Any]]:
        from receipt_vlm.serve.extract import extract_receipt

        res = extract_receipt(self.backend, image.convert("RGB"))
        if res.ok:
            return 200, {
                "receipt": res.receipt,
                "consistency": res.consistency,
                "latency_s": round(res.latency_s, 2),
                "attempts": res.attempts,
            }
        return 422, {
            "error": "model output is not valid receipt JSON",
            "raw": res.raw,
            "detail": res.error,
            "attempts": res.attempts,
            "latency_s": round(res.latency_s, 2),
        }


def _check(flag: bool | None) -> str:
    return "n/a" if flag is None else ("✅ yes" if flag else "❌ no")


def render(status: int, body: dict[str, Any]) -> tuple[dict[str, Any], list[list[Any]], str]:
    if status != 200:
        md = (
            f"### ⚠️ Could not extract a valid receipt (HTTP {status})\n\n"
            f"{body.get('error', '')}  \n`{body.get('detail') or ''}`\n\n**Raw model output:**\n\n"
            f"```\n{str(body.get('raw', ''))[:2000]}\n```"
        )
        return body, [], md
    r, c = body["receipt"], body["consistency"]
    rows = [[it["name"], it["quantity"], it["unit_price"], it["total"]] for it in r["items"]]
    cur = r.get("currency") or ""
    md = "\n".join(
        [
            f"### {r.get('merchant') or 'Unknown merchant'}",
            f"**Date** {r.get('date') or '–'} {r.get('time') or ''} · **Language** {r.get('language') or '–'} · "
            f"**Payment** {r.get('payment_method') or '–'}",
            "",
            "| | |",
            "|---|---|",
            f"| Subtotal | {r.get('subtotal') if r.get('subtotal') is not None else '–'} {cur} |",
            f"| Tax | {r.get('tax') if r.get('tax') is not None else '–'} {cur} |",
            f"| **Total** | **{r.get('total') if r.get('total') is not None else '–'} {cur}** |",
            f"| Σ items | {c.get('items_sum') if c.get('items_sum') is not None else '–'} |",
            f"| Σ items = subtotal | {_check(c.get('items_match_subtotal'))} |",
            f"| subtotal + tax = total | {_check(c.get('all_consistent'))} |",
            "",
            f"_{body.get('latency_s', '?')} s, {body.get('attempts', 1)} attempt(s)_",
        ]
    )
    return r, rows, md


def build_demo(client: Any) -> Any:
    import gradio as gr

    def predict(image: Image.Image | None) -> tuple[Any, Any, str]:
        if image is None:
            return {}, [], "Upload a receipt photo."
        return render(*client(image))

    with gr.Blocks(title="receipt-vlm") as demo:
        gr.Markdown(
            "# 🧾 receipt-vlm\nReceipt photo → strict JSON (Qwen3-VL-2B + LoRA). "
            "English, Arabic, Russian. Output is validated against the schema; one retry on invalid JSON."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Receipt photo")
                btn = gr.Button("Extract", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown()
                table = gr.Dataframe(
                    headers=["name", "quantity", "unit_price", "total"], label="Items", wrap=True
                )
                js = gr.JSON(label="Receipt JSON")
        btn.click(predict, inputs=img, outputs=[js, table, summary])
        examples = example_images()
        if examples:
            gr.Examples(
                examples=examples, inputs=img, label="Synthetic test receipts (EN / AR / RU)"
            )
    return demo


def make_client(api: str | None) -> Any:
    url = api or os.environ.get("EXTRACT_API_URL")
    if url:
        return APIClient(url)
    from receipt_vlm.serve.backends import backend_from_env

    return LocalClient(backend_from_env())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default=None, help="URL of a running /extract endpoint")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_demo(make_client(args.api)).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    if not Path.cwd().joinpath("pyproject.toml").exists():
        os.chdir(resolve("."))
    main()
