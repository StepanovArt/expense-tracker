"""FastAPI service: ``POST /extract`` (multipart ``image``) -> validated Receipt JSON.

* the model output is parsed and validated with the same code as the evaluation;
* if it is not valid, the request is retried **once** with the model's answer and a
  "return valid JSON only" reminder appended to the conversation;
* if it is still invalid the API answers **422** with the raw text and the parse error.

Run: ``uvicorn receipt_vlm.serve.api:app`` with ``RECEIPT_VLM_BACKEND=openai`` (default; vLLM
at ``VLLM_BASE_URL``) or ``RECEIPT_VLM_BACKEND=hf`` (in-process model ``RECEIPT_VLM_MODEL``).
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from receipt_vlm.prompts import PROMPT_VERSION
from receipt_vlm.serve.backends import Backend, backend_from_env
from receipt_vlm.serve.extract import MAX_ATTEMPTS, extract_receipt
from receipt_vlm.utils.logging import get_logger

log = get_logger("receipt_vlm.api")

MAX_BYTES = 15 * 1024 * 1024


def create_app(backend: Backend | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "backend", None) is None:
            app.state.backend = backend_from_env()
        log.info("backend: %s", app.state.backend.name)
        yield

    app = FastAPI(title="receipt-vlm", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend

    @app.get("/health")
    def health() -> dict[str, Any]:
        b = app.state.backend
        return {
            "status": "ok",
            "backend": None if b is None else b.name,
            "prompt_version": PROMPT_VERSION,
        }

    @app.post("/extract")
    async def extract(image: UploadFile = File(...)) -> Any:  # noqa: B008 - FastAPI idiom
        data = await image.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(
                status_code=413, detail=f"image larger than {MAX_BYTES // 2**20} MB"
            )
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"not a readable image: {e}") from e
        b: Backend = app.state.backend
        try:
            res = await run_in_threadpool(extract_receipt, b, img)
        except Exception as e:  # backend down / timeout
            log.exception("backend failure")
            raise HTTPException(status_code=502, detail=f"model backend error: {e}") from e
        if res.ok:
            return {
                "receipt": res.receipt,
                "consistency": res.consistency,
                "latency_s": round(res.latency_s, 3),
                "attempts": res.attempts,
                "model": b.name,
                "prompt_version": PROMPT_VERSION,
            }
        return JSONResponse(
            status_code=422,
            content={
                "error": "model output is not valid receipt JSON",
                "status": res.status,
                "detail": res.error,
                "raw": res.raw,
                "attempts": MAX_ATTEMPTS,
                "latency_s": round(res.latency_s, 3),
            },
        )

    return app


app = create_app()
