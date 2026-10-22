"""FastAPI /extract with mocked model backends (no GPU, no network)."""

import io
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from receipt_vlm.prompts import RETRY_REMINDER
from receipt_vlm.serve.api import create_app
from receipt_vlm.serve.backends import MockBackend, OpenAIBackend

VALID = json.dumps(
    {
        "merchant": "FRESHMART",
        "date": "2024-03-05",
        "time": None,
        "currency": "USD",
        "items": [{"name": "Milk", "quantity": 2, "unit_price": 1.5, "total": 3.0}],
        "subtotal": 3.0,
        "tax": 0.24,
        "total": 3.24,
        "payment_method": "card",
        "language": "en",
    }
)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 48, 3), 200, dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _post(
    client: TestClient, data: bytes | None = None, ctype: str = "image/jpeg"
) -> httpx.Response:
    return client.post(
        "/extract", files={"image": ("r.jpg", data if data is not None else _jpeg(), ctype)}
    )


def test_valid_first_attempt() -> None:
    backend = MockBackend([f"```json\n{VALID}\n```"])
    with TestClient(create_app(backend)) as c:
        r = _post(c)
    assert r.status_code == 200
    body = r.json()
    assert body["receipt"]["total"] == 3.24 and body["attempts"] == 1
    assert body["consistency"] == {
        "items_sum": 3.0,
        "items_match_subtotal": True,
        "all_consistent": True,
    }
    assert backend.calls == [[]]


def test_retry_once_with_reminder_then_success() -> None:
    backend = MockBackend(['{"total": "3.24"}', VALID])
    with TestClient(create_app(backend)) as c:
        r = _post(c)
    assert r.status_code == 200 and r.json()["attempts"] == 2
    retry_history = backend.calls[1]
    assert retry_history[0] == {"role": "assistant", "content": '{"total": "3.24"}'}
    assert retry_history[1] == {"role": "user", "content": RETRY_REMINDER}


def test_still_invalid_returns_422_with_raw_text() -> None:
    backend = MockBackend(["sorry", "still not json"])
    with TestClient(create_app(backend)) as c:
        r = _post(c)
    assert r.status_code == 422
    body = r.json()
    assert (
        body["raw"] == "still not json" and body["status"] == "parse_fail" and body["attempts"] == 2
    )
    assert len(backend.calls) == 2


def test_bad_uploads() -> None:
    with TestClient(create_app(MockBackend([VALID]))) as c:
        assert _post(c, b"not an image", "text/plain").status_code == 400
        assert _post(c, b"0" * (16 * 1024 * 1024)).status_code == 413
        assert c.get("/health").json()["status"] == "ok"


def test_backend_failure_is_502() -> None:
    class Down(MockBackend):
        def chat(self, image, history):  # type: ignore[no-untyped-def]
            raise ConnectionError("vLLM not reachable")

    with TestClient(create_app(Down([]))) as c:
        assert _post(c).status_code == 502


def test_openai_backend_request_format() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID}}]})

    be = OpenAIBackend(
        "http://vllm:8000/v1", "receipt-vlm", max_tokens=99, transport=httpx.MockTransport(handler)
    )
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    text, _ = be.chat(
        img, [{"role": "assistant", "content": "x"}, {"role": "user", "content": RETRY_REMINDER}]
    )
    assert text == VALID
    req = seen[0]
    assert req["temperature"] == 0.0 and req["max_tokens"] == 99 and req["model"] == "receipt-vlm"
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert req["messages"][1]["content"][0]["type"] == "image_url"
    assert req["messages"][1]["content"][0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


@pytest.mark.slow
def test_hf_backend_with_tiny_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from receipt_vlm.models.tiny import build_tiny_qwen3_vl
    from receipt_vlm.serve.backends import HFBackend

    be = HFBackend(
        str(build_tiny_qwen3_vl(tmp_path / "tiny")),
        max_new_tokens=8,
        max_pixels=65536,
        dtype="float32",
    )
    with TestClient(create_app(be)) as c:
        r = _post(c)
    assert (
        r.status_code == 422 and r.json()["attempts"] == 2
    )  # random weights -> invalid JSON, retried
