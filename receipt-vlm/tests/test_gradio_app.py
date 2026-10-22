"""Gradio demo: rendering logic + the app actually serves (mocked model)."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
VALID = {
    "merchant": "كارفور",
    "date": "2024-03-05",
    "time": "10:00",
    "currency": "AED",
    "items": [{"name": "حليب", "quantity": 2, "unit_price": 3.5, "total": 7.0}],
    "subtotal": 7.0,
    "tax": 0.35,
    "total": 7.35,
    "payment_method": "card",
    "language": "ar",
}


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("gradio_app", ROOT / "app" / "gradio_app.py")
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def test_render_success_and_failure() -> None:
    m = _module()
    from receipt_vlm.serve.backends import MockBackend

    status, body = m.LocalClient(MockBackend([json.dumps(VALID)]))(
        Image.new("RGB", (40, 60), "white")
    )
    receipt, rows, md = m.render(status, body)
    assert status == 200 and receipt["total"] == 7.35
    assert rows == [["حليب", 2.0, 3.5, 7.0]]
    assert "كارفور" in md and "✅ yes" in md
    status, body = m.LocalClient(MockBackend(["nope"]))(Image.new("RGB", (40, 60), "white"))
    _, rows, md = m.render(status, body)
    assert status == 422 and rows == [] and "nope" in md


@pytest.mark.slow
def test_demo_builds_and_serves(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import httpx

    m = _module()
    monkeypatch.setattr(m, "CACHE", str(tmp_path / "examples"))
    from receipt_vlm.serve.backends import MockBackend

    demo = m.build_demo(m.LocalClient(MockBackend([json.dumps(VALID)])))
    assert len(m.example_images()) == 6
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    _, url, _ = demo.launch(prevent_thread_lock=True, server_port=port, server_name="127.0.0.1")
    try:
        r = httpx.get(url, timeout=30)
        assert r.status_code == 200 and "receipt-vlm" in r.text
    finally:
        demo.close()
