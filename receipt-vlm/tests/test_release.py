"""Model card rendering and the Telegram client formatter (no network)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _load(rel: str, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def test_model_card_without_metrics_says_not_evaluated() -> None:
    m = _load("scripts/push_to_hub.py", "push")
    card = m.render_model_card(
        "me/receipts",
        None,
        {"counts": {"train": 1, "val": 2, "test": 3}, "gaps": {"cord": "x"}},
        "Qwen/Qwen3-VL-2B-Instruct",
    )
    assert card.startswith("---\nlicense: apache-2.0") and "Not evaluated yet" in card
    assert "1 / 2 / 3" in card and "missing from this build: cord" in card


def test_model_card_with_metrics_shows_cis() -> None:
    m = _load("scripts/push_to_hub.py", "push")
    ci = {"point": 0.812, "lo": 0.79, "hi": 0.83, "n": 300}
    metrics = {
        "n": 300,
        "headline": {"overall_f1": ci, "total_em": ci},
        "breakdown": {"language": {"ar": {"n": 40, "overall_f1": ci, "low_confidence": True}}},
        "latency_s": {"p50": 0.5, "p95": 0.9},
    }
    card = m.render_model_card("me/receipts", metrics, None, "base")
    assert (
        "81.2 [79.0, 83.0]" in card
        and "| ar | 40 |" in card
        and "low-n" in card
        and "Not evaluated" not in card
    )


def test_telegram_summary() -> None:
    m = _load("examples/telegram_bot_client.py", "tg")
    body = {
        "receipt": {
            "merchant": "Магазин «Берёзка»",
            "date": "2024-03-05",
            "time": None,
            "currency": "RUB",
            "items": [
                {"name": "Молоко", "quantity": 2, "total": 179.8},
                {"name": "Хлеб", "quantity": 1, "total": 45},
            ],
            "tax": None,
            "total": 224.8,
        },
        "consistency": {"all_consistent": False},
    }
    text = m.format_summary(body)
    assert (
        "Магазин «Берёзка»" in text
        and "• 2 × Молоко: 179.80 RUB" in text
        and "• Хлеб: 45.00 RUB" in text
    )
    assert "Total: 224.80 RUB" in text and "do not add up" in text
