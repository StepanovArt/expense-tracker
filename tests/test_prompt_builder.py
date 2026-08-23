"""Tests para src/llm/prompt_builder.py"""
import json
import re
from src.llm.prompt_builder import build_prompt

CATEGORIES = ["Supermercado", "Salidas", "Delivery"]


def test_prompt_contains_user_message():
    prompt = build_prompt("gasté 200 en super", CATEGORIES)
    assert "gasté 200 en super" in prompt


def test_prompt_contains_categories():
    prompt = build_prompt("taxi 50", CATEGORIES)
    for cat in CATEGORIES:
        assert cat in prompt


def test_prompt_contains_today():
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = build_prompt("cafe 80", CATEGORIES)
    assert today in prompt


def test_prompt_requests_array():
    prompt = build_prompt("cafe 80", CATEGORIES)
    # El prompt debe mencionar array JSON, no objeto individual
    assert "[{" in prompt or "array" in prompt.lower()


def test_prompt_mentions_multiple_expenses():
    prompt = build_prompt("cafe 80", CATEGORIES)
    assert "múltiples" in prompt or "multiple" in prompt.lower()
