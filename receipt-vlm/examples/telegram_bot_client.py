#!/usr/bin/env python
"""Telegram bot that sends receipt photos to the receipt-vlm API and replies with a summary.

This is the bridge to the expense bot in the parent repository: the same Telegram account can
log a whole receipt instead of typing "productos 80 y taxi 25".

    pip install "python-telegram-bot>=21" httpx
    TELEGRAM_BOT_TOKEN=... EXTRACT_API_URL=http://localhost:8000/extract python examples/telegram_bot_client.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.environ.get("EXTRACT_API_URL", "http://localhost:8000/extract")


def _money(x: float | None, cur: str) -> str:
    return "–" if x is None else f"{x:,.2f} {cur}".strip()


def format_summary(body: dict[str, Any]) -> str:
    """API response (200) -> short plain-text message."""
    r, c = body["receipt"], body.get("consistency", {})
    cur = r.get("currency") or ""
    lines = [f"🧾 {r.get('merchant') or 'Unknown merchant'}"]
    when = " ".join(x for x in (r.get("date"), r.get("time")) if x)
    if when:
        lines.append(f"📅 {when}")
    for it in r.get("items", [])[:15]:
        qty = f"{it['quantity']:g} × " if it.get("quantity") not in (None, 1) else ""
        lines.append(f"• {qty}{it['name']}: {_money(it.get('total'), cur)}")
    if len(r.get("items", [])) > 15:
        lines.append(f"… and {len(r['items']) - 15} more items")
    if r.get("tax") is not None:
        lines.append(f"Tax: {_money(r['tax'], cur)}")
    lines.append(f"💰 Total: {_money(r.get('total'), cur)}")
    if c.get("all_consistent") is False:
        lines.append("⚠️ The numbers do not add up — please check the receipt.")
    return "\n".join(lines)


def extract(image_bytes: bytes, url: str = API_URL) -> tuple[int, dict[str, Any]]:
    r = httpx.post(url, files={"image": ("receipt.jpg", image_bytes, "image/jpeg")}, timeout=300)
    return r.status_code, r.json()


def main() -> None:  # pragma: no cover - needs a Telegram token
    from telegram import Update
    from telegram.ext import Application, ContextTypes, MessageHandler, filters

    async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        photo = update.message.photo[-1]  # largest size
        data = bytes(await (await photo.get_file()).download_as_bytearray())
        status, body = extract(data)
        if status == 200:
            await update.message.reply_text(format_summary(body))
        else:
            await update.message.reply_text(
                f"Could not read this receipt ({status}). Try a sharper, flatter photo."
            )

    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.run_polling()


if __name__ == "__main__":
    main()
