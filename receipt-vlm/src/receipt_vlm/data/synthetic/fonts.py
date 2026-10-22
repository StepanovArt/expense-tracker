"""Fonts for synthetic receipts: OFL fonts pinned to a google/fonts commit, plus matplotlib's
bundled DejaVu as an always-available fallback. Every font is checked for glyph coverage of
the text it renders, so a missing glyph can never silently produce tofu boxes."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from receipt_vlm.utils.io import resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger(__name__)

GOOGLE_FONTS_COMMIT = "45e3d5269e4a17d823f6d84470bc084dc5ae46f6"
_BASE = f"https://raw.githubusercontent.com/google/fonts/{GOOGLE_FONTS_COMMIT}/"
# file name -> path in google/fonts
DOWNLOADS: dict[str, str] = {
    "Amiri-Regular.ttf": "ofl/amiri/Amiri-Regular.ttf",
    "Amiri-Bold.ttf": "ofl/amiri/Amiri-Bold.ttf",
    "NotoNaskhArabic.ttf": "ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf",
    "NotoSansArabic.ttf": "ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf",
    "IBMPlexSansArabic-Regular.ttf": "ofl/ibmplexsansarabic/IBMPlexSansArabic-Regular.ttf",
    "NotoSansMono.ttf": "ofl/notosansmono/NotoSansMono%5Bwdth%2Cwght%5D.ttf",
    "Cousine-Regular.ttf": "ofl/cousine/Cousine-Regular.ttf",
    "IBMPlexMono-Regular.ttf": "ofl/ibmplexmono/IBMPlexMono-Regular.ttf",
    "PTM55FT.ttf": "ofl/ptmono/PTM55FT.ttf",
    "VT323-Regular.ttf": "ofl/vt323/VT323-Regular.ttf",
}
# Which fonts each script may use. Arabic fonts must contain Arabic Presentation Forms-B,
# because we shape with arabic-reshaper and draw with Pillow's BASIC layout (no raqm).
SCRIPT_FONTS: dict[str, list[str]] = {
    "ar": [
        "Amiri-Regular.ttf",
        "NotoNaskhArabic.ttf",
        "NotoSansArabic.ttf",
        "IBMPlexSansArabic-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    "ru": [
        "Cousine-Regular.ttf",
        "IBMPlexMono-Regular.ttf",
        "NotoSansMono.ttf",
        "PTM55FT.ttf",
        "DejaVuSansMono.ttf",
    ],
    "en": [
        "Cousine-Regular.ttf",
        "IBMPlexMono-Regular.ttf",
        "NotoSansMono.ttf",
        "PTM55FT.ttf",
        "VT323-Regular.ttf",
        "DejaVuSansMono.ttf",
    ],
}
BOLD_OF = {
    "Amiri-Regular.ttf": "Amiri-Bold.ttf",
    "DejaVuSansMono.ttf": "DejaVuSansMono-Bold.ttf",
    "DejaVuSans.ttf": "DejaVuSans-Bold.ttf",
}


def font_dir() -> Path:
    return resolve(os.environ.get("RECEIPT_VLM_FONT_DIR", "data/cache/fonts"))


def _mpl_font(name: str) -> Path | None:
    import matplotlib

    p = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name
    return p if p.exists() else None


def ensure_fonts(download: bool = True) -> None:
    """Download the pinned OFL fonts once (skipped silently if offline; DejaVu remains)."""
    d = font_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, rel in DOWNLOADS.items():
        dest = d / name
        if dest.exists() or not download:
            continue
        try:
            urllib.request.urlretrieve(_BASE + rel, dest)
        except Exception as e:
            log.warning("font %s not downloaded (%s); falling back to DejaVu", name, e)
            dest.unlink(missing_ok=True)


def font_path(name: str) -> Path | None:
    p = font_dir() / name
    if p.exists():
        return p
    return _mpl_font(name)


@lru_cache(maxsize=64)
def _cmap(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont

    return frozenset(TTFont(path, fontNumber=0).getBestCmap())


def covers(path: Path, text: str) -> bool:
    cmap = _cmap(str(path))
    return all(ord(c) in cmap for c in text if not c.isspace())


@dataclass(frozen=True)
class FontChoice:
    regular: Path
    bold: Path

    def get(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return _load(str(self.bold if bold else self.regular), size)


@lru_cache(maxsize=256)
def _load(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)


def available(lang: str) -> list[str]:
    return [n for n in SCRIPT_FONTS[lang] if font_path(n) is not None]


def pick_font(name: str) -> FontChoice:
    regular = font_path(name)
    if regular is None:
        raise FileNotFoundError(name)
    bold = font_path(BOLD_OF.get(name, name)) or regular
    return FontChoice(regular, bold)
