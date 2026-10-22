"""Photo-like degradations: crumpled paper, uneven lighting, perspective, rotation, blur, JPEG.

Pure Pillow + numpy + scipy (no albumentations) so the generator runs anywhere. Every random
draw comes from the caller's generator, so a sample is reproducible from its seed.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """Low-frequency noise in [-1, 1] of shape (h, w)."""
    small = rng.standard_normal((cells, max(2, int(cells * w / max(h, 1)))))
    f = zoom(small, (h / small.shape[0], w / small.shape[1]), order=3)[:h, :w]
    f = np.pad(f, ((0, h - f.shape[0]), (0, w - f.shape[1])), mode="edge")
    out: np.ndarray = f / (np.abs(f).max() + 1e-6)
    return out


def crumple(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    """Crumpled paper: smooth displacement of the print + shading along random creases."""
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    amp = strength * float(rng.uniform(1.0, 4.0))
    dy = _smooth_field(rng, h, w, 6) * amp
    dx = _smooth_field(rng, h, w, 6) * amp
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        out[..., c] = map_coordinates(a[..., c], [yy + dy, xx + dx], order=1, mode="nearest")
    shade = _smooth_field(rng, h, w, 10)
    creases = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(0, 5))):  # straight folds
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        dist = np.abs((xx - x0) * np.sin(ang) - (yy - y0) * np.cos(ang))
        creases += np.exp(-dist / rng.uniform(1.5, 4.0)) * rng.uniform(0.04, 0.12)
    mult = 1.0 + 0.06 * strength * shade - creases * strength
    out *= mult[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def place_on_background(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Receipt lying on a table: textured background with margins."""
    w, h = img.size
    pad_x, pad_y = int(w * rng.uniform(0.05, 0.35)), int(h * rng.uniform(0.03, 0.15))
    # tables/desks: mostly greys and muted browns, never saturated colours
    gray = float(rng.uniform(35, 205))
    warm = float(rng.uniform(0, 30)) if rng.random() < 0.5 else 0.0  # wood-like tint
    base = np.array([gray + warm, gray + warm * 0.55, gray]) + rng.uniform(-10, 10, size=3)
    base = np.clip(base, 0, 255).astype(np.float32)
    bg = np.ones((h + 2 * pad_y, w + 2 * pad_x, 3), np.float32) * base
    tex = gaussian_filter(rng.standard_normal(bg.shape[:2]), sigma=float(rng.uniform(1, 6)))
    bg += (tex / (np.abs(tex).max() + 1e-6) * rng.uniform(5, 25))[..., None]
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    # soft shadow under the paper
    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(80, (pad_x + 6, pad_y + 8, pad_x + w + 6, pad_y + h + 8))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    canvas = Image.composite(Image.new("RGB", canvas.size, (0, 0, 0)), canvas, shadow)
    canvas.paste(img, (pad_x, pad_y))
    return canvas


def perspective(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    w, h = img.size
    m = strength * 0.08
    src: list[tuple[float, float]] = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    dst = [(x + rng.uniform(-m, m) * w, y + rng.uniform(-m, m) * h) for x, y in src]
    coeffs = _perspective_coeffs(dst, src)
    fill = tuple(int(v) for v in np.asarray(img)[0, 0])
    return img.transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC, fillcolor=fill
    )


def _perspective_coeffs(
    pa: list[tuple[float, float]], pb: list[tuple[float, float]]
) -> list[float]:
    rows = []
    for (x, y), (u, v) in zip(pa, pb, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(rows, dtype=np.float64)
    b = np.array(pb, dtype=np.float64).reshape(8)
    coeffs: list[float] = np.linalg.solve(a, b).tolist()
    return coeffs


def lighting(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    grad = 1.0 - strength * rng.uniform(0.15, 0.45) * r
    direction = rng.uniform(0, 2 * np.pi)
    lin = (
        (np.cos(direction) * xx / w + np.sin(direction) * yy / h)
        * strength
        * rng.uniform(-0.2, 0.2)
    )
    tint = 1.0 + rng.uniform(-0.03, 0.03) * np.array([1.0, 0.6, -0.4])  # warm/cool white balance
    out = a * (grad + lin)[..., None] * tint
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def degrade(
    img: Image.Image, rng: np.random.Generator, level: float | None = None
) -> tuple[Image.Image, dict[str, float]]:
    """Apply the full chain. ``level`` in [0, 1] scales all effects (random if None)."""
    lvl = float(rng.uniform(0.0, 1.0)) if level is None else level
    params: dict[str, float] = {"level": round(lvl, 3)}
    if rng.random() < 0.7:
        img = crumple(img, rng, lvl)
        params["crumple"] = 1
    img = place_on_background(img, rng)
    if rng.random() < 0.8:
        img = perspective(img, rng, lvl)
        params["perspective"] = 1
    angle = float(rng.uniform(-7, 7))
    fill = tuple(int(v) for v in np.asarray(img)[0, 0])
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    params["rotation"] = round(angle, 2)
    img = lighting(img, rng, lvl)
    blur = float(rng.uniform(0, 1.6) * lvl)
    if blur > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    params["blur"] = round(blur, 2)
    if lvl > 0.3:  # sensor noise
        a = np.asarray(img).astype(np.float32)
        a += rng.normal(0, 3 + 6 * lvl, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # the caller encodes with this quality when saving (one lossy step, not two)
    params["jpeg_quality"] = int(rng.integers(35, 92))
    return img, params
