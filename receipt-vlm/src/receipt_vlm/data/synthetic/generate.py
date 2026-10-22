"""Generate synthetic multilingual receipts with exact ground truth.

Sample ``i`` is fully determined by ``(seed, i)``: language and layout cycle through the
3 x 3 grid so every cell is balanced, and all randomness comes from a generator seeded with
``derive_seed(seed, "synthetic", i)``. Output: ``data/synthetic/images/syn_XXXXX.jpg`` and
``data/synthetic/samples.jsonl`` (read by the dataset builder).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

from PIL import Image

from receipt_vlm.data.record import LABEL_KEYS
from receipt_vlm.data.synthetic.degrade import degrade
from receipt_vlm.data.synthetic.fonts import ensure_fonts
from receipt_vlm.data.synthetic.render import render
from receipt_vlm.data.synthetic.templates import Lang, Layout, ReceiptSpec, make_spec
from receipt_vlm.schema import Receipt
from receipt_vlm.utils.io import resolve, write_jsonl
from receipt_vlm.utils.logging import get_logger
from receipt_vlm.utils.seed import rng

log = get_logger(__name__)

LANGS: tuple[Lang, ...] = ("en", "ar", "ru")
LAYOUTS: tuple[Layout, ...] = ("supermarket", "restaurant", "pharmacy")
MAX_SIDE = 1600


def is_consistent(r: Receipt, tol: float = 0.01) -> bool:
    """Σ item totals = subtotal and subtotal + tax = total, whenever the values are present."""
    item_sum = round(sum(i.total or 0.0 for i in r.items), 2)
    sub = r.subtotal if r.subtotal is not None else item_sum
    if r.subtotal is not None and r.items and abs(item_sum - r.subtotal) > tol:
        return False
    if r.total is not None and abs(sub + (r.tax or 0.0) - r.total) > tol:
        return False
    return all(
        i.quantity is None
        or i.unit_price is None
        or i.total is None
        or abs(round(i.quantity * i.unit_price, 2) - i.total) <= tol
        for i in r.items
    )


def spec_for(
    index: int, seed: int, langs: tuple[str, ...] = LANGS, layouts: tuple[str, ...] = LAYOUTS
) -> tuple[ReceiptSpec, Any]:
    g = rng(seed, "synthetic", str(index))
    lang = cast(Lang, langs[index % len(langs)])
    layout = cast(Layout, layouts[(index // len(langs)) % len(layouts)])
    return make_spec(g, lang, layout), g


def generate_one(args: tuple[int, int, str]) -> dict[str, Any]:
    index, seed, out_dir = args
    spec, g = spec_for(index, seed)
    gold = spec.gold()
    if not is_consistent(gold):
        raise AssertionError(f"inconsistent synthetic receipt {index}: {gold}")
    clean, rmeta = render(spec, g)
    img, dmeta = degrade(clean, g)
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    sid = f"syn_{index:05d}"
    path = resolve(out_dir) / "images" / f"{sid}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=int(dmeta["jpeg_quality"]))
    return {
        "source_id": sid,
        "image": str(path.relative_to(resolve("."))),
        "target": Receipt.model_validate(gold.model_dump()).model_dump(mode="json"),
        "labelled": list(LABEL_KEYS),  # the generator knows every field exactly
        "language": spec.lang,
        "meta": {"layout": spec.layout, "currency": spec.currency, **rmeta, "degrade": dmeta},
    }


def generate(n: int, seed: int, out_dir: str | Path, workers: int = 4) -> list[dict[str, Any]]:
    ensure_fonts()
    jobs = [(i, seed, str(out_dir)) for i in range(n)]
    if workers <= 1:
        rows = [generate_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(generate_one, jobs, chunksize=4))
    write_jsonl(resolve(out_dir) / "samples.jsonl", rows)
    log.info("generated %d synthetic receipts into %s", len(rows), out_dir)
    return rows


def sample_grid(
    rows: list[dict[str, Any]], out_path: str | Path, n: int = 20, seed: int = 0
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = rng(seed, "grid").choice(len(rows), size=min(n, len(rows)), replace=False)
    cols = 5
    nrows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 4.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), picks, strict=False):
        r = rows[int(i)]
        im = Image.open(resolve(r["image"]))
        im.thumbnail((480, 720))
        ax.imshow(im)
        ax.set_title(
            f"{r['source_id']} {r['language']}/{r['meta']['layout']}\n"
            f"total={r['target']['total']} {r['target']['currency']}",
            fontsize=8,
        )
    fig.tight_layout()
    out = resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
