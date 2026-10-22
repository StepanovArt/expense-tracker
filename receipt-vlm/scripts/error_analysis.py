#!/usr/bin/env python
"""RQ4 error analysis for one model: taxonomy table, error rates by language / source / image
sharpness, and 6 annotated examples. Writes the ``errors`` auto-section of RESULTS.md.

    uv run python scripts/error_analysis.py --model M2 --split test
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from receipt_vlm.eval.errors import CATEGORIES, ErrorEvent, classify
from receipt_vlm.eval.metrics import score_sample
from receipt_vlm.eval.report import replace_auto_section
from receipt_vlm.eval.runner import load_split, pred_path, select_subset
from receipt_vlm.utils.config import load_config
from receipt_vlm.utils.io import read_jsonl, resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger("error_analysis")


def sharpness(image_path: str) -> float:
    """Variance of the Laplacian on a 768-px grayscale copy (higher = sharper)."""
    from PIL import Image
    from scipy.ndimage import laplace

    im = Image.open(resolve(image_path)).convert("L")
    im.thumbnail((768, 768))
    return float(laplace(np.asarray(im, dtype=np.float32)).var())


def _md(hdr: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(
        ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def _short(v: Any, n: int = 38) -> str:
    s = "null" if v is None else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


OUTPUT_LEVEL = {
    "parse_fail",
    "truncated_output",
    "schema:date_format",
    "schema:number_as_str",
    "schema:other",
}


def plot_examples(examples: list[tuple[dict[str, Any], list[ErrorEvent], str]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(examples)
    if n == 0:
        return
    fig, axes = plt.subplots(
        n, 2, figsize=(11, 3.4 * n), gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False
    )
    for (sample, events, raw), (ax_im, ax_txt) in zip(examples, axes, strict=True):
        im = Image.open(resolve(sample["image"])).convert("RGB")
        im.thumbnail((500, 700))
        ax_im.imshow(im)
        ax_im.axis("off")
        ax_im.set_title(f"{sample['id']}  ({sample['language']})", fontsize=8)
        lines = []
        for e in events[:9]:
            if e.category in OUTPUT_LEVEL:  # show what the model actually wrote
                snippet = " ".join(raw.split())[:150]
                wrapped = "\n    ".join(snippet[i : i + 50] for i in range(0, len(snippet), 50))
                lines.append(f"{e.category} ({e.field})\n    output: {wrapped}")
                continue
            # matplotlib >= 3.11 shapes Arabic and applies bidi itself: pass logical text as is
            g, p = _short(e.gold), _short(e.pred)
            lrm = "\u200e"  # keeps "gold:" left-to-right when the value is Arabic
            lines.append(f"{e.category:<19} {e.field:<15}\n    gold:{lrm} {g}\n    pred:{lrm} {p}")
        if len(events) > 9:
            lines.append(f"... +{len(events) - 9} more")
        ax_txt.text(
            0, 1, "\n".join(lines), va="top", ha="left", family="DejaVu Sans Mono", fontsize=7.5
        )
        ax_txt.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyse(
    cfg: dict[str, Any],
    model: str,
    split: str,
    n_errors: int,
    n_examples: int,
    out_md: str,
    fig_dir: str,
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    samples = select_subset(
        load_split(cfg["data_dir"], split), cfg.get("subset", {}).get(split), seed
    )
    p = pred_path(cfg["preds_dir"], model, split, cfg.get("pred_tag", ""))
    preds = {r["id"]: r for r in read_jsonl(p)} if p.exists() else {}
    if any(s["id"] not in preds for s in samples):
        raise SystemExit(f"{model}: predictions for {split} are missing or incomplete ({p})")
    max_new = int(cfg["generation"]["max_new_tokens"])

    per_sample: list[tuple[dict[str, Any], list[ErrorEvent], float]] = []
    for s in samples:
        r = preds[s["id"]]
        ev = classify(s, r["raw"], r.get("n_output_tokens"), max_new)
        per_sample.append((s, ev, score_sample(s, r["raw"]).f1))

    wrong = [(s, ev) for s, ev, _ in per_sample if ev]
    rng = np.random.default_rng(seed)
    pick = (
        sorted(rng.choice(len(wrong), size=min(n_errors, len(wrong)), replace=False).tolist())
        if wrong
        else []
    )
    sampled = [wrong[i] for i in pick]

    langs = sorted({s["language"] for s, _ in wrong})
    rows = []
    for c in CATEGORIES:
        n_all = sum(any(e.category == c for e in ev) for _, ev in wrong)
        if not n_all:
            continue
        n_s = sum(any(e.category == c for e in ev) for _, ev in sampled)
        n_e = sum(e.category == c for _, ev in sampled for e in ev)
        # per-language counts over *all* errors, so rare language-specific failures are not
        # lost to the random 100-sample draw
        by_lang = [
            sum(any(e.category == c for e in ev) for s, ev in wrong if s["language"] == lg)
            for lg in langs
        ]
        rows.append([c, n_s, n_e, n_all, *by_lang])
    rows.sort(key=lambda r: (-int(r[1]), -int(r[3])))
    tax = _md(
        [
            "category",
            f"samples (of {len(sampled)} sampled)",
            "events (sampled)",
            f"samples (all {len(wrong)} wrong)",
            *[f"{lg} (all)" for lg in langs],
        ],
        rows,
    )

    field_counts = Counter(e.field for _, ev in sampled for e in ev)
    fields = _md(["field", "error events"], [[f, n] for f, n in field_counts.most_common(15)])

    # error rate over the whole split by language and source (not just the sample)
    rate_rows = []
    for dim in ("language", "source"):
        groups: dict[str, list[float]] = defaultdict(list)
        f1s: dict[str, list[float]] = defaultdict(list)
        for s, ev, f1 in per_sample:
            groups[s[dim]].append(float(bool(ev)))
            f1s[s[dim]].append(f1)
        for g in sorted(groups):
            n = len(groups[g])
            rate_rows.append(
                [
                    dim,
                    g + (" ⚠ low-n" if n < 50 else ""),
                    n,
                    f"{100 * np.mean(groups[g]):.1f}",
                    f"{100 * np.mean(f1s[g]):.1f}",
                ]
            )
    rates = _md(["by", "group", "n", "% samples with ≥1 error", "mean sample F1"], rate_rows)

    sharp = np.array([sharpness(s["image"]) for s, _, _ in per_sample])
    q = np.quantile(sharp, [1 / 3, 2 / 3]) if len(sharp) >= 3 else np.array([np.inf, np.inf])
    bucket = np.digitize(sharp, q)
    qrows = []
    for b, name in enumerate(("blurriest third", "middle third", "sharpest third")):
        idx = np.flatnonzero(bucket == b)
        if idx.size:
            err = np.mean([bool(per_sample[i][1]) for i in idx])
            f1 = np.mean([per_sample[i][2] for i in idx])
            qrows.append(
                [
                    name,
                    idx.size,
                    f"{np.median(sharp[idx]):.0f}",
                    f"{100 * err:.1f}",
                    f"{100 * f1:.1f}",
                ]
            )
    quality = _md(
        ["image sharpness (var. of Laplacian)", "n", "median", "% with error", "mean sample F1"],
        qrows,
    )

    # annotated examples: cover the most frequent categories first
    order = [r[0] for r in rows]
    chosen: list[tuple[dict[str, Any], list[ErrorEvent], str]] = []
    used: set[str] = set()
    for c in order:
        # prefer the random sample; fall back to all errors so rare categories get an example
        for s, ev in [*sampled, *wrong]:
            if s["id"] not in used and any(e.category == c for e in ev):
                chosen.append((s, ev, preds[s["id"]]["raw"]))
                used.add(s["id"])
                break
        if len(chosen) >= n_examples:
            break
    fig = resolve(fig_dir)
    fig.mkdir(parents=True, exist_ok=True)
    fig_name = f"errors_{model}_{split}.png"
    plot_examples(chosen, fig / fig_name)

    body = "\n".join(
        [
            f"_Auto-generated by `scripts/error_analysis.py --model {model} --split {split}`: "
            f"{len(sampled)} of {len(wrong)} erroneous predictions sampled with seed {seed}; categories are "
            "assigned automatically (`src/receipt_vlm/eval/errors.py`)._",
            "",
            f"### Error taxonomy ({model}, {split})",
            "",
            tax,
            "",
            "### Fields involved",
            "",
            fields,
            "",
            "### Error rate by language and source (all images of the split)",
            "",
            rates,
            "",
            "### Error rate by image sharpness (all images, terciles)",
            "",
            quality,
            "",
            f"![Annotated error examples](figures/{fig_name})",
        ]
    )
    replace_auto_section(
        resolve(out_md),
        f"errors:{model}:{split}",
        body,
        f"Error analysis: {model} on `{split}` (auto-generated)",
    )
    return {
        "n_wrong": len(wrong),
        "n_sampled": len(sampled),
        "categories": {r[0]: r[1] for r in rows},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval/final.yaml")
    ap.add_argument("--model", default="M2")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--out", default="reports/RESULTS.md")
    ap.add_argument("--fig-dir", default="reports/figures")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    info = analyse(cfg, args.model, args.split, args.n, args.examples, args.out, args.fig_dir)
    log.info("%s", info)


if __name__ == "__main__":
    main()
