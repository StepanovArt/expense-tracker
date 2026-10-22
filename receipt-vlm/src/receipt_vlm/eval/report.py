"""Metrics payloads, RESULTS.md tables and figures - all recomputed from saved predictions.

``reports/RESULTS.md`` is partly hand-written (research-question answers) and partly
generated: everything between ``<!-- BEGIN AUTO:<split> -->`` and ``<!-- END AUTO:<split> -->``
is replaced on every ``make report``; the rest is never touched.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from receipt_vlm.eval.bootstrap import CI, PairedResult, bootstrap_ci, paired_bootstrap
from receipt_vlm.eval.metrics import (
    COUNT_KEYS,
    HEADLINE,
    SampleScore,
    ScoreTable,
    field_support,
)
from receipt_vlm.eval.significance import McNemarResult, mcnemar

LOW_N = 50
COMPARE_METRICS: tuple[str, ...] = ("overall_f1", "total_em", "schema_valid_rate", "item_f1")
# figure colours (dataviz reference palette): emphasis = ours, recessive grey = baselines
OURS, BASE, INK, MUTED, GRID = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e1"


def _latency(scores: Sequence[SampleScore]) -> dict[str, float]:
    lat = np.array([s.latency_s for s in scores if s.latency_s is not None], dtype=float)
    if lat.size == 0:
        return {"p50": float("nan"), "p95": float("nan")}
    return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}


def _ci_dict(ci: CI) -> dict[str, float]:
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


def metrics_payload(
    scores: Sequence[SampleScore], table: ScoreTable, b: int, seed: int
) -> dict[str, Any]:
    support = field_support(table)
    fields = [k for k in COUNT_KEYS if support[k] > 0]
    names = (
        list(HEADLINE)
        + [f"f1.{k}" for k in fields]
        + [f"attr_acc.{a}" for a in ("name", "quantity", "unit_price", "total")]
    )
    cis = bootstrap_ci(table, names, b=b, seed=seed)
    status = {k: sum(s.status == k for s in scores) for k in ("ok", "schema_fail", "parse_fail")}
    null_rate: dict[str, float] = {}
    for key in {k for s in scores for k in s.pred_null}:
        vals = [s.pred_null[key] for s in scores if key in s.pred_null]
        null_rate[key] = float(np.mean(vals)) if vals else float("nan")
    breakdown: dict[str, dict[str, Any]] = {}
    for dim, values in (("language", table.languages), ("source", table.sources)):
        breakdown[dim] = {}
        for v in sorted(set(values)):
            sub = table.subset([x == v for x in values])
            c = bootstrap_ci(sub, ["overall_f1", "total_em", "schema_valid_rate"], b=b, seed=seed)
            breakdown[dim][v] = {
                "n": len(sub),
                "low_confidence": len(sub) < LOW_N,
                **{k: _ci_dict(x) for k, x in c.items()},
            }
    return {
        "n": len(table),
        "headline": {k: _ci_dict(cis[k]) for k in HEADLINE},
        "fields": {k: {"support": support[k], **_ci_dict(cis[f"f1.{k}"])} for k in fields},
        "attr_acc": {
            a: _ci_dict(cis[f"attr_acc.{a}"]) for a in ("name", "quantity", "unit_price", "total")
        },
        "status": status,
        "latency_s": _latency(scores),
        "pred_null_rate": null_rate,
        "breakdown": breakdown,
        "errors_top": _top_errors(scores),
    }


def _top_errors(scores: Sequence[SampleScore], k: int = 5) -> list[tuple[str, int]]:
    from collections import Counter

    c = Counter(re.sub(r"\d+", "N", s.error or "")[:80] for s in scores if s.error)
    return c.most_common(k)


# ----------------------------------------------------------------------------- report
@dataclass
class ModelRun:
    model_id: str
    description: str
    scores: list[SampleScore]
    table: ScoreTable
    payload: dict[str, Any]


def _pct(ci: Mapping[str, float], digits: int = 1) -> str:
    p = ci["point"]
    if p is None or np.isnan(p):
        return "–"
    return f"{100 * p:.{digits}f} [{100 * ci['lo']:.{digits}f}, {100 * ci['hi']:.{digits}f}]"


def headline_table(runs: Sequence[ModelRun], not_run: Mapping[str, str]) -> str:
    hdr = [
        "Model",
        "Setup",
        "n",
        "Overall F1",
        "Lenient F1",
        "JSON parse",
        "Schema-valid",
        "Total EM",
        "Item F1",
        "p50 / p95 s",
    ]
    rows = []
    for r in runs:
        h, lat = r.payload["headline"], r.payload["latency_s"]
        rows.append(
            [
                f"**{r.model_id}**",
                r.description,
                str(r.payload["n"]),
                _pct(h["overall_f1"]),
                _pct(h["lenient_f1"]),
                _pct(h["parse_rate"]),
                _pct(h["schema_valid_rate"]),
                _pct(h["total_em"]),
                _pct(h["item_f1"]),
                f"{lat['p50']:.2f} / {lat['p95']:.2f}" if not np.isnan(lat["p50"]) else "–",
            ]
        )
    for mid, why in not_run.items():
        rows.append([f"{mid}", why, "–", *["not run"] * 6, "–"])
    return _md(hdr, rows)


def field_table(runs: Sequence[ModelRun]) -> str:
    fields = sorted({f for r in runs for f in r.payload["fields"]}, key=COUNT_KEYS.index)
    hdr = ["Field", "support", *[r.model_id for r in runs]]
    rows = []
    for f in fields:
        sup = max(r.payload["fields"].get(f, {}).get("support", 0) for r in runs)
        rows.append(
            [
                f,
                str(sup),
                *[_pct(r.payload["fields"][f]) if f in r.payload["fields"] else "–" for r in runs],
            ]
        )
    return _md(hdr, rows)


def breakdown_table(runs: Sequence[ModelRun], dim: str) -> str:
    groups = sorted({g for r in runs for g in r.payload["breakdown"][dim]})
    hdr = [dim, "n", *[f"{r.model_id} F1" for r in runs]]
    rows = []
    for g in groups:
        n = max(r.payload["breakdown"][dim].get(g, {}).get("n", 0) for r in runs)
        flag = " ⚠ low-n" if 0 < n < LOW_N else ""
        cells = []
        for r in runs:
            b = r.payload["breakdown"][dim].get(g)
            cells.append(_pct(b["overall_f1"]) if b else "–")
        rows.append([f"{g}{flag}", str(n), *cells])
    return _md(hdr, rows)


@dataclass
class Comparison:
    a: str
    b: str
    paired: dict[str, PairedResult]
    mcnemar: McNemarResult


def compare(a: ModelRun, b: ModelRun, n_boot: int, seed: int) -> Comparison:
    paired = paired_bootstrap(a.table, b.table, COMPARE_METRICS, b=n_boot, seed=seed)
    bt = {s.id: s.total_correct for s in b.scores}
    mc = mcnemar([s.total_correct for s in a.scores], [bt[s.id] for s in a.scores])
    return Comparison(a.model_id, b.model_id, paired, mc)


def comparison_table(comps: Sequence[Comparison]) -> str:
    hdr = ["A vs B", "metric", "Δ (A−B), pts", "95% CI", "P(Δ ≤ 0)", "significant"]
    rows = []
    for c in comps:
        for m, r in c.paired.items():
            rows.append(
                [
                    f"{c.a} vs {c.b}",
                    m,
                    f"{100 * r.diff:+.1f}",
                    f"[{100 * r.lo:+.1f}, {100 * r.hi:+.1f}]",
                    f"{r.p_le_0:.3f}",
                    "yes" if r.significant else "no",
                ]
            )
        mc = c.mcnemar
        rows.append(
            [
                f"{c.a} vs {c.b}",
                "total EM, McNemar",
                f"{mc.a_only} vs {mc.b_only} discordant",
                f"stat {mc.statistic:.2f}",
                f"p = {mc.p_value:.4f} ({'exact' if mc.exact else 'χ²'})",
                "yes" if mc.p_value < 0.05 else "no",
            ]
        )
    return _md(hdr, rows)


def _md(hdr: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- figures
def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_overall(
    runs: Sequence[ModelRun], ours: set[str], path: Path, metric: str = "overall_f1"
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 * len(runs) + 2, 3.6))
    x = np.arange(len(runs))
    pts = [100 * r.payload["headline"][metric]["point"] for r in runs]
    lo = [100 * r.payload["headline"][metric]["lo"] for r in runs]
    hi = [100 * r.payload["headline"][metric]["hi"] for r in runs]
    colors = [OURS if r.model_id in ours else BASE for r in runs]
    ax.bar(x, pts, width=0.55, color=colors, edgecolor="white", linewidth=2)
    ax.errorbar(
        x,
        pts,
        yerr=[np.subtract(pts, lo), np.subtract(hi, pts)],
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=4,
    )
    for xi, p, h in zip(x.tolist(), pts, hi, strict=True):
        # label above the CI cap so it never collides with the error bar
        ax.text(float(xi), h + 1.5, f"{p:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [r.model_id for r in runs])
    label = metric.replace("_", " ").replace("f1", "F1").replace("em", "EM")
    ax.set_ylabel(f"{label[0].upper() + label[1:]} (%), 95% CI", color=MUTED)
    ax.set_ylim(0, 110)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_field_heatmap(runs: Sequence[ModelRun], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fields = sorted({f for r in runs for f in r.payload["fields"]}, key=COUNT_KEYS.index)
    m = np.array(
        [[100 * r.payload["fields"].get(f, {}).get("point", np.nan) for f in fields] for r in runs]
    )
    blues = LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    fig, ax = plt.subplots(figsize=(0.75 * len(fields) + 2, 0.5 * len(runs) + 1.6))
    im = ax.imshow(m, cmap=blues, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isnan(m[i, j]):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if m[i, j] > 55 else INK,
                )
    ax.set_xticks(range(len(fields)), fields, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(runs)), [r.model_id for r in runs], fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("field F1 (%)", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency_vs_f1(runs: Sequence[ModelRun], ours: set[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(r, r.payload["latency_s"]["p50"], r.payload["headline"]["overall_f1"]) for r in runs]
    pts = [p for p in pts if not np.isnan(p[1])]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r, lat, f1 in pts:
        c = OURS if r.model_id in ours else BASE
        ax.errorbar(
            lat,
            100 * f1["point"],
            yerr=[[100 * (f1["point"] - f1["lo"])], [100 * (f1["hi"] - f1["point"])]],
            fmt="o",
            ms=8,
            color=c,
            ecolor=c,
            elinewidth=1.2,
            capsize=3,
            mec="white",
            mew=2,
        )
    # direct labels with greedy collision avoidance in display space
    fig.canvas.draw()
    placed: list[tuple[float, float]] = []
    for r, lat, f1 in sorted(pts, key=lambda t: -float(t[2]["point"])):
        px, py = ax.transData.transform((lat, 100 * f1["point"]))
        dy = 2.0
        for _ in range(6):
            if all(abs(px - qx) > 40 or abs(py + dy - qy) > 14 for qx, qy in placed):
                break
            dy -= 14
        placed.append((px, py + dy))
        ax.annotate(
            r.model_id,
            (lat, 100 * f1["point"]),
            xytext=(9, dy),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            va="bottom",
        )
    ax.set_xlabel("p50 latency per image (s)", color=MUTED)
    ax.set_ylabel("Overall F1 (%), 95% CI", color=MUTED)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def replace_auto_section(md_path: Path, split: str, body: str, title: str) -> None:
    begin, end = f"<!-- BEGIN AUTO:{split} -->", f"<!-- END AUTO:{split} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Results\n"
    if begin in text and end in text:
        text = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + f"\n\n## {title}\n\n{block}\n"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
