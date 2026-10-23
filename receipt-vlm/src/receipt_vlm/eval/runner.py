"""Run models over a split, save raw predictions, score them.

Predictions are always saved to ``outputs/preds/{model_id}/{split}.jsonl`` (raw text, latency,
generation settings) so metrics can be recomputed without re-running models; runs resume from
an existing file. Scoring never re-runs models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from receipt_vlm.eval.metrics import SampleScore, ScoreTable, score_sample
from receipt_vlm.models.inference import Generator, build_generator, load_image
from receipt_vlm.models.load import ImageCap
from receipt_vlm.prompts import PROMPT_VERSION
from receipt_vlm.utils.io import append_jsonl, read_jsonl, resolve
from receipt_vlm.utils.logging import get_logger

log = get_logger(__name__)


def load_split(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    path = resolve(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `make data` (and `make synth`) first")
    return read_jsonl(path)


def select_subset(
    samples: Sequence[dict[str, Any]], n: int | None, seed: int, stratify: str | None = "source"
) -> list[dict[str, Any]]:
    """Deterministic subset, fixed for a given seed and independent of file order.

    With ``stratify="source"`` the subset interleaves sources round-robin (each in seeded hash
    order), so a checkpoint-selection subset is not dominated by the largest source.
    """
    if not n or n >= len(samples):
        return list(samples)

    def key(s: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{s['id']}".encode()).hexdigest()

    if not stratify:
        return sorted(samples, key=key)[:n]
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        groups.setdefault(str(s.get(stratify, "")), []).append(s)
    queues = [sorted(g, key=key) for _, g in sorted(groups.items())]
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n:
        for q in queues:
            if i < len(q) and len(out) < n:
                out.append(q[i])
        i += 1
    return out


def pred_path(preds_dir: str | Path, model_id: str, split: str, tag: str = "") -> Path:
    return resolve(preds_dir) / model_id / f"{split}{tag}.jsonl"


def run_predictions(
    gen: Generator,
    samples: Sequence[dict[str, Any]],
    out_path: Path,
    batch_size: int = 1,
    meta: dict[str, Any] | None = None,
) -> Path:
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [s for s in samples if s["id"] not in done]
    if done:
        log.info("%s: resuming, %d done, %d to go", out_path, len(done), len(todo))
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        outs = gen.generate([load_image(s["image"]) for s in batch])
        for s, o in zip(batch, outs, strict=True):
            row: dict[str, Any] = {
                "id": s["id"],
                "raw": o.raw,
                "latency_s": round(o.latency_s, 4),
                "n_output_tokens": o.n_output_tokens,
                **({"extra": o.extra} if o.extra else {}),
                **(meta or {}),
            }
            append_jsonl(out_path, row)
        if (i // batch_size) % 20 == 0:
            log.info("%s: %d/%d", gen.name, min(i + batch_size, len(todo)), len(todo))
    return out_path


def score_predictions(
    samples: Sequence[dict[str, Any]], preds: Iterable[dict[str, Any]]
) -> list[SampleScore]:
    by_id = {p["id"]: p for p in preds}
    missing = [s["id"] for s in samples if s["id"] not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} samples have no prediction (e.g. {missing[:3]}); refusing to score a partial run"
        )
    return [
        score_sample(s, by_id[s["id"]]["raw"], by_id[s["id"]].get("latency_s")) for s in samples
    ]


def device_name() -> str:
    """The accelerator predictions run on; latency is only comparable on the same one."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"


def evaluate_model(
    model_id: str,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    split: str,
    generator: Generator | None = None,
) -> tuple[list[SampleScore], ScoreTable]:
    samples = select_subset(
        load_split(cfg["data_dir"], split), cfg.get("subset", {}).get(split), int(cfg["seed"])
    )
    cap = ImageCap(**cfg.get("image", {}))
    gen_cfg = cfg.get("generation", {})
    tag = cfg.get("pred_tag", "")
    out = pred_path(cfg["preds_dir"], model_id, split, tag)
    meta = {
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "max_new_tokens": gen_cfg.get("max_new_tokens"),
        "max_pixels": cap.max_pixels,
        "backend": spec.get("backend", "hf"),
        "device": device_name(),
    }
    need = not out.exists() or len(
        {r["id"] for r in read_jsonl(out)} & {s["id"] for s in samples}
    ) < len(samples)
    if need:
        gen = generator or build_generator(spec, gen_cfg, cap, model_id)
        run_predictions(
            gen, samples, out, int(spec.get("batch_size", cfg.get("batch_size", 1))), meta
        )
    scores = score_predictions(samples, read_jsonl(out))
    return scores, ScoreTable.from_scores(scores)
