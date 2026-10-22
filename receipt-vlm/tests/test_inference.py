"""Inference backends, runner resume, and report generation (tiny model / mocks, CPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from receipt_vlm.eval.runner import run_predictions, score_predictions, select_subset
from receipt_vlm.models.inference import MockGenerator
from receipt_vlm.utils.io import read_jsonl


def _img(h: int, w: int, seed: int) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    from receipt_vlm.models.tiny import build_tiny_qwen3_vl

    return build_tiny_qwen3_vl(tmp_path_factory.mktemp("tiny"))


@pytest.mark.slow
def test_batched_greedy_equals_unbatched(tiny: Path) -> None:
    from receipt_vlm.models.inference import HFVisionGenerator
    from receipt_vlm.models.load import ImageCap

    gen = HFVisionGenerator(
        str(tiny), dtype="float32", max_new_tokens=12, cap=ImageCap(4096, 65536)
    )
    ims = [_img(300, 200, 0), _img(120, 400, 1)]  # different sizes -> different prompt lengths
    batched = [o.raw for o in gen.generate(ims)]
    single = [gen.generate([im])[0].raw for im in ims]
    assert batched == single


@pytest.mark.slow
def test_image_cap_bounds_visual_tokens(tiny: Path) -> None:
    from transformers import AutoProcessor

    from receipt_vlm.models.load import ImageCap, set_image_cap
    from receipt_vlm.prompts import vlm_messages

    proc = AutoProcessor.from_pretrained(tiny)
    image_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    counts = []
    for max_px in (32 * 32 * 16, 32 * 32 * 64):
        set_image_cap(proc, ImageCap(min_pixels=4096, max_pixels=max_px))
        ids = proc.apply_chat_template(
            vlm_messages(_img(1600, 700, 2)),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"][0]
        n = sum(1 for t in ids if t == image_id)
        counts.append(n)
        assert n <= max_px // (32 * 32)  # one merged token per 32x32 pixels
    assert counts[0] < counts[1]


def _samples(n: int) -> list[dict]:
    return [
        {
            "id": f"s/{i}",
            "source": "s",
            "language": "en",
            "image": "unused",
            "labelled": ["total"],
            "target": {"total": float(i), "items": []},
        }
        for i in range(n)
    ]


def test_runner_resumes_and_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import receipt_vlm.eval.runner as runner

    monkeypatch.setattr(runner, "load_image", lambda p: _img(10, 10, 0))
    out = tmp_path / "preds.jsonl"
    samples = _samples(5)
    calls: list[int] = []

    def answer(i: int) -> str:
        calls.append(i)
        return json.dumps({"total": 3.0})

    run_predictions(MockGenerator(answer), samples[:2], out, batch_size=2)
    run_predictions(MockGenerator(answer), samples, out, batch_size=2)  # resumes: 3 new
    assert len(calls) == 5 and len(read_jsonl(out)) == 5
    scores = score_predictions(samples, read_jsonl(out))
    assert [s.total_correct for s in scores] == [0, 0, 0, 1, 0]
    with pytest.raises(ValueError, match="no prediction"):
        score_predictions(_samples(6), read_jsonl(out))


def test_select_subset_is_deterministic_and_stratified() -> None:
    rows = [{"id": f"a/{i}", "source": "a"} for i in range(100)] + [
        {"id": f"b/{i}", "source": "b"} for i in range(10)
    ]
    sub = select_subset(rows, 20, seed=1)
    assert sub == select_subset(list(reversed(rows)), 20, seed=1)
    assert sum(r["source"] == "b" for r in sub) == 10  # small source fully represented
    assert len(select_subset(rows, None, 1)) == 110


def test_report_section_is_regenerated_in_place(tmp_path: Path) -> None:
    from receipt_vlm.eval.report import replace_auto_section

    md = tmp_path / "RESULTS.md"
    md.write_text("# Results\n\n## RQ1\nhand-written analysis\n")
    replace_auto_section(md, "val", "table v1", "Val")
    replace_auto_section(md, "val", "table v2", "Val")
    text = md.read_text()
    assert "hand-written analysis" in text and "table v2" in text and "table v1" not in text
    assert text.count("BEGIN AUTO:val") == 1


def test_headline_table_lists_models_not_run() -> None:
    from receipt_vlm.eval.report import headline_table

    t = headline_table([], {"B3": "Qwen3-VL-8B — no complete predictions"})
    assert "B3" in t and "not run" in t


def test_full_report_from_mock_predictions(tmp_path: Path) -> None:
    """End-to-end make_report on fake predictions: tables, comparisons and figures render."""
    import importlib.util

    from receipt_vlm.utils.io import write_jsonl

    spec = importlib.util.spec_from_file_location(
        "make_report", Path(__file__).parents[1] / "scripts" / "make_report.py"
    )
    mr = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mr)  # type: ignore[union-attr]

    rng = np.random.default_rng(0)
    samples = []
    for i in range(60):
        lang = ["en", "ar", "ru"][i % 3]
        samples.append(
            {
                "id": f"synthetic/{i}",
                "source": "synthetic" if i % 2 else "sroie",
                "language": lang,
                "image": "x",
                "labelled": ["total", "merchant"],
                "target": {"merchant": f"Shop {i}", "total": float(i), "items": []},
            }
        )
    write_jsonl(tmp_path / "data" / "val.jsonl", samples)
    for mid, p_ok in (("GOOD", 0.9), ("BAD", 0.4)):
        rows = []
        for s in samples:
            ok = rng.random() < p_ok
            pred = {
                "merchant": s["target"]["merchant"],
                "total": s["target"]["total"] + (0 if ok else 1),
            }
            rows.append(
                {"id": s["id"], "raw": json.dumps(pred), "latency_s": 0.5 if mid == "GOOD" else 1.5}
            )
        write_jsonl(tmp_path / "preds" / mid / "val.jsonl", rows)
    cfg = {
        "seed": 1,
        "data_dir": str(tmp_path / "data"),
        "preds_dir": str(tmp_path / "preds"),
        "bootstrap": {"resamples": 200},
        "generation": {"max_new_tokens": 16},
        "image": {"max_pixels": 1},
        "models": {
            "GOOD": {"desc": "good"},
            "BAD": {"desc": "bad"},
            "MISSING": {"desc": "never run"},
        },
        "report": {
            "ours": ["GOOD"],
            "order": ["BAD", "GOOD", "MISSING"],
            "comparisons": [["GOOD", "BAD"]],
        },
    }
    out = tmp_path / "RESULTS.md"
    mr.build_split(cfg, "val", str(out), str(tmp_path / "fig"))
    text = out.read_text()
    assert "| **GOOD** | good | 60 |" in text and "MISSING" in text and "not run" in text
    assert "GOOD vs BAD" in text and "McNemar" in text
    assert "⚠ low-n" in text  # 20 images per language < 50
    for f in ("val_overall_f1.png", "val_field_f1.png", "val_latency_vs_f1.png"):
        assert (tmp_path / "fig" / f).stat().st_size > 1000
