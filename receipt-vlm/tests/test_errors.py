import json

from receipt_vlm.eval.errors import classify, script_of

KEYS = (
    "merchant",
    "date",
    "time",
    "currency",
    "subtotal",
    "tax",
    "total",
    "payment_method",
    "language",
)


def _s(target: dict, labelled: list[str]) -> dict:
    return {
        "id": "x",
        "target": {**dict.fromkeys(KEYS), "items": [], **target},
        "labelled": labelled,
    }


GOLD = _s(
    {
        "merchant": "كارفور",
        "date": "2024-03-05",
        "total": 34.5,
        "items": [{"name": "حليب", "total": 7.0}],
    },
    ["merchant", "date", "time", "total", "tax", "items.name", "items.total"],
)


def _cats(raw: str, **kw: int) -> list[str]:
    return sorted(e.category for e in classify(GOLD, raw, **kw))


def test_output_level_failures() -> None:
    assert _cats("no json here") == ["parse_fail"]
    assert _cats(
        '{"merchant": "كارفور", "items": [{"name": "حل', n_output_tokens=50, max_new_tokens=50
    ) == ["truncated_output"]
    assert _cats(json.dumps({"date": "05/03/2024"})) == ["schema:date_format"]
    assert _cats(json.dumps({"total": "34.50"})) == ["schema:number_as_str"]
    assert _cats(json.dumps({"vendor": "x"})) == ["schema:other"]


def test_field_level_taxonomy() -> None:
    pred = {
        "merchant": "Carrefour",
        "date": "2024-03-06",
        "time": "10:00",
        "total": 35.0,
        "tax": None,
        "items": [{"name": "حليب", "total": 7.0}, {"name": "خبز", "total": 2.0}],
    }
    events = classify(GOLD, json.dumps(pred))
    cats = sorted(e.category for e in events)
    # Arabic merchant transliterated to Latin -> script_mismatch; time not printed -> hallucinated
    assert cats == [
        "extra_item",
        "hallucinated_field",
        "script_mismatch",
        "wrong_date",
        "wrong_number",
    ]
    assert {e.field for e in events if e.category == "wrong_number"} == {"total"}


def test_missing_item_and_field() -> None:
    pred = {"merchant": "كارفور", "date": "2024-03-05", "items": []}
    assert _cats(json.dumps(pred)) == ["missing_field", "missing_item"]
    assert classify(GOLD, json.dumps(GOLD["target"])) == []


def test_script_of() -> None:
    assert (script_of("كارفور"), script_of("Магазин"), script_of("Shop 24"), script_of("123")) == (
        "arabic",
        "cyrillic",
        "latin",
        "none",
    )


def test_error_analysis_script_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Runs scripts/error_analysis.py on mock predictions over generated images."""
    import importlib.util
    from pathlib import Path

    from receipt_vlm.data.synthetic.generate import generate_one
    from receipt_vlm.utils.io import write_jsonl

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("ea", root / "scripts" / "error_analysis.py")
    ea = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ea)  # type: ignore[union-attr]

    samples, preds = [], []
    for i in range(9):
        r = generate_one((i, 7, str(tmp_path / "syn")))
        s = {
            "id": f"synthetic/{i}",
            "source": "synthetic",
            "language": r["language"],
            "image": r["image"],
            "target": r["target"],
            "labelled": r["labelled"],
        }
        samples.append(s)
        t = dict(r["target"])
        if i % 3 == 0:
            t["total"] = (t["total"] or 0) + 1
        raw = "garbage" if i == 4 else json.dumps(t, ensure_ascii=False)
        preds.append({"id": s["id"], "raw": raw, "n_output_tokens": 5})
    write_jsonl(tmp_path / "data" / "val.jsonl", samples)
    write_jsonl(tmp_path / "preds" / "X" / "val.jsonl", preds)
    cfg = {
        "seed": 1,
        "data_dir": str(tmp_path / "data"),
        "preds_dir": str(tmp_path / "preds"),
        "generation": {"max_new_tokens": 100},
    }
    info = ea.analyse(cfg, "X", "val", 100, 6, str(tmp_path / "R.md"), str(tmp_path / "fig"))
    assert info["n_wrong"] == 4 and info["categories"]["wrong_number"] == 3
    assert info["categories"]["parse_fail"] == 1
    text = (tmp_path / "R.md").read_text()
    assert "Error taxonomy" in text and "sharpness" in text
    assert (tmp_path / "fig" / "errors_X_val.png").stat().st_size > 10_000
