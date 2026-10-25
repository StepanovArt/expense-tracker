"""scripts/restore_outputs.py: an archive of the val run is restored into outputs/ or rejected."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "restore_outputs", ROOT / "scripts" / "restore_outputs.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _archive(path: Path, files: list[str]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name in files:
            data = b"{}"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_restore_finds_outputs_at_any_depth(tmp_path: Path) -> None:
    mod = _module()
    prefix = "workspace/expense-tracker/receipt-vlm/outputs"
    files = [f"{prefix}/{f}" for f in (*mod.REQUIRED, *mod.EXPECTED, "metrics/val.json")]
    files.append(f"{prefix}/sft/checkpoints/epoch_1/sft/best/adapter_config.json")  # deeper copy
    copied = mod.restore(_archive(tmp_path / "r.tgz", files), tmp_path / "outputs")
    assert copied == ["metrics", "preds", "sft"]
    for f in mod.REQUIRED:
        assert (tmp_path / "outputs" / f).exists()


def test_restore_rejects_an_archive_without_what_the_ablation_needs(tmp_path: Path) -> None:
    mod = _module()
    no_adapter = _archive(tmp_path / "a.tgz", ["outputs/preds/M1/val.jsonl"])
    with pytest.raises(SystemExit, match="SFT adapter"):
        mod.restore(no_adapter, tmp_path / "o1")
    no_preds = _archive(tmp_path / "b.tgz", ["outputs/sft/best/adapter_config.json"])
    with pytest.raises(SystemExit, match=r"preds/B1/val\.jsonl"):
        mod.restore(no_preds, tmp_path / "o2")
