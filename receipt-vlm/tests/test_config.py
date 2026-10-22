from pathlib import Path

import pytest

from receipt_vlm.utils.config import apply_overrides, deep_merge, load_config
from receipt_vlm.utils.seed import derive_seed, seed_everything


def test_inheritance_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("train:\n  lr: 1.0e-4\n  epochs: 2\nseed: 1\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain:\n  epochs: 3\n")
    cfg = load_config(tmp_path / "child.yaml", ["train.lr=2e-4", "data.langs=[en,ar]"])
    assert cfg["train"] == {"lr": 2e-4, "epochs": 3}
    assert cfg["seed"] == 1
    assert cfg["data"]["langs"] == ["en", "ar"]


def test_cycle_detected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"b": 1}}
    out = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}} and out == {"a": {"b": 1, "c": 2}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


@pytest.mark.parametrize(
    "path",
    sorted(str(p) for p in (Path(__file__).parents[1] / "configs").rglob("*.yaml")),
)
def test_all_repo_configs_load(path: str) -> None:
    cfg = load_config(path)
    assert isinstance(cfg, dict) and cfg


def test_seeds_are_stable() -> None:
    import random

    import numpy as np

    seed_everything(7)
    a = (random.random(), float(np.random.rand()))
    seed_everything(7)
    assert a == (random.random(), float(np.random.rand()))
    assert derive_seed(1, "sroie", "en") == derive_seed(1, "sroie", "en")
    assert derive_seed(1, "sroie", "en") != derive_seed(1, "cord", "en")
