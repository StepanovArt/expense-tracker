from pathlib import Path

import pytest

from receipt_vlm.utils.config import apply_overrides, deep_merge, load_config, require_cuda
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


def test_gpu_configs_refuse_to_run_without_cuda() -> None:
    configs = Path(__file__).parents[1] / "configs"
    gpu = ("sft/gpu16.yaml", "sft/gpu24plus.yaml", "sft/control_gpu16.yaml")
    for name in (*gpu, "grpo/gpu16.yaml", "grpo/from_base_gpu16.yaml"):
        cfg = load_config(configs / name)
        with pytest.raises(RuntimeError, match="needs a CUDA GPU"):
            require_cuda(cfg, cuda_available=False)
        require_cuda(cfg, cuda_available=True)
    for name in ("sft/smoke.yaml", "grpo/smoke.yaml"):  # CPU smoke runs stay allowed
        require_cuda(load_config(configs / name), cuda_available=False)


def _flat(d: dict, prefix: str = "") -> dict:  # type: ignore[type-arg]
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_ablation_runs_differ_from_m2_only_where_intended() -> None:
    configs = Path(__file__).parents[1] / "configs"
    m2 = load_config(configs / "grpo/gpu16.yaml")
    m3 = load_config(configs / "grpo/from_base_gpu16.yaml")
    a, b = _flat(m2), _flat(m3)
    changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert changed == {
        "run_name",
        "output_dir",
        "model.sft_adapter",
        "model.merged_sft_dir",
        "_meta.config_path",
    }

    # C1 = more SFT from M2's starting point, on the same subset and as many receipts
    c1 = load_config(configs / "sft/control_gpu16.yaml")
    assert c1["model"]["name"] == m2["model"]["merged_sft_dir"]
    assert not c1["model"]["load_in_4bit"]
    assert (c1["seed"], c1["data"]["max_train_samples"]) == (
        m2["seed"],
        m2["data"]["max_train_samples"],
    )
    assert all(c1["lora"][k] == m2["lora"][k] for k in ("r", "alpha", "dropout"))
    g, t, ct = m2["grpo"], m2["train"], c1["train"]
    grpo_receipts = t["max_steps"] * t["per_device_batch_size"] * t["grad_accum"]
    grpo_receipts //= g["num_generations"]
    assert ct["max_steps"] * ct["per_device_batch_size"] * ct["grad_accum"] == grpo_receipts

    # the eval config evaluates exactly what these runs write
    ev = load_config(configs / "eval/final.yaml")
    assert ev["models"]["M3"]["adapter"] == f"{m3['output_dir']}/best"
    assert ev["models"]["M3"]["model"] == m3["model"]["name"]
    assert ev["models"]["C1"]["adapter"] == f"{c1['output_dir']}/best"
    assert ev["models"]["C1"]["model"] == c1["model"]["name"]
    rep = ev["report"]
    assert set(rep["order"]) <= set(ev["models"])
    assert {m for pair in rep["comparisons"] for m in pair} <= set(rep["order"])
