"""YAML config loader with inheritance and dotted CLI overrides.

A config may declare ``_base_: other.yaml`` (path relative to itself); the base is loaded
first and the child is deep-merged on top. Overrides use ``a.b.c=value`` with YAML-parsed
values, so ``train.lr=1e-4`` gives a float and ``data.langs=[en,ar]`` gives a list.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from receipt_vlm.utils.io import resolve

BASE_KEY = "_base_"


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-4`` as a float (YAML 1.1 requires ``1.0e-4``)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, (*seen, path)))}")
    data = yaml_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(data).__name__}")
    base_ref = data.pop(BASE_KEY, None)
    if base_ref is None:
        return data
    bases = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    merged: dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, _load_yaml(path.parent / b, (*seen, path)))
    return deep_merge(merged, data)


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must look like key.sub=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key}: {p} is not a mapping")
        node[parts[-1]] = yaml_load(raw)
    return out


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    cfg = _load_yaml(resolve(path))
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def require_cuda(cfg: dict[str, Any], cuda_available: bool | None = None) -> None:
    """Fail fast when a config marked ``require_cuda: true`` runs without a CUDA device.

    The GPU configs (``*/gpu16.yaml``, ``*/gpu24plus.yaml``) would otherwise fall back to the
    CPU silently and run for days, which is what a Colab runtime without a GPU looks like.
    """
    if not cfg.get("require_cuda"):
        return
    if cuda_available is None:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError(
            f"{cfg.get('run_name', 'this config')} needs a CUDA GPU, but none is visible "
            "(check `nvidia-smi`; on Colab: Runtime -> Change runtime type -> T4 GPU)"
        )
