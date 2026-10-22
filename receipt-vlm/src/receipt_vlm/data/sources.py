"""Download + convert each real source. A source that cannot be fetched is reported as a gap
(CLAUDE.md §4.1: continue with the others and document it), never silently replaced."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from receipt_vlm.data.record import Converted
from receipt_vlm.utils.logging import get_logger

log = get_logger(__name__)


class SourceUnavailableError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=1800)


def fetch_sroie(raw: Path, cfg: dict[str, Any]) -> Path:
    dest = raw / "sroie"
    if (dest / "data" / "key").is_dir():
        return dest
    try:
        tmp = raw / "_sroie_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        _run(["git", "clone", "--depth", "50", cfg["git_url"], str(tmp)])
        rev = cfg.get("git_rev")
        if rev:
            try:
                _run(["git", "checkout", "-q", rev], cwd=tmp)
            except subprocess.CalledProcessError:
                _run(["git", "fetch", "-q", "--depth", "1", "origin", rev], cwd=tmp)
                _run(["git", "checkout", "-q", rev], cwd=tmp)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / "data"), str(dest / "data"))
        shutil.rmtree(tmp, ignore_errors=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        raise SourceUnavailableError(f"SROIE mirror clone failed: {e}") from e
    return dest


def fetch_cord(raw: Path, cfg: dict[str, Any]) -> Path:
    """Export CORD v2 from the Hub to local parquet files (one-time)."""
    dest = raw / "cord"
    if sorted(dest.glob("*.parquet")):
        return dest
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.get("hf_dataset", "naver-clova-ix/cord-v2"))
    except Exception as e:  # network/policy errors surface as many exception types
        raise SourceUnavailableError(f"CORD v2 download from the HF Hub failed: {e}") from e
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        ds[split].to_parquet(str(dest / f"{split}-00000.parquet"))
    return dest


def fetch_wildreceipt(raw: Path, cfg: dict[str, Any]) -> Path:
    dest = raw / "wildreceipt"
    if (dest / "train.txt").exists():
        return dest
    tar_path = raw / "wildreceipt.tar"
    try:
        if not tar_path.exists():
            urllib.request.urlretrieve(cfg["url"], tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(raw, filter="data")
    except Exception as e:
        raise SourceUnavailableError(f"WildReceipt download failed: {e}") from e
    if not (dest / "train.txt").exists():
        raise SourceUnavailableError(f"WildReceipt archive did not contain {dest}/train.txt")
    return dest


def iter_source(name: str, raw: Path, cfg: dict[str, Any]) -> Iterator[Converted]:
    if name == "sroie":
        from receipt_vlm.data.converters.sroie import iter_sroie

        yield from iter_sroie(fetch_sroie(raw, cfg))
    elif name == "cord":
        import pandas as pd

        from receipt_vlm.data.converters.cord import convert_row

        root = fetch_cord(raw, cfg)
        for f in sorted(root.glob("*.parquet")):
            split = f.stem.split("-")[0]
            for i, row in enumerate(pd.read_parquet(f).to_dict("records")):
                yield convert_row({str(k): v for k, v in row.items()}, split, i)
    elif name == "wildreceipt":
        from receipt_vlm.data.converters.wildreceipt import iter_wildreceipt

        yield from iter_wildreceipt(fetch_wildreceipt(raw, cfg))
    else:
        raise ValueError(f"unknown source {name}")
