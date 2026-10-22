# PROGRESS.md

## Current phase

Phase 0 — Scaffold (in progress)

## Environment facts (session 1, 2026-10-22)

- No GPU. 4 CPU cores, 15 GB RAM.
- Network policy blocks huggingface.co (no Qwen3-VL weights, no CORD/WildReceipt from the Hub),
  download.pytorch.org, download.openmmlab.com, Google Drive, Kaggle. PyPI, apt and public
  GitHub git clones work.
- Consequence: all real-model runs (B0–B3, M1, M2) must happen on a GPU machine with Hub access
  (Colab/Kaggle/cloud) using `notebooks/train_colab.ipynb`. Nothing in this repo reports numbers
  for runs that were not executed.

## Done

- Repo layout, uv project (Python 3.12), Makefile, config loader with `_base_` inheritance,
  seed utils, IO utils, metrics logger (W&B optional).

## Next

- `make check` green → close Phase 0.

## Known issues

- (none yet)

## Must run on GPU (exact commands)

(filled in as phases land)
