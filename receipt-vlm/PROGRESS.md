# PROGRESS.md

## Current phase

Phase 3 — Evaluation harness + baselines (next)

## Environment facts (session 1, 2026-10-22)

- No GPU. 4 CPU cores, 15 GB RAM.
- Network policy blocks huggingface.co (no Qwen3-VL weights, no CORD from the Hub),
  download.pytorch.org, download.openmmlab.com (WildReceipt), Google Drive, Kaggle.
  PyPI, apt and public GitHub git clones work.
- Consequence: all real-model runs (B0–B3, M1, M2) must happen on a GPU machine with Hub access
  (Colab/Kaggle/cloud) using `notebooks/train_colab.ipynb`. Nothing in this repo reports numbers
  for runs that were not executed.

## Done

- **Phase 0** — uv project (Python 3.12), Makefile, config loader (`_base_` inheritance, dotted
  overrides, `1e-4` parsed as float), seed/IO/logging utils. `make check` green.
- **Phase 1** — `schema.py` (strict: numbers must be numbers, real ISO dates, extra keys
  rejected), shared value parsers (`values.py`), converters for SROIE / CORD / WildReceipt with
  per-sample label availability, dedup (sha256 + 256-bit pHash, threshold calibrated on SROIE),
  split assignment (official splits respected, near-dup groups atomic, per-stratum seeds),
  frozen test manifest with append-only semantics, leakage tests on committed manifests,
  `reports/data_stats.md`, `notebooks/01_data_exploration.ipynb`.
  - Built here: SROIE only → 497 / 62 / 62 (5 exact duplicates dropped).
  - CORD and WildReceipt are **gaps in this session** (hosts blocked); `make data` on a machine
    with access appends them (frozen SROIE test entries cannot move).

- **Phase 2** — synthetic generator (`data/synthetic/`): 3 languages × 3 layouts, currencies
  AED/USD/RUB/EUR, exact consistent ground truth, Arabic shaped + RTL, Arabic-Indic digits on
  ~30 % of Arabic receipts, degradations (crumple, perspective, ±7° rotation, lighting, blur,
  noise, JPEG 35–91). 3,000 samples in ~8 min on 4 CPUs. Merged with the 50 % cap:
  train 994 (497 SROIE + 497 synthetic), val 362, test 362 (synthetic test = 300, 100/language).
  - Visual check of `reports/figures/synthetic_samples.png` (20 random samples): Arabic letters
    are joined and read right-to-left (merchant names, items, "الإجمالي"), RTL layout mirrored
    (labels right, amounts left), Arabic-Indic digits render; Cyrillic correct; every checked
    receipt (one per language, by hand) matches its gold JSON line by line and the arithmetic
    holds. Fixed after the first look: pink/green paper tint and saturated backgrounds
    (now near-white paper on grey/brown tables). Remaining realism gaps: no handwriting, no
    logos, no folded-over corners, no motion blur; long receipts get small text after the
    1600 px cap.

## Next

- Phase 3: eval harness (done in code, metric tests pass) + CPU smoke of the baseline runner
  with the tiny model; baselines B0–B3 themselves need the GPU machine.

## Known issues

- CORD converter is tested on structure-faithful fixtures, not verbatim records (HF blocked).
  First thing on the GPU machine: `make data && uv run pytest -m data` (runs the converter over
  all 1,000 real CORD records) and inspect `reports/data_stats.md`.
- pHash threshold (≤ 6 of 256 bits) was calibrated on SROIE only; after adding CORD/WildReceipt,
  check `data/processed/build_info.json["dropped_ids"]` for false near-duplicates.

## Must run on a GPU / Hub-connected machine (exact commands)

```bash
make setup-gpu
make data            # adds CORD + WildReceipt; SROIE test entries stay frozen
uv run pytest -m data
git add data/splits && git commit -m "data: freeze CORD/WildReceipt test manifests"
```
