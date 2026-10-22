# PROGRESS.md

## Current phase

Phase 8 — Serving, demo, release (next); GPU runs pending for Phases 3–7

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

- **Phase 3** — evaluation harness: parsing (code fences, `<think>`, first balanced object),
  normalisation, strict + lenient field P/R/F1, Hungarian item matching, total EM, arithmetic
  consistency, latency; bootstrap CIs, paired bootstrap, McNemar; per-language / per-source
  breakdowns with low-n flags; RESULTS.md auto-sections + 3 figures (checked visually on mock
  data). Backends: HF transformers (batched greedy = unbatched, tested), offline vLLM,
  OpenAI-compatible HTTP, OCR + LLM (Tesseract two-pass script-aware, PSM 6 chosen on val).
  Smoke: `scripts/run_eval.py --config configs/eval/smoke.yaml` runs the tiny random model end
  to end (preds JSONL → metrics JSON with CIs).
  - **DoD gap:** the baseline validation table needs B0–B3 predictions, which need Hub access
    + GPU. RESULTS.md lists them as *not run*; nothing is invented.

- **Phase 4** — SFT: `train/sft.py` (LoRA on LM attention+MLP, vision frozen, optional
  projector, QLoRA for gpu16, Unsloth switch), `data/collate.py` (assistant-only + partial-label
  masking), configs smoke/gpu16/gpu24plus, generation-based val-F1 checkpoint selection.
  CPU smoke: `make sft CONFIG=configs/sft/smoke.yaml` (14 s). Wiring test: tiny model memorises
  two receipts exactly (val F1 = 1.0) in 250 steps.
  - **DoD gap:** M1 training + paired comparison with B1 needs the GPU machine.
- **Phase 5** (test-set evaluation) is code-complete through `run_eval.py --split test` +
  `make report` (figures + breakdowns + paired tests); it is blocked only by missing predictions.

- **Phase 6** — GRPO: rewards reuse the eval code (17 tests incl. adversarial: empty, garbage,
  all-null, schema-invalid-but-right, correct abstention); `train/grpo.py` merges the SFT
  adapter so KL is to the SFT policy, logs reward components, null rate per field, output
  length, KL; flags null-rate-up-while-reward-up. CPU: PyTorch fallback for TRL's Triton
  log-prob kernel; slow test shows reward 1.0 for exact completions, non-zero gradients when
  group rewards vary, LoRA weights moving, KL > 0.
  - **DoD gap:** M2 training + M2 vs M1 paired test need the GPU machine.
- **Phase 7** — error analysis: `eval/errors.py` taxonomy + `scripts/error_analysis.py`
  (100 sampled errors + all-error counts per language, field table, error rate by language /
  source / image-sharpness tercile, 6 annotated examples). Verified on mock predictions only.
  - **DoD gap:** needs real predictions of the best model.

## Next

- Phase 8: merge script, FastAPI /extract (retry + 422), vLLM/llama.cpp serving, Docker,
  Gradio demo, HF Hub push + model card, Telegram client example.

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
make synth           # re-merge synthetic: the 50 % cap now admits more synthetic train
uv run pytest -m data
git add data/splits reports/data_stats.md && git commit -m "data: freeze CORD/WildReceipt test manifests"
make eval-baselines SPLIT=val      # B0-B3 zero-shot on validation (Phase 3 DoD)
make report                        # fills reports/RESULTS.md (val section)
make sft CONFIG=configs/sft/gpu16.yaml                      # M1 (Phase 4)
uv run python scripts/train_sft.py --config configs/sft/gpu16.yaml \
    output_dir=outputs/sft_proj lora.train_projector=true   # projector ablation; keep the val winner
make eval MODEL=M1 SPLIT=val && make report                 # RQ1 on val (M1 vs B1 paired)
make grpo CONFIG=configs/grpo/gpu16.yaml                    # M2 (Phase 6); watch reward_hacking_warning.json
make eval MODEL=M2 SPLIT=val && make report                 # RQ2 on val first
# --- frozen test set, once ---
uv run python scripts/run_eval.py --config configs/eval/final.yaml --split test --models B0 B1 B2 B3 M1 M2
make report
uv run python scripts/error_analysis.py --model M2 --split test   # RQ4 (or the best model)
# then write the RQ1-RQ4 paragraphs in reports/RESULTS.md from the generated tables
```
