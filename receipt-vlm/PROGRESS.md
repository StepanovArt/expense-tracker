# PROGRESS.md

## Current phase

Validation complete on a GPU (session 2, see below); the single test run (all six models +
RQ4 error analysis) is in progress. Remaining after it: RQ1–RQ4 paragraphs from the test
tables, `make merge` / serving through vLLM, Hub upload, wiring `/extract` into the bot.

## GPU run (session 2, 2026-10-23/24)

- Colab (free T4) was abandoned: the GPU quota ran out mid-run, and SFT silently trained on
  the CPU. `torch 2.13.0+cu130` needs an NVIDIA driver >= 580; on an older driver
  `torch.cuda.is_available()` is False. Added `make gpu-check`, `require_cuda: true` in the GPU
  configs and the device in every prediction file (DECISIONS #45). Colab's WildReceipt build
  was also incomplete (92 instead of 141 val receipts), so its B0 predictions were discarded.
- RunPod, 1x RTX 4090 24 GB (driver with CUDA 13.2), `gpu16` configs (DECISIONS #43).
- Data: all four sources built, 6,324 samples (train 4,788 / val 602 / test 934); dedup
  dropped 35 (15 near-duplicates of test, 13 of val, 7 exact); `pytest -m data` passed, i.e. the
  CORD converter now ran over the real records. The frozen SROIE + synthetic test entries were
  reproduced exactly; the CORD / WildReceipt test entries are appended and committed.
- Val (n = 602), strict micro-F1 [95 % CI]: B0 17.7 [14.9, 20.3], B1 29.9 [26.2, 33.3],
  B2 60.8 [57.4, 63.9], B3 78.1 [76.0, 80.0], M1 92.4 [91.1, 93.4], M2 94.6 [93.5, 95.6];
  p50 latency M1 3.57 s vs B3 8.11 s. Full tables: `reports/RESULTS.md`.
- GRPO checks: the hacking monitor flagged step 240 (null rate +5.6 pts while reward rose by
  0.007), but precision and recall both rose from M1 to M2 (92.8 / 92.0 -> 94.6 / 94.7), so the
  extra nulls replaced hallucinated values. Invalid outputs fell from 22 to 3; on the 580 val
  receipts where both outputs are valid, F1 goes 93.1 -> 94.9. The best GRPO step was the last
  one (300), so GRPO had not plateaued.
- Not done: the projector ablation (`lora.train_projector=true`); a check of
  `build_info.json["dropped_ids"]` for false near-duplicates among CORD / WildReceipt.

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

- **Phase 8** — `scripts/merge_lora.py` (stacked adapters; merged logits = adapter logits within
  1e-5, tested), FastAPI `/extract` (validation, one retry, 422/400/413/502; mocked-backend
  tests), `scripts/serve.sh` + `docker-compose.yml` (vLLM GPU) + `docker-compose.cpu.yml`,
  Gradio demo (serves, 6 deterministic synthetic test examples), `scripts/push_to_hub.py` +
  model card (says "not evaluated" until test metrics exist), `examples/telegram_bot_client.py`.
  - **DoD check run here:** `docker compose -f docker-compose.cpu.yml up` → `curl -F image=@…
    localhost:8000/extract` → 200 with valid receipt JSON (tiny memorising model, CPU).
  - Not run here: the vLLM container (needs GPU), HF upload (no token), llama.cpp recipe.
- `make smoke`: whole pipeline with real CLI scripts in 81 s on CPU.

- **Phase 9** — README as a technical report (method formulas, protocol, honest status,
  reproduction, verification summary, limitations, licences); `.python-version` = 3.12; CI
  workflow (`.github/workflows/receipt-vlm.yml`: `make check` + `make smoke`).
  - **DoD check run here:** fresh clone of the pushed branch → `make setup && make smoke` →
    exit 0 (setup ≈ 2.5 min, smoke 91 s). README contains no numbers that are not in
    RESULTS.md (there are none yet).
  - Gradio UI checked in a headless browser (mocked model): summary, items table, JSON and the
    6 examples render, Arabic correctly.

## Next (on a GPU machine — see "Must run" below)

1. Finish the test run (in progress) → `make report` → `scripts/error_analysis.py --model M2`
   → write the RQ1–RQ4 paragraphs in RESULTS.md and the README results section from the
   generated tables only.
2. `make merge-m2` (M2 is the best model on val), `make serve` (vLLM), measure vLLM latency,
   `scripts/push_to_hub.py`.

## Known issues

- pHash threshold (≤ 6 of 256 bits) was calibrated on SROIE only; after adding CORD/WildReceipt,
  check `data/processed/build_info.json["dropped_ids"]` for false near-duplicates.

## Must run on a GPU / Hub-connected machine (exact commands)

```bash
make setup-gpu
make gpu-check      # must print the GPU name; torch is built for CUDA 13 (driver >= 580)
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
