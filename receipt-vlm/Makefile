.DEFAULT_GOAL := help
SHELL := /bin/bash

PY      ?= uv run python
SPLIT   ?= val
CONFIG  ?= configs/sft/gpu16.yaml
MODEL   ?= M1
EVALCFG ?= configs/eval/final.yaml
EXTRAS  ?= --extra ml --extra ocr --extra serve --extra hub

.PHONY: help setup setup-gpu data synth eval-baselines sft grpo eval report merge merge-m2 serve api demo smoke \
        lint format typecheck test test-all check docker clean-smoke

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

setup:  ## uv sync (CPU extras) + pre-commit hooks
	uv sync $(EXTRAS)
	-uv run pre-commit install

setup-gpu:  ## uv sync incl. bitsandbytes/unsloth/vllm (GPU machine only)
	uv sync $(EXTRAS) --extra gpu --extra tracking

data:  ## download sources, convert, dedup, split (freezes the test manifest)
	$(PY) scripts/build_data.py --config configs/data.yaml

synth:  ## generate synthetic EN/AR/RU receipts and merge them into the dataset
	$(PY) scripts/generate_synthetic.py --config configs/data.yaml
	$(PY) scripts/build_data.py --config configs/data.yaml --merge-only

eval-baselines:  ## B0-B3 zero-shot on SPLIT (val|test)
	$(PY) scripts/run_eval.py --config configs/eval/baselines.yaml --split $(SPLIT) --models B0 B1 B2 B3

sft:  ## LoRA SFT (CONFIG=configs/sft/{smoke,gpu16,gpu24plus}.yaml)
	$(PY) scripts/train_sft.py --config $(CONFIG)

grpo:  ## GRPO on top of the best SFT checkpoint (CONFIG=configs/grpo/{smoke,gpu16}.yaml)
	$(PY) scripts/train_grpo.py --config $(CONFIG)

eval:  ## evaluate one model id from the eval config (MODEL=M1 SPLIT=val)
	$(PY) scripts/run_eval.py --config $(EVALCFG) --split $(SPLIT) --models $(MODEL)

report:  ## regenerate reports/RESULTS.md + figures from saved predictions
	$(PY) scripts/make_report.py --config configs/eval/final.yaml

merge:  ## merge the SFT adapter (M1) into the base weights -> outputs/merged
	$(PY) scripts/merge_lora.py --adapter outputs/sft/best --out outputs/merged

merge-m2:  ## M2: GRPO adapter on top of the merged SFT model -> outputs/merged
	$(PY) scripts/merge_lora.py --base outputs/sft/merged --adapter outputs/grpo/best --out outputs/merged

serve:  ## vLLM OpenAI server on the merged model + FastAPI /extract (needs GPU)
	bash scripts/serve.sh

api:  ## FastAPI only (expects a running OpenAI-compatible backend at VLLM_BASE_URL)
	uv run uvicorn receipt_vlm.serve.api:app --host 0.0.0.0 --port 8000

demo:  ## Gradio demo
	$(PY) app/gradio_app.py

smoke:  ## full pipeline on tiny data with a tiny random model, CPU, < 5 min
	$(PY) scripts/smoke.py

docker:  ## build + start the API stack
	docker compose up --build

lint:  ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## autoformat
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## mypy (strict on schema/eval/rewards)
	uv run mypy

test:  ## fast unit tests (no model building)
	uv run pytest -m "not slow"

test-all:  ## all tests incl. tiny-model tests (CPU, a few minutes)
	uv run pytest

check: lint typecheck test-all  ## everything CI runs

clean-smoke:
	rm -rf outputs/smoke
