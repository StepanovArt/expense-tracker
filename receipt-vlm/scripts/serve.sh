#!/usr/bin/env bash
# Serve the merged model with vLLM (OpenAI-compatible, port 8001) and the FastAPI validator
# (port 8000). Fallback without vLLM: see docs/serving.md (llama.cpp / in-process backend).
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-outputs/merged}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
[ -f "$MODEL_DIR/config.json" ] || { echo "no merged model in $MODEL_DIR - run 'make merge' first" >&2; exit 1; }

uv run vllm serve "$MODEL_DIR" --served-model-name receipt-vlm --port 8001 \
  --max-model-len 8192 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs "{\"min_pixels\": 65536, \"max_pixels\": $MAX_PIXELS}" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -fsS localhost:8001/v1/models >/dev/null 2>&1; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited" >&2; exit 1; }
  sleep 3
done
RECEIPT_VLM_BACKEND=openai VLLM_BASE_URL=http://localhost:8001/v1 RECEIPT_VLM_MODEL=receipt-vlm \
  uv run uvicorn receipt_vlm.serve.api:app --host 0.0.0.0 --port 8000
