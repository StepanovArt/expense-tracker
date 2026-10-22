# Serving

The API (`src/receipt_vlm/serve/api.py`) never trusts the model: every answer is parsed and
validated against the `Receipt` schema with the evaluation code; an invalid answer is retried
once with a "return valid JSON only" reminder appended to the conversation, and a second
failure returns **HTTP 422** with the raw text. Responses also carry an arithmetic check
(Σ items = subtotal, subtotal + tax = total).

```bash
curl -F image=@receipt.jpg localhost:8000/extract
# 200 {"receipt": {...}, "consistency": {...}, "latency_s": 0.58, "attempts": 1, "model": "...", "prompt_version": "v1"}
# 422 {"error": "model output is not valid receipt JSON", "status": "parse_fail", "raw": "...", "attempts": 2}
# 400 not an image, 413 > 15 MB, 502 model backend unreachable
```

## 1. GPU: vLLM + API (recommended)

```bash
make merge                      # outputs/merged = base + SFT (+ GRPO) adapters
make serve                      # vllm serve on :8001 + FastAPI on :8000   (scripts/serve.sh)
# or containers:
docker compose up --build       # vllm/vllm-openai:v0.30.0 (same version as uv.lock) + API
```

vLLM gets the same image cap as evaluation (`--mm-processor-kwargs '{"min_pixels": 65536,
"max_pixels": 1048576}'`) and greedy decoding (the API sends `temperature: 0`).

## 2. CPU / no vLLM: in-process transformers backend

```bash
RECEIPT_VLM_BACKEND=hf RECEIPT_VLM_MODEL=outputs/merged make api
# or containers (image built with the `ml` extra):
RECEIPT_VLM_MODEL_DIR=./outputs/merged docker compose -f docker-compose.cpu.yml up --build
```

Verified in the development container (no GPU) with the tiny test model: `docker compose -f
docker-compose.cpu.yml up` → `curl -F image=@… localhost:8000/extract` returned a valid
receipt (200), an unreadable input returned 422 after one retry, a non-image 400. A 2B model
on CPU takes tens of seconds per receipt.

## 3. Fallback: llama.cpp (GGUF)

If vLLM cannot run (no supported GPU, Windows, Apple Silicon), convert the merged model to
GGUF and serve it with `llama-server`, which exposes the same OpenAI-compatible API, so the
FastAPI layer is unchanged (`RECEIPT_VLM_BACKEND=openai`, `VLLM_BASE_URL=http://localhost:8080/v1`):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j
python convert_hf_to_gguf.py ../outputs/merged --outfile receipt-vlm-f16.gguf --outtype f16
python convert_hf_to_gguf.py ../outputs/merged --mmproj --outfile mmproj-receipt-vlm-f16.gguf
./build/bin/llama-quantize receipt-vlm-f16.gguf receipt-vlm-q4_k_m.gguf Q4_K_M
./build/bin/llama-server -m receipt-vlm-q4_k_m.gguf --mmproj mmproj-receipt-vlm-f16.gguf \
    --port 8080 --temp 0 -c 8192
```

Not executed in this repository's development environment (no GPU, and Qwen3-VL support in
llama.cpp depends on its version): treat as a recipe, and re-run `make eval` against it before
reporting any number, since quantisation and llama.cpp's image preprocessing differ from HF.
