# receipt-vlm API image.
#   default (EXTRAS="--extra serve"):  thin client for an OpenAI-compatible model server (vLLM)
#   EXTRAS="--extra serve --extra ml": in-process transformers backend (CPU or GPU)
# Behind a TLS-intercepting proxy, pass its CA bundle as a build secret:
#   docker build --secret id=ca_bundle,src=/path/ca.crt ...
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export PIP_CERT=/run/secrets/ca_bundle; fi; \
    pip install --no-cache-dir uv==0.11.32

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
ARG EXTRAS="--extra serve"
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --frozen --no-dev --no-editable $EXTRAS && rm -rf /root/.cache

ENV PATH="/app/.venv/bin:$PATH" \
    RECEIPT_VLM_BACKEND=openai \
    VLLM_BASE_URL=http://vllm:8000/v1 \
    RECEIPT_VLM_MODEL=receipt-vlm
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "receipt_vlm.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
