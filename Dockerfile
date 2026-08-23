FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml run.py ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

CMD ["python", "run.py"]
