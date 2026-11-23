FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    DATA_DIR=/data \
    MODEL_NAME=intfloat/multilingual-e5-small \
    SEMANTIC_MODE=shadow

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake ONNX multilingual-e5-small weights so Coolify starts offline in <2s
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='intfloat/multilingual-e5-small')"

COPY normalizer.py app.py train_setfit.py ./FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    DATA_DIR=/data \
    MODEL_NAME=intfloat/multilingual-e5-small \
    SEMANTIC_MODE=shadow

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake step commented out to resolve fastembed model name error
# RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='intfloat/multilingual-e5-small')"

COPY normalizer.py app.py train_setfit.py ./
RUN mkdir -p /data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

RUN mkdir -p /data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
