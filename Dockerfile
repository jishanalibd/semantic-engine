FROM python:3.11-slim

ARG MODEL_NAME=intfloat/multilingual-e5-small

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    DATA_DIR=/data \
    MODEL_NAME=${MODEL_NAME} \
    SEMANTIC_MODE=shadow

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake ONNX weights (registers intfloat/multilingual-e5-small when not built-in) so Coolify starts offline in <2s
RUN printf '%s\n' \
    'import os' \
    'from fastembed import TextEmbedding' \
    'from fastembed.common.model_description import ModelSource, PoolingType' \
    'm = os.environ.get("MODEL_NAME", "intfloat/multilingual-e5-small").strip()' \
    's = {str(x.get("model", "")).lower() for x in TextEmbedding.list_supported_models()}' \
    'if m.lower() == "intfloat/multilingual-e5-small" and m.lower() not in s:' \
    '    TextEmbedding.add_custom_model(model=m, pooling=PoolingType.MEAN, normalization=True, sources=ModelSource(hf=m), dim=384, model_file="onnx/model.onnx", size_in_gb=0.47)' \
    'TextEmbedding(model_name=m)' \
    | python -

COPY normalizer.py app.py train_setfit.py ./
RUN mkdir -p /data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
