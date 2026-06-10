# Цель: только инференс из готовой .cbm модели.
# Контракт: /data/input/holdout.csv → /data/output/predictions.csv

FROM python:3.12-slim

# catboost требует libgomp для OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

COPY src/ ./src/
COPY pipeline.py .
COPY models/ ./models/

ENV PYTHONHASHSEED=42
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Контракт хакатона
ENV INPUT_PATH=/data/input/holdout.csv
ENV OUTPUT_PATH=/data/output/predictions.csv
ENV MODEL_PATH=/app/models/catboost_t2.cbm

# Entry point
CMD ["python3", "pipeline.py"]