FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only Torch (smaller RAM) — better fit for Render Free
RUN pip install --no-cache-dir $(grep -vE '^(torch|torchvision)' requirements.txt) \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.6.0" "torchvision==0.21.0"

COPY app ./app
COPY models ./models
COPY static ./static

RUN mkdir -p data/uploads data/heatmaps

EXPOSE 8000
# Render Free sets $PORT; single worker to stay under memory limits
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
