# Full-stack image: React UI + FastAPI/PyTorch (same origin = no CORS issues)
# Build context = repository root
# Render: Dockerfile path = Dockerfile, Root Directory = (blank)

FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Same-origin API when UI is served by FastAPI
RUN printf '%s\n' '{"apiBaseUrl":""}' > public/config.json
RUN npm run build

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

COPY backend/requirements.txt .
RUN pip install --no-cache-dir $(grep -vE '^(torch|torchvision)' requirements.txt) \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.6.0" "torchvision==0.21.0"

COPY backend/app ./app
COPY backend/models ./models
COPY --from=frontend /fe/dist ./static

RUN mkdir -p data/uploads data/heatmaps

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
