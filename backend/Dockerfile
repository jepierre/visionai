FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt

RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
    && python3.10 -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY images /app/images
COPY .env.example /app/.env.example

RUN mkdir -p /app/output/annotated /app/output/thumbnails /models/huggingface

EXPOSE 8000
CMD ["python3.10", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
