# =============================================================
#  AutoLyrics — Dockerfile
#  Multi-stage build supporting CPU and CUDA runtimes.
# =============================================================
ARG BASE=python:3.11-slim-bookworm

FROM ${BASE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY . .
RUN pip install --no-cache-dir -e .

# ---- Default: run the API ----
EXPOSE 8000 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
