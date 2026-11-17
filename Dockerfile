# =============================================================================
# TTS serverless worker — RunPod serverless (contract tts-serverless/v1).
#
# Model stack: Qwen3-TTS 1.7B Base (Apache-2.0) voice cloning + GigaAM-v3 (MIT)
# aligner. Weights are NOT in the image: they live on the network volume at
# /runpod-volume/models and are verified by populate-volume.sh (pinned
# URL/revision/sha256 in models-manifest.txt).
#
# Real (GPU) image: pinned CUDA/PyTorch runtime base + pinned model runtime
# (qwen-tts, transformers, soundfile, librosa). Torch comes from the base image
# and is never reinstalled here (qwen-tts is installed with --no-deps and its
# runtime deps are pinned explicitly, so PyPI does not replace the CUDA torch).
#
# Structural build without CUDA/GPU (checks the package/handler wiring; the
# model runtime is skipped because the base has no torch, and the engine
# callable then fails visibly with `engine_unavailable`, never a silent stub):
#   docker build --build-arg BASE_IMAGE=python:3.12-slim -t tts-serverless:stub .
#   docker run --rm --entrypoint python tts-serverless:stub -c "import tts_worker, tts_worker.handler"
#
# Boundaries: no secrets/tokens/weights in the image; only the RunPod handler is
# exposed.
# =============================================================================

# Pinned CUDA/PyTorch runtime (CUDA 12.8, cuDNN 9). The tag is published on
# Docker Hub; host CUDA must be >= 12.8 (Hub listing allows 13.2/13.0).
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE}

ARG TTS_WORKER_VERSION=0.3.0
# auto: install the Qwen3-TTS runtime only when the base image already carries
# torch (the real CUDA image). 1/0 force it. The python:3.12-slim stub keeps
# `auto` -> 0, so the structural build stays fast and GPU-free.
ARG INSTALL_MODEL_RUNTIME=auto

# VOLUME_AUTOPOPULATE=true — автоскачивание pinned-весов Qwen3-TTS/GigaAM на
# том при старте (docker/entrypoint.sh); false — только ручной populate.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TERM=xterm \
    TTS_WORKER_DIR=/opt/tts-worker \
    VOLUME_ROOT=/runpod-volume \
    MODELS_ROOT=/runpod-volume/models \
    MODELS_MANIFEST=/opt/tts-worker/models-manifest.txt \
    TTS_DICTIONARIES=/opt/tts-worker/data/normalization.ru.json \
    TTS_DEVICE=cuda \
    VOLUME_AUTOPOPULATE=true \
    PYTHONPATH=/opt/tts-worker/src

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Worker Python dependencies. The pinned model runtime (qwen-tts 0.1.1 +
# transformers 4.57.3 + soundfile 0.13.1 + librosa 0.11.0, Qwen3-TTS native
# 24 kHz) is installed only on a base that already has torch — the CUDA image.
# `--no-deps` for qwen-tts/torchaudio keeps the base CUDA torch build intact.
RUN set -eux; \
    py="$(command -v python3 || command -v python)"; \
    runtime="${INSTALL_MODEL_RUNTIME}"; \
    if [ "${runtime}" = "auto" ]; then \
        if "${py}" -c "import torch" >/dev/null 2>&1; then runtime=1; else runtime=0; fi; \
    fi; \
    "${py}" -m pip install --no-cache-dir \
        "runpod==2.2.1" "httpx>=0.27,<0.29" "jsonschema>=4.21" "boto3>=1.35"; \
    if [ "${runtime}" = "1" ]; then \
        "${py}" -m pip install --no-cache-dir \
            "transformers==4.57.3" "accelerate==1.12.0" "einops>=0.8,<0.9" \
            "numpy>=1.26,<3" "soundfile==0.13.1" "librosa==0.11.0"; \
        "${py}" -m pip install --no-cache-dir --no-deps "qwen-tts==0.1.1"; \
        "${py}" -m pip install --no-cache-dir --no-deps "torchaudio==2.7.1"; \
        "${py}" -c "import librosa, soundfile, transformers; from qwen_tts import Qwen3TTSModel; print('model runtime ok')"; \
    else \
        echo "stub base (no torch): model runtime deps skipped (visible engine_unavailable)"; \
    fi; \
    "${py}" -c "import boto3, httpx, jsonschema, runpod; print('worker deps ok')"

WORKDIR /opt/tts-worker

COPY src/ /opt/tts-worker/src/
COPY handler.py /opt/tts-worker/handler.py
COPY contract/ /opt/tts-worker/contract/
COPY data/ /opt/tts-worker/data/
COPY models-manifest.txt /opt/tts-worker/models-manifest.txt
COPY populate-volume.sh /opt/tts-worker/populate-volume.sh
COPY docker/entrypoint.sh /opt/tts-worker/entrypoint.sh

RUN chmod 0755 /opt/tts-worker/populate-volume.sh /opt/tts-worker/entrypoint.sh

# Structural smoke: imports must not require GPU, torch, network or runpod.
RUN set -eux; \
    py="$(command -v python3 || command -v python)"; \
    PYTHONPATH=/opt/tts-worker/src "${py}" -c \
        "import tts_worker, tts_worker.engine, tts_worker.handler, tts_worker.reference; print('worker wiring ok')"

# Entrypoint: авто-популяция тома (VOLUME_AUTOPOPULATE=true) → handler.
# Модель загружается лениво при первом job (warm-load не обязателен): холодный
# старт 3–10 мин амортизируется батчем фраз. До загрузки весов job падает
# видимым `model_file_missing`/`engine_unavailable`, а не тихим фолбэком.
ENTRYPOINT ["/opt/tts-worker/entrypoint.sh"]
