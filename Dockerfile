# =============================================================================
# TTS serverless worker — RunPod serverless (contract tts-serverless/v1).
#
# Model stack: Qwen3-TTS 1.7B Base (Apache-2.0) + GigaAM-v3 (MIT) aligner.
# Weights are NOT in the image: they live on the network volume at
# /runpod-volume/models and are verified by populate-volume.sh (pinned
# URL/revision/sha256 in models-manifest.txt).
#
# Structural build without CUDA/GPU (checks the package/handler wiring; the
# engine callable intentionally raises `engine_unavailable` until TODO(live) is
# wired, so this image is NOT meant to synthesize audio):
#   docker build --build-arg BASE_IMAGE=python:3.12-slim -t tts-serverless:stub .
#   docker run --rm --entrypoint python tts-serverless:stub -c "import tts_worker"
#
# TODO(live): re-pin BASE_IMAGE to a CUDA runtime image (e.g. nvidia/cuda
# 12.8 runtime by digest) and install pinned torch + the Qwen3-TTS and GigaAM
# runtime packages; warm-load both models in an entrypoint before the handler
# (cold start 3–10 min on the stand, amortized by the phrase batch — plan §7.2).
#
# Boundaries: no secrets/tokens/weights in the image; ComfyUI is not used here;
# only the RunPod handler is exposed.
# =============================================================================

ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE}

ARG TTS_WORKER_VERSION=0.1.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TERM=xterm \
    VOLUME_ROOT=/runpod-volume \
    MODELS_ROOT=/runpod-volume/models \
    MODELS_MANIFEST=/opt/tts-worker/models-manifest.txt \
    TTS_DICTIONARIES=/opt/tts-worker/data/normalization.ru.json \
    TTS_DEVICE=cuda \
    PYTHONPATH=/opt/tts-worker/src

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Worker Python dependencies. TODO(live): add pinned torch/CUDA and the
# Qwen3-TTS / GigaAM runtime packages to the real (GPU) image; keep versions
# pinned and record them in README/release notes.
RUN set -eux; \
    py="$(command -v python3 || command -v python)"; \
    "${py}" -m pip install --no-cache-dir "runpod==2.2.1" "httpx>=0.27,<0.29" "jsonschema>=4.21" "boto3>=1.35"; \
    "${py}" -c "import boto3, httpx, jsonschema, runpod; print('worker deps ok')"

WORKDIR /opt/tts-worker

COPY src/ /opt/tts-worker/src/
COPY handler.py /opt/tts-worker/handler.py
COPY contract/ /opt/tts-worker/contract/
COPY data/ /opt/tts-worker/data/
COPY models-manifest.txt /opt/tts-worker/models-manifest.txt
COPY populate-volume.sh /opt/tts-worker/populate-volume.sh

RUN chmod 0755 /opt/tts-worker/populate-volume.sh

# TODO(live): entrypoint that (1) downloads/verifies missing pinned weights,
# (2) warm-loads Qwen3-TTS + GigaAM (или валидирует том), (3) starts this
# handler. Until then the handler starts and returns a visible
# `engine_unavailable` for real synthesis jobs — no silent fallback.
ENTRYPOINT ["python", "/opt/tts-worker/handler.py"]
