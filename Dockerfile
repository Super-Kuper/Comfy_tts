# =============================================================================
# TTS serverless worker — RunPod serverless (contract tts-serverless/v1.2).
#
# Base image is PREBUILT and published separately:
#   crappytackle/tts-serverless-base:0.3.4
#   (docker/Dockerfile.base, scripts/build-base.sh)
#
# Why: RunPod Hub GitHub builds have a 30-minute `docker build` limit; installing
# CUDA/PyTorch/qwen-tts on every release does not fit reliably. The base carries
# torch/CUDA + pinned worker/model dependencies; this Dockerfile only adds the
# worker code (a few hundred KB), so the Hub build layer is fast. If the base
# changes, rebuild/push it first and bump BASE_IMAGE.
#
# Boundaries:
#   - no secrets/tokens/weights in the image: S3_* come from endpoint env, the
#     pinned Qwen3-TTS/GigaAM weights live on the network volume;
#   - the handler is the only exposed process; no EXPOSE on purpose.
# =============================================================================
ARG BASE_IMAGE=crappytackle/tts-serverless-base:0.3.4

FROM ${BASE_IMAGE}

ARG TTS_WORKER_VERSION=0.4.0

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

WORKDIR /opt/tts-worker

COPY src/ /opt/tts-worker/src/
COPY handler.py /opt/tts-worker/handler.py
COPY contract/ /opt/tts-worker/contract/
COPY data/ /opt/tts-worker/data/
COPY models-manifest.txt /opt/tts-worker/models-manifest.txt
COPY populate-volume.sh /opt/tts-worker/populate-volume.sh
COPY docker/entrypoint.sh /opt/tts-worker/entrypoint.sh

RUN chmod 0755 /opt/tts-worker/populate-volume.sh /opt/tts-worker/entrypoint.sh

# Structural check: package and handler import with the same interpreter the
# base image uses for the handler (the base already installs all runtime deps).
RUN python -c "import tts_worker.handler; print('tts_worker.handler import ok')"

ENTRYPOINT ["/opt/tts-worker/entrypoint.sh"]
