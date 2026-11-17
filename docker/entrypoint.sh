#!/usr/bin/env bash
# =============================================================================
# Entrypoint serverless-воркера TTS (контракт tts-serverless/v1).
#
# Порядок работы:
#   1. проверить том и при VOLUME_AUTOPOPULATE=true (по умолчанию) докачать
#      отсутствующие pinned-веса Qwen3-TTS/GigaAM через populate-volume.sh
#      (URL/sha256 из models-manifest.txt; несовпадение хеша — видимая ошибка,
#      а не тихая подмена);
#   2. запустить handler (runpod SDK слушает очередь endpoint'а).
#
# Ошибка автоскачивания не валит контейнер: health остаётся доступным, а job'ы
# на отсутствующих весах падают с видимым `model_file_missing` — лучше, чем
# «тихо» работать на других файлах. Секретов здесь нет: S3_* приходят только
# из env endpoint'а.
# =============================================================================

set -euo pipefail

TTS_WORKER_DIR="${TTS_WORKER_DIR:-/opt/tts-worker}"
VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
MODELS_ROOT="${MODELS_ROOT:-${VOLUME_ROOT}/models}"
MANIFEST="${MODELS_MANIFEST:-${TTS_WORKER_DIR}/models-manifest.txt}"
VOLUME_AUTOPOPULATE="${VOLUME_AUTOPOPULATE:-true}"
HF_TOKEN="${HF_TOKEN:-}"
TTS_PYTHON="${TTS_PYTHON:-python}"

log() { printf '[entrypoint] %s\n' "$*"; }
warn() { printf '[entrypoint] WARN: %s\n' "$*" >&2; }

_populate_volume() {
    if [[ "${VOLUME_AUTOPOPULATE}" != "true" ]]; then
        log "VOLUME_AUTOPOPULATE=${VOLUME_AUTOPOPULATE}: автоскачивание моделей выключено"
        return 0
    fi
    if [[ ! -d "${VOLUME_ROOT}" ]]; then
        warn "том ${VOLUME_ROOT} не смонтирован — автоскачивание моделей пропущено"
        return 0
    fi
    local script="${TTS_WORKER_DIR}/populate-volume.sh"
    if [[ ! -x "${script}" ]]; then
        warn "нет исполняемого ${script}: автоскачивание моделей пропущено"
        return 0
    fi
    log "автоскачивание моделей: ${script} (манифест ${MANIFEST})"
    if ! MODELS_ROOT="${MODELS_ROOT}" MODELS_MANIFEST="${MANIFEST}" HF_TOKEN="${HF_TOKEN}" "${script}"; then
        warn "автоскачивание моделей завершилось с ошибкой: job'ы на отсутствующих моделях упадут видимо"
    fi
}

main() {
    log "tts-serverless worker (Qwen3-TTS + GigaAM; VOLUME_ROOT=${VOLUME_ROOT}, MODELS_ROOT=${MODELS_ROOT})"
    _populate_volume
    export MODELS_ROOT MODELS_MANIFEST="${MANIFEST}" VOLUME_ROOT
    log "handler: ${TTS_PYTHON} ${TTS_WORKER_DIR}/handler.py"
    exec "${TTS_PYTHON}" "${TTS_WORKER_DIR}/handler.py"
}

main "$@"
