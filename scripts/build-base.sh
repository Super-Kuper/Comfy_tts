#!/usr/bin/env bash
# Собрать и опубликовать предсобранный base-образ воркера.
#
#   IMAGE=crappytackle/tts-serverless-base:0.3.4 scripts/build-base.sh
#
# Запускать один раз (или при обновлении зависимостей), не при каждом релизе.
set -euo pipefail

IMAGE="${IMAGE:-crappytackle/tts-serverless-base:0.3.4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build -f "${ROOT}/docker/Dockerfile.base" -t "${IMAGE}" "${ROOT}"
docker push "${IMAGE}"
