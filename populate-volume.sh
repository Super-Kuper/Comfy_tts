#!/usr/bin/env bash
# Download and verify the pinned model files of models-manifest.txt onto the
# network volume (or any local MODELS_ROOT). Idempotent: a file whose sha256
# already matches the manifest is skipped.
#
# Usage:
#   MODELS_ROOT=/runpod-volume/models ./populate-volume.sh
#   MODELS_ROOT=./.local/models MODEL_MANIFEST=./models-manifest.txt ./populate-volume.sh
#
# Runs at boot only when VOLUME_AUTOPOPULATE=true in the image entrypoint, and can
# be run from a CPU pod by the service's runpod.provision (contract §4/§6).
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/runpod-volume/models}"
MANIFEST="${MODELS_MANIFEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models-manifest.txt}"
PIN_RE='/resolve/[0-9a-f]{40}/'
SHA_RE='^[0-9a-f]{64}$'

[ -r "${MANIFEST}" ] || { echo "populate-volume: манифест не читается: ${MANIFEST}" >&2; exit 1; }
mkdir -p "${MODELS_ROOT}"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

downloaded=0
skipped=0
while IFS= read -r line || [ -n "${line}" ]; do
    case "${line}" in
        ''|'#'*) continue ;;
    esac
    IFS=$'\t' read -r path url sha256 version license role <<< "${line}"
    if [ -z "${path:-}" ] || [ -z "${url:-}" ] || [ -z "${sha256:-}" ] || [ -z "${role:-}" ]; then
        echo "populate-volume: плохая строка манифеста: ${line}" >&2
        exit 1
    fi
    case "${path}" in
        /*|*..*) echo "populate-volume: путь должен быть относительным без ..: ${path}" >&2; exit 1 ;;
    esac
    if [[ ! "${url}" =~ ${PIN_RE} ]]; then
        echo "populate-volume: ${path}: URL не запинен на commit: ${url}" >&2
        exit 1
    fi
    if [[ ! "${sha256}" =~ ${SHA_RE} ]]; then
        echo "populate-volume: ${path}: sha256 не 64 hex" >&2
        exit 1
    fi
    target="${MODELS_ROOT}/${path}"
    if [ -f "${target}" ] && [ "$(sha256_of "${target}")" = "${sha256}" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    mkdir -p "${target%/*}"
    tmp="$(mktemp "${target}.part.XXXXXX")"
    echo "populate-volume: ${path} ← ${url}"
    if ! curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 -o "${tmp}" "${url}"; then
        rm -f "${tmp}"
        echo "populate-volume: скачивание не удалось: ${path}" >&2
        exit 1
    fi
    actual="$(sha256_of "${tmp}")"
    if [ "${actual}" != "${sha256}" ]; then
        rm -f "${tmp}"
        echo "populate-volume: provision_hash_mismatch ${path}: ${actual} != ${sha256}" >&2
        exit 1
    fi
    mv -f "${tmp}" "${target}"
    downloaded=$((downloaded + 1))
done < "${MANIFEST}"

echo "populate-volume: готово, скачано=${downloaded}, уже было=${skipped}, root=${MODELS_ROOT}"
