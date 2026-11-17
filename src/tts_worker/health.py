"""Health-ответ воркера (`contract/worker-contract.md` §3).

Проверка не грузит модели и не запускает синтез: только наличие тома
``VOLUME_ROOT`` и pinned-файлов engine/aligner на нём (наличие, без хеширования).

Статусы (документированы в контракте):
- ``status=ok`` — том смонтирован и оба pinned-файла (engine/aligner) на месте;
- ``starting`` — том смонтирован, но файлы ещё не докачаны (populate-volume:
  нормальное transient-состояние холодного старта);
- ``degraded`` — тома нет.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .errors import WorkerError
from .manifest import ModelManifest


def _role_status(config: Config, manifest: ModelManifest, role: str) -> str:
    """``ready``, если pinned-файл роли есть на томе; иначе ``missing``."""

    try:
        entry = manifest.entry(role)
    except WorkerError:
        return "missing"
    return "ready" if (config.models_root / entry.relative_path).is_file() else "missing"


def health_payload(config: Config, manifest: ModelManifest) -> dict[str, Any]:
    """Собрать ``{status, volume_mounted, engine, aligner}`` без синтеза."""

    volume_mounted = Path(config.volume_root).is_dir()
    engine = _role_status(config, manifest, "engine")
    aligner = _role_status(config, manifest, "aligner")

    if not volume_mounted:
        status = "degraded"
    elif engine == "ready" and aligner == "ready":
        status = "ok"
    else:
        status = "starting"

    return {
        "status": status,
        "volume_mounted": volume_mounted,
        "engine": engine,
        "aligner": aligner,
    }


__all__ = ["health_payload"]
