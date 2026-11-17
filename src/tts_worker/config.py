"""Настройки TTS-воркера из env endpoint'а.

Секретов в образе нет: S3-ключи приходят только из env. Невалидное значение
даёт ``config_invalid`` с именем переменной (contract §5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import WorkerError

# Локальный клон и образ (/opt/tts-worker) держат contract/ и data/ рядом с src/.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _number(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise WorkerError("config_invalid", f"{name}={raw!r}: ожидалось число") from exc
    if value <= 0:
        raise WorkerError("config_invalid", f"{name}={raw!r}: ожидалось > 0")
    return value


def _integer(source: Mapping[str, str], name: str, default: int) -> int:
    return int(_number(source, name, float(default)))


def _boolean(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw = source.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise WorkerError("config_invalid", f"{name}={raw!r}: ожидалось true/false")


@dataclass(frozen=True)
class Config:
    """Настройки одного endpoint'а (значения по умолчанию — как в contract §5)."""

    volume_root: Path = Path("/runpod-volume")
    models_root: Path = Path("/runpod-volume/models")
    manifest_path: Path = Path("/opt/tts-worker/models-manifest.txt")
    contract_dir: Path = _REPO_ROOT / "contract"
    dictionaries_path: Path = _REPO_ROOT / "data" / "normalization.ru.json"
    device: str = "cuda"
    job_timeout_seconds: float = 120.0
    max_text_chars: int = 1000
    output_inline_max_mb: float = 14.0
    output_inline_budget_mb: float = 20.0
    output_url_ttl_seconds: int = 86400
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_secure: bool = True

    @property
    def storage_configured(self) -> bool:
        return bool(self.s3_bucket and self.s3_access_key and self.s3_secret_key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        source = os.environ if env is None else env
        contract_dir = Path(source.get("CONTRACT_DIR", str(_REPO_ROOT / "contract")))
        if not (contract_dir / "input.schema.json").is_file():
            raise WorkerError(
                "config_invalid",
                f"CONTRACT_DIR={contract_dir}: нет input.schema.json",
            )
        volume_root = Path(source.get("VOLUME_ROOT", "/runpod-volume"))
        return cls(
            volume_root=volume_root,
            models_root=Path(source.get("MODELS_ROOT", str(volume_root / "models"))),
            manifest_path=Path(
                source.get("MODELS_MANIFEST", "/opt/tts-worker/models-manifest.txt")
            ),
            contract_dir=contract_dir,
            dictionaries_path=Path(
                source.get(
                    "TTS_DICTIONARIES",
                    str(_REPO_ROOT / "data" / "normalization.ru.json"),
                )
            ),
            device=str(source.get("TTS_DEVICE", "cuda")).strip() or "cuda",
            job_timeout_seconds=_number(source, "JOB_TIMEOUT_SECONDS", 120.0),
            max_text_chars=_integer(source, "TTS_MAX_TEXT_CHARS", 1000),
            output_inline_max_mb=_number(source, "OUTPUT_INLINE_MAX_MB", 14.0),
            output_inline_budget_mb=_number(source, "OUTPUT_INLINE_BUDGET_MB", 20.0),
            output_url_ttl_seconds=_integer(source, "OUTPUT_URL_TTL_SECONDS", 86400),
            s3_endpoint=source.get("S3_ENDPOINT") or None,
            s3_access_key=source.get("S3_ACCESS_KEY") or None,
            s3_secret_key=source.get("S3_SECRET_KEY") or None,
            s3_bucket=source.get("S3_BUCKET") or source.get("OUTPUT_BUCKET") or None,
            s3_region=str(source.get("S3_REGION", "us-east-1")),
            s3_secure=_boolean(source, "S3_SECURE", True),
        )


__all__ = ["Config"]
