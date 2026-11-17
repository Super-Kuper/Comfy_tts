"""Доставка аудио дубля: inline base64 или presigned URL (contract tts-serverless/v1 §2)."""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any, Protocol

from .audio import AudioInfo
from .config import Config
from .errors import WorkerError

_MB = 1024 * 1024


class AudioStorage(Protocol):
    """Внешнее S3-совместимое хранилище больших результатов."""

    async def upload(self, data: bytes, content_type: str) -> str:
        """Загрузить файл и вернуть presigned GET URL."""


def _content_type(info: AudioInfo) -> str:
    return "audio/wav" if info.codec == "wav" else "audio/flac"


def fits_inline(data: bytes, config: Config) -> bool:
    """Помещается ли результат в base64-лимиты ответа."""

    encoded_size = (len(data) + 2) // 3 * 4
    return (
        len(data) <= config.output_inline_max_mb * _MB
        and encoded_size <= config.output_inline_budget_mb * _MB
    )


async def deliver_audio(
    audio: bytes,
    info: AudioInfo,
    config: Config,
    storage: AudioStorage | None,
) -> dict[str, Any]:
    """Вернуть ровно один ключ контракта: ``audio_b64`` либо ``audio_url``."""

    if fits_inline(audio, config):
        return {"audio_b64": base64.b64encode(audio).decode("ascii")}
    if storage is None:
        raise WorkerError(
            "output_storage_missing",
            f"аудио {len(audio)} байт не влезает в inline-лимиты, а S3_* не настроены",
        )
    url = await storage.upload(audio, _content_type(info))
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise WorkerError("output_storage_failed", f"хранилище вернуло не URL: {url!r}")
    return {"audio_url": url}


class S3Storage:
    """S3-совместимое хранилище (boto3 импортируется лениво)."""

    def __init__(self, config: Config) -> None:
        self._config = config

    async def upload(self, data: bytes, content_type: str) -> str:
        return await asyncio.to_thread(self._upload_sync, data, content_type)

    def _upload_sync(self, data: bytes, content_type: str) -> str:
        config = self._config
        try:
            import boto3  # локальный импорт: тесты и handler без S3 не требуют boto3
        except ImportError as exc:  # pragma: no cover - образ ставит boto3
            raise WorkerError("output_storage_failed", "boto3 не установлен в образе") from exc
        try:
            client = boto3.client(
                "s3",
                endpoint_url=config.s3_endpoint,
                aws_access_key_id=config.s3_access_key,
                aws_secret_access_key=config.s3_secret_key,
                region_name=config.s3_region,
                use_ssl=config.s3_secure,
            )
            extension = ".wav" if content_type == "audio/wav" else ".flac"
            key = f"audio/{uuid.uuid4().hex}{extension}"
            client.put_object(
                Bucket=config.s3_bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": config.s3_bucket, "Key": key},
                ExpiresIn=config.output_url_ttl_seconds,
            )
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - стабильный код контракта
            raise WorkerError("output_storage_failed", f"S3 upload: {exc}") from exc


__all__ = ["AudioStorage", "S3Storage", "deliver_audio", "fits_inline"]
