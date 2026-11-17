"""Референс голоса для клонирования: скачивание и декодирование.

Контракт ``tts-serverless/v1`` (аддитивно, worker 0.3.0): вход
``voice.mode="clone"`` несёт ``reference {url, sha256, text, language}`` —
presigned HTTP(S) URL референса, его sha256 (он же ``media_items.content_hash``
в сервисе) и точный текст записи. Воркер скачивает файл (httpx, таймаут,
ограничение 25 МБ), сверяет sha256, декодирует WAV/FLAC/MP3/OGG
(soundfile/librosa) в PCM 24 кГц mono и отдаёт его движку Qwen3-TTS Base.

Ограничения из плана (``plan-clone.md`` §4): размер ≤ 25 МБ, длительность
1–15 с (продуктовый ориентир 3–10 с задаёт приложение), текст 1–500 символов.
Приватность: URL и содержимое референса не логируются; в ошибки и телеметрию
попадает только sha256. Любой сбой — видимый код (``reference_missing``,
``reference_invalid``, ``reference_hash``, ``reference_too_large``); тихого
перехода на пресет нет.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .errors import WorkerError

#: Лимиты контракта §1.4 (те же числа, что в приложении: «≤25 МБ, 1–15 с»).
MAX_REFERENCE_BYTES = 25 * 1024 * 1024
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 15.0
MAX_REFERENCE_TEXT_CHARS = 500
#: Qwen3-TTS Base работает на 24 кГц; референс ресемплится здесь.
REFERENCE_SAMPLE_RATE = 24000
DOWNLOAD_TIMEOUT_SECONDS = 30.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class DecodedAudio:
    """Факт декодированного файла: PCM mono float32 и фактическая частота."""

    pcm: Any
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class ReferenceAudio:
    """Готовый референс для движка: PCM + проверенный sha256 + точный текст."""

    pcm: Any
    sample_rate: int
    text: str
    language: str
    sha256: str
    duration_seconds: float


class ReferenceDownloader(Protocol):
    """Скачать байты референса по presigned URL (инъекция для тестов)."""

    async def download(self, url: str) -> bytes:
        ...


class AudioDecoder(Protocol):
    """Декодировать аудиофайл в PCM mono float32 (инъекция для тестов)."""

    def decode(self, data: bytes) -> DecodedAudio:
        ...


def sha256_bytes(data: bytes) -> str:
    """sha256 в нижнем hex — та же форма, что ``reference.sha256``."""

    return hashlib.sha256(data).hexdigest()


def check_reference_url(url: Any) -> str:
    """Проверить URL референса: только http(s), без записей в лог."""

    if not isinstance(url, str) or not url.strip():
        raise WorkerError(
            "reference_missing",
            "voice.reference.url отсутствует: нужен presigned HTTP(S)-URL референса",
        )
    value = url.strip()
    if _HTTP_RE.match(value) is None:
        raise WorkerError(
            "reference_invalid",
            "voice.reference.url: ожидался HTTP(S)-URL, получена другая схема",
        )
    return value


def check_reference_sha256(value: Any) -> str:
    """Проверить ожидаемый sha256: ровно 64 hex (регистр нормализуется)."""

    raw = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(raw) is None:
        raise WorkerError(
            "reference_invalid",
            "voice.reference.sha256: ожидалось 64 hex-символа sha256",
        )
    return raw


def check_reference_duration(duration_seconds: float) -> float:
    """Проверить длительность референса: 1–15 с, иначе видимый ``reference_invalid``."""

    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError) as exc:
        raise WorkerError("reference_invalid", f"длительность референса: {duration_seconds!r}") from exc
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        raise WorkerError(
            "reference_invalid",
            f"длительность референса {duration:.3f} с вне "
            f"[{MIN_REFERENCE_SECONDS:g}, {MAX_REFERENCE_SECONDS:g}] с",
        )
    return duration


def check_reference_text(value: Any) -> str:
    """Точный текст референса: 1–500 символов, не пустой после strip."""

    if not isinstance(value, str) or not value.strip():
        raise WorkerError(
            "reference_missing",
            "voice.reference.text отсутствует: нужен точный текст записи референса",
        )
    if len(value) > MAX_REFERENCE_TEXT_CHARS:
        raise WorkerError(
            "reference_invalid",
            f"voice.reference.text: {len(value)} > {MAX_REFERENCE_TEXT_CHARS} символов",
        )
    return value


class HttpxReferenceDownloader:
    """Скачивание референса через httpx с таймаутом и лимитом размера.

    ``transport`` — только для тестов (``httpx.MockTransport``): в образе
    никакой сети, кроме presigned URL сервиса, воркер не открывает.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DOWNLOAD_TIMEOUT_SECONDS,
        transport: Any | None = None,
    ) -> None:
        self._timeout_seconds = float(timeout_seconds)
        self._transport = transport

    async def download(self, url: str) -> bytes:
        check_reference_url(url)
        try:
            import httpx  # ленивый импорт: модуль и тесты не требуют httpx
        except ImportError as exc:  # pragma: no cover - зависимость образа
            raise WorkerError("config_invalid", "httpx не установлен в образе воркера") from exc

        kwargs: dict[str, Any] = {
            "timeout": self._timeout_seconds,
            "follow_redirects": True,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        try:
            chunks: list[bytes] = []
            total = 0
            async with httpx.AsyncClient(**kwargs) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise WorkerError(
                            "reference_invalid",
                            f"скачивание референса: HTTP {response.status_code}",
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except ValueError:
                            declared_bytes = 0
                        if declared_bytes > MAX_REFERENCE_BYTES:
                            raise WorkerError(
                                "reference_too_large",
                                f"референс {declared_bytes} байт > {MAX_REFERENCE_BYTES}",
                            )
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_REFERENCE_BYTES:
                            raise WorkerError(
                                "reference_too_large",
                                f"референс > {MAX_REFERENCE_BYTES} байт",
                            )
                        chunks.append(chunk)
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - стабильный код контракта, URL не логируем
            raise WorkerError(
                "reference_invalid",
                f"скачивание референса не удалось: {type(exc).__name__}",
            ) from exc
        data = b"".join(chunks)
        if not data:
            raise WorkerError("reference_invalid", "скачанный референс пуст (0 байт)")
        return data


class SoundFileDecoder:
    """Декодирование WAV/FLAC/MP3/OGG через soundfile + ресемплинг librosa.

    soundfile/librosa импортируются лениво: тесты и stub-образ не требуют
    аудио-стека. Отсутствие пакетов — видимый ``engine_unavailable``, а не
    ImportError внутри job'а.
    """

    def decode(self, data: bytes) -> DecodedAudio:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - зависимость образа
            raise WorkerError(
                "engine_unavailable",
                f"soundfile/numpy не установлены в образе воркера: {exc}",
            ) from exc
        if not data:
            raise WorkerError("reference_invalid", "референс пуст (0 байт)")
        try:
            pcm, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001 - любой сбой декодера = reference_invalid
            raise WorkerError(
                "reference_invalid",
                f"референс не декодируется как WAV/FLAC/MP3/OGG: {exc}",
            ) from exc
        if not pcm.size:
            raise WorkerError("reference_invalid", "референс без аудиокадров (пустой звук)")
        if int(sample_rate) <= 0:
            raise WorkerError("reference_invalid", f"референс: sample_rate={sample_rate}")
        frames = int(pcm.shape[0])
        duration = frames / float(sample_rate)
        mono = pcm.mean(axis=1) if pcm.shape[1] > 1 else pcm[:, 0]
        if int(sample_rate) != REFERENCE_SAMPLE_RATE:
            try:
                import librosa
            except ImportError as exc:  # pragma: no cover - зависимость образа
                raise WorkerError(
                    "engine_unavailable",
                    f"librosa не установлена в образе воркера: {exc}",
                ) from exc
            mono = librosa.resample(
                mono,
                orig_sr=int(sample_rate),
                target_sr=REFERENCE_SAMPLE_RATE,
            )
            sample_rate = REFERENCE_SAMPLE_RATE
        return DecodedAudio(
            pcm=np.ascontiguousarray(mono, dtype=np.float32),
            sample_rate=int(sample_rate),
            duration_seconds=duration,
        )


async def prepare_reference(
    voice: Mapping[str, Any],
    *,
    downloader: ReferenceDownloader,
    decoder: AudioDecoder,
) -> ReferenceAudio:
    """Скачать, проверить и декодировать референс (contract §1.4 / plan §4).

    Порядок проверок: форма ``voice.reference`` → URL/sha256/текст →
    скачивание (≤25 МБ) → sha256 → декодирование → длительность 1–15 с.
    Возвращает PCM 24 кГц mono и проверенный sha256 для телеметрии.
    """

    reference = voice.get("reference") if isinstance(voice, Mapping) else None
    if not isinstance(reference, Mapping):
        raise WorkerError(
            "reference_missing",
            "voice.mode=clone требует voice.reference {url, sha256, text, language}",
        )
    url = check_reference_url(reference.get("url"))
    expected_sha256 = check_reference_sha256(reference.get("sha256"))
    text = check_reference_text(reference.get("text"))
    language = str(reference.get("language") or "ru").strip() or "ru"

    data = await downloader.download(url)
    if not isinstance(data, (bytes, bytearray)):
        raise WorkerError("reference_invalid", "загрузчик референса вернул не байты")
    data = bytes(data)
    if len(data) > MAX_REFERENCE_BYTES:
        # Дублирующая проверка на случай кастомного загрузчика: лимит — контракт.
        raise WorkerError(
            "reference_too_large",
            f"референс {len(data)} байт > {MAX_REFERENCE_BYTES}",
        )
    if not data:
        raise WorkerError("reference_invalid", "скачанный референс пуст (0 байт)")
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != expected_sha256:
        raise WorkerError(
            "reference_hash",
            f"sha256 референса {actual_sha256} != ожидаемый {expected_sha256}",
        )

    try:
        decoded = decoder.decode(data)
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001 - сбой декодера = видимый reference_invalid
        raise WorkerError("reference_invalid", f"референс не декодируется: {exc}") from exc
    duration = check_reference_duration(decoded.duration_seconds)
    return ReferenceAudio(
        pcm=decoded.pcm,
        sample_rate=int(decoded.sample_rate),
        text=text,
        language=language,
        sha256=actual_sha256,
        duration_seconds=duration,
    )


__all__ = [
    "DOWNLOAD_TIMEOUT_SECONDS",
    "MAX_REFERENCE_BYTES",
    "MAX_REFERENCE_SECONDS",
    "MAX_REFERENCE_TEXT_CHARS",
    "MIN_REFERENCE_SECONDS",
    "REFERENCE_SAMPLE_RATE",
    "AudioDecoder",
    "DecodedAudio",
    "HttpxReferenceDownloader",
    "ReferenceAudio",
    "ReferenceDownloader",
    "SoundFileDecoder",
    "check_reference_duration",
    "check_reference_sha256",
    "check_reference_text",
    "check_reference_url",
    "prepare_reference",
    "sha256_bytes",
]
