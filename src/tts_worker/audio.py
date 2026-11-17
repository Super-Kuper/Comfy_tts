"""Факты о произведённом аудиофайле: кодек, частота, каналы, длительность.

Размеры и длительность берутся из заголовка файла, а не из входного снимка
``parameter`` (contract/worker-contract.md §2.2). Контракт допускает WAV и FLAC:
ComfyUI умеет писать оба через ``SaveAudio``/``SaveAudioAdvanced``.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from .errors import WorkerError


@dataclass(frozen=True)
class AudioInfo:
    """Измеренные факты аудиофайла."""

    sample_rate: int
    channels: int
    duration_seconds: float
    codec: str


def read_audio_info(data: bytes) -> AudioInfo:
    """Прочитать факты WAV/FLAC; пустой/битый файл — ошибка, не успех."""

    if not data:
        raise WorkerError("output_empty", "аудиофайл пуст (0 байт)")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return _wav_info(data)
    if data[:4] == b"fLaC":
        return _flac_info(data)
    # TODO(live): mp3/opus требуют декодера (ffmpeg/mutagen) — не входит в
    # контракт v1; граф каталога должен писать wav (SaveAudioAdvanced format=wav).
    raise WorkerError(
        "output_invalid",
        "ожидался WAV/FLAC; mp3/opus пока не декодируются — граф должен писать wav",
    )


def _wav_info(data: bytes) -> AudioInfo:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.getnframes()
    except Exception as exc:  # noqa: BLE001 - любой сбой разбора = output_invalid
        raise WorkerError("output_invalid", f"WAV не читается: {exc}") from exc
    if sample_rate <= 0 or channels <= 0:
        raise WorkerError("output_invalid", f"WAV: sample_rate={sample_rate}, channels={channels}")
    if frames <= 0:
        raise WorkerError("output_empty", "WAV без кадров (пустой звук)")
    return AudioInfo(
        sample_rate=int(sample_rate),
        channels=int(channels),
        duration_seconds=frames / float(sample_rate),
        codec="wav",
    )


def _flac_info(data: bytes) -> AudioInfo:
    """STREAMINFO (первый метадата-блок FLAC): 20 бит частоты, 3 бита каналов."""
    position = 4
    while position + 4 <= len(data):
        header = data[position]
        is_last = bool(header & 0x80)
        block_type = header & 0x7F
        length = int.from_bytes(data[position + 1 : position + 4], "big")
        body_start = position + 4
        body_end = body_start + length
        if body_end > len(data):
            raise WorkerError("output_invalid", "FLAC: обрезанный метадата-блок")
        if block_type == 0:  # STREAMINFO
            body = data[body_start:body_end]
            if len(body) < 18:
                raise WorkerError("output_invalid", "FLAC: короткий STREAMINFO")
            # min/max blocksize (2+2), min/max framesize (3+3) = 10 байт, затем
            # 20 бит частоты + 3 бита (каналы-1) + 5 бит (битность-1) + 36 бит кадров.
            packed = int.from_bytes(body[10:18], "big")
            sample_rate = packed >> 44
            channels = ((packed >> 41) & 0x07) + 1
            total_samples = packed & ((1 << 36) - 1)
            if sample_rate == 0:
                raise WorkerError("output_invalid", "FLAC: sample_rate=0")
            if total_samples == 0:
                raise WorkerError("output_empty", "FLAC без кадров (пустой звук)")
            return AudioInfo(
                sample_rate=int(sample_rate),
                channels=int(channels),
                duration_seconds=total_samples / float(sample_rate),
                codec="flac",
            )
        if is_last:
            break
        position = body_end
    raise WorkerError("output_invalid", "FLAC: нет блока STREAMINFO")


__all__ = ["AudioInfo", "read_audio_info"]
