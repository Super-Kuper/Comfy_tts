"""Факты WAV из заголовка: пустой и битый файл — ошибки, не успех."""

from __future__ import annotations

import pytest
from helpers import flac_bytes, wav_bytes

from tts_worker.audio import read_audio_info
from tts_worker.errors import WorkerError


def test_wav_facts() -> None:
    info = read_audio_info(wav_bytes(2.1, sample_rate=48000, channels=2))
    assert info.codec == "wav"
    assert (info.sample_rate, info.channels) == (48000, 2)
    assert info.duration_seconds == pytest.approx(2.1, abs=1e-3)


def test_flac_facts() -> None:
    info = read_audio_info(flac_bytes(1.0, sample_rate=24000, channels=1))
    assert info.codec == "flac"
    assert (info.sample_rate, info.channels) == (24000, 1)


def test_empty_audio_is_output_empty() -> None:
    with pytest.raises(WorkerError) as error:
        read_audio_info(b"")
    assert error.value.code == "output_empty"


def test_zero_frame_wav_is_output_empty() -> None:
    with pytest.raises(WorkerError) as error:
        read_audio_info(wav_bytes(0.0))
    assert error.value.code == "output_empty"


def test_broken_audio_is_output_invalid() -> None:
    with pytest.raises(WorkerError) as error:
        read_audio_info(b"not-an-audio-file")
    assert error.value.code == "output_invalid"
