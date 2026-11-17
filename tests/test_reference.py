"""Референс голоса: скачивание, sha256/размер/длительность, декодирование.

Тесты без сети: httpx подменяется ``MockTransport``, декодер и загрузчик —
фейки. Живой ``SoundFileDecoder`` без soundfile/librosa даёт видимый
``engine_unavailable`` (проверяется в тестовом окружении).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from helpers import FakeReferenceDecoder, FakeReferenceDownloader, reference_bytes, reference_sha256

from tts_worker.errors import WorkerError
from tts_worker.reference import (
    MAX_REFERENCE_BYTES,
    HttpxReferenceDownloader,
    SoundFileDecoder,
    check_reference_duration,
    check_reference_sha256,
    prepare_reference,
    sha256_bytes,
)

REFERENCE_BYTES = reference_bytes(3.0)
REFERENCE_SHA = reference_sha256(REFERENCE_BYTES)
REFERENCE_URL = "https://files.example/voice/reference-001.wav?X-Amz-Signature=test"


def _voice(**reference_overrides: object) -> dict:
    reference = {
        "url": REFERENCE_URL,
        "sha256": REFERENCE_SHA,
        "text": "Проверка голоса для клонирования.",
        "language": "ru",
    }
    reference.update(reference_overrides)
    return {"mode": "clone", "reference": reference}


def _run(coro):
    return asyncio.run(coro)


def _prepare(voice: dict, *, downloader=None, decoder=None):
    downloader = downloader or FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    decoder = decoder or FakeReferenceDecoder()
    return _run(prepare_reference(voice, downloader=downloader, decoder=decoder))


# --- prepare_reference --------------------------------------------------------


def test_prepare_reference_happy_path() -> None:
    downloader = FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    decoder = FakeReferenceDecoder(duration_seconds=3.25)
    reference = _prepare(_voice(), downloader=downloader, decoder=decoder)
    assert downloader.urls == [REFERENCE_URL]
    assert decoder.calls == [REFERENCE_BYTES]
    assert reference.pcm is decoder.pcm
    assert reference.sample_rate == 24000
    assert reference.sha256 == REFERENCE_SHA
    assert reference.text == "Проверка голоса для клонирования."
    assert reference.language == "ru"
    assert reference.duration_seconds == pytest.approx(3.25)


def test_prepare_reference_normalizes_uppercase_sha_and_spaces() -> None:
    voice = _voice(sha256=" " + REFERENCE_SHA.upper() + " ")
    reference = _prepare(voice)
    assert reference.sha256 == REFERENCE_SHA


def test_missing_reference_object_is_reference_missing() -> None:
    downloader = FakeReferenceDownloader()
    with pytest.raises(WorkerError) as error:
        _run(prepare_reference({"mode": "clone"}, downloader=downloader, decoder=FakeReferenceDecoder()))
    assert error.value.code == "reference_missing"
    assert downloader.urls == []


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"url": ""}, "reference_missing"),
        ({"url": None}, "reference_missing"),
        ({"url": "ftp://files.example/ref.wav"}, "reference_invalid"),
        ({"sha256": "abc"}, "reference_invalid"),
        ({"sha256": "A" * 64}, "reference_hash"),  # uppercase: prepare нормализует, но файл не совпадёт
        ({"text": ""}, "reference_missing"),
        ({"text": "   "}, "reference_missing"),
        ({"text": 5}, "reference_missing"),
        ({"text": "x" * 501}, "reference_invalid"),
    ],
)
def test_bad_reference_parts_are_visible(overrides: dict, code: str) -> None:
    downloader = FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    decoder = FakeReferenceDecoder()
    with pytest.raises(WorkerError) as error:
        _prepare(_voice(**overrides), downloader=downloader, decoder=decoder)
    assert error.value.code == code


def test_hash_mismatch_is_reference_hash() -> None:
    with pytest.raises(WorkerError) as error:
        _prepare(_voice(sha256="b" * 64))
    assert error.value.code == "reference_hash"
    assert REFERENCE_SHA in error.value.message


def test_downloader_error_is_propagated() -> None:
    failure = WorkerError("reference_too_large", "референс > 25 МБ")
    with pytest.raises(WorkerError) as error:
        _prepare(_voice(), downloader=FakeReferenceDownloader(error=failure))
    assert error.value.code == "reference_too_large"


def test_decoder_worker_error_is_propagated() -> None:
    failure = WorkerError("reference_invalid", "не декодируется")
    with pytest.raises(WorkerError) as error:
        _prepare(_voice(), decoder=FakeReferenceDecoder(error=failure))
    assert error.value.code == "reference_invalid"


def test_decoder_generic_error_becomes_reference_invalid() -> None:
    with pytest.raises(WorkerError) as error:
        _prepare(_voice(), decoder=FakeReferenceDecoder(error=ValueError("битый файл")))
    assert error.value.code == "reference_invalid"


def test_size_guard_after_custom_downloader(monkeypatch) -> None:
    import tts_worker.reference as reference_module

    monkeypatch.setattr(reference_module, "MAX_REFERENCE_BYTES", len(REFERENCE_BYTES) - 1)
    with pytest.raises(WorkerError) as error:
        _prepare(_voice())
    assert error.value.code == "reference_too_large"


@pytest.mark.parametrize("duration", [1.0, 5.0, 15.0])
def test_allowed_reference_duration(duration: float) -> None:
    reference = _prepare(_voice(), decoder=FakeReferenceDecoder(duration_seconds=duration))
    assert reference.duration_seconds == pytest.approx(duration)


@pytest.mark.parametrize("duration", [0.0, 0.999, 15.001, 30.0])
def test_rejected_reference_duration(duration: float) -> None:
    with pytest.raises(WorkerError) as error:
        _prepare(_voice(), decoder=FakeReferenceDecoder(duration_seconds=duration))
    assert error.value.code == "reference_invalid"


def test_check_reference_duration_accepts_bounds() -> None:
    assert check_reference_duration(1.0) == pytest.approx(1.0)
    assert check_reference_duration(15.0) == pytest.approx(15.0)


def test_sha256_value() -> None:
    assert sha256_bytes(b"hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_choice_of_sha_is_64_hex() -> None:
    assert len(check_reference_sha256(REFERENCE_SHA)) == 64


# --- HttpxReferenceDownloader (MockTransport, без сети) ----------------------


def _downloader(handler, **kwargs) -> HttpxReferenceDownloader:
    return HttpxReferenceDownloader(transport=httpx.MockTransport(handler), **kwargs)


def test_httpx_downloader_returns_bytes() -> None:
    downloader = _downloader(lambda request: httpx.Response(200, content=REFERENCE_BYTES))
    assert _run(downloader.download(REFERENCE_URL)) == REFERENCE_BYTES


def test_httpx_downloader_http_error_is_reference_invalid() -> None:
    downloader = _downloader(lambda request: httpx.Response(403, content=b"denied"))
    with pytest.raises(WorkerError) as error:
        _run(downloader.download(REFERENCE_URL))
    assert error.value.code == "reference_invalid"
    assert "403" in error.value.message


def test_httpx_downloader_empty_body_is_reference_invalid() -> None:
    downloader = _downloader(lambda request: httpx.Response(200, content=b""))
    with pytest.raises(WorkerError) as error:
        _run(downloader.download(REFERENCE_URL))
    assert error.value.code == "reference_invalid"


def test_httpx_downloader_network_error_is_reference_invalid() -> None:
    def boom(request):
        raise httpx.ConnectTimeout("timeout")

    downloader = _downloader(boom)
    with pytest.raises(WorkerError) as error:
        _run(downloader.download(REFERENCE_URL))
    assert error.value.code == "reference_invalid"
    assert REFERENCE_URL not in error.value.message


def test_httpx_downloader_rejects_content_length_over_limit(monkeypatch) -> None:
    import tts_worker.reference as reference_module

    monkeypatch.setattr(reference_module, "MAX_REFERENCE_BYTES", 10)
    downloader = _downloader(lambda request: httpx.Response(200, content=b"x" * 11))
    with pytest.raises(WorkerError) as error:
        _run(downloader.download(REFERENCE_URL))
    assert error.value.code == "reference_too_large"


def test_httpx_downloader_rejects_non_http_url() -> None:
    downloader = _downloader(lambda request: httpx.Response(200, content=b"x"))
    with pytest.raises(WorkerError) as error:
        _run(downloader.download("ftp://files.example/ref.wav"))
    assert error.value.code == "reference_invalid"


# --- SoundFileDecoder без аудио-стека ---------------------------------------


def test_soundfile_decoder_without_runtime_is_engine_unavailable() -> None:
    # В GPU-free окружении soundfile/librosa не установлены: видимый код, а не
    # ImportError и не тихий пропуск референса.
    with pytest.raises(WorkerError) as error:
        SoundFileDecoder().decode(REFERENCE_BYTES)
    assert error.value.code == "engine_unavailable"
    assert MAX_REFERENCE_BYTES == 25 * 1024 * 1024
