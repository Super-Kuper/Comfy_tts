"""Handler: контракт, факты исполнения, пустой результат, «модель не выбирается»."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import sys

import pytest
from helpers import (
    DATA,
    REFERENCE_PCM,
    FakeAligner,
    FakeClock,
    FakeEngine,
    FakeNormalizer,
    FakeReferenceDecoder,
    FakeReferenceDownloader,
    FakeStorage,
    contract_dir,
    load_json,
    model_sha,
    reference_bytes,
    reference_sha256,
    wav_bytes,
)
from jsonschema import Draft202012Validator

from tts_worker.aligner import WordTiming
from tts_worker.engine import Qwen3TtsEngine, SynthesisTake
from tts_worker.errors import WorkerError
from tts_worker.handler import build_handler
from tts_worker.normalization import DictionaryNormalizer
from tts_worker.reference import prepare_reference

REQUEST_ID = "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001"


# --- Вспомогательное ---------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _job(**input_overrides) -> dict:
    payload = copy.deepcopy(load_json(contract_dir() / "examples" / "request.json")["input"])
    payload.update(input_overrides)
    return {"id": "job-1", "input": payload}


def _handler(
    setup,
    *,
    engine=None,
    aligner=None,
    normalizer=None,
    storage=None,
    clock=None,
    downloader=None,
    decoder=None,
):
    config, manifest, _models_root = setup
    engine = engine or FakeEngine()
    aligner = aligner or FakeAligner()
    normalizer = normalizer or FakeNormalizer()
    clock = clock or FakeClock([10.0, 11.7])
    handler = build_handler(
        config,
        manifest,
        engine,
        aligner,
        normalizer,
        storage=storage,
        clock=clock,
        reference_downloader=downloader,
        reference_decoder=decoder,
    )
    return handler, engine, aligner, normalizer


REFERENCE_URL = "https://files.example/voice/reference-001.wav?X-Amz-Signature=test"
REFERENCE_BYTES = reference_bytes(3.0)
REFERENCE_SHA = reference_sha256(REFERENCE_BYTES)


def _clone_job(**reference_overrides) -> dict:
    """Job с валидным clone-голосом; sha256 сходится с REFERENCE_BYTES."""

    reference = {
        "url": REFERENCE_URL,
        "sha256": REFERENCE_SHA,
        "text": "Проверка голоса для клонирования.",
        "language": "ru",
    }
    reference.update(reference_overrides)
    job = _job()
    job["input"]["voice"] = {"mode": "clone", "reference": reference}
    return job


def _clone_handler(setup, *, downloader=None, decoder=None, **kwargs):
    downloader = downloader or FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    decoder = decoder or FakeReferenceDecoder()
    return _handler(setup, downloader=downloader, decoder=decoder, **kwargs)


def _output_validator() -> Draft202012Validator:
    return Draft202012Validator(load_json(contract_dir() / "output.schema.json"))


# --- Happy path --------------------------------------------------------------


def test_success_returns_takes_words_and_execution_telemetry(setup) -> None:
    handler, engine, aligner, _normalizer = _handler(setup)
    result = _run(handler(_job()))

    assert set(result) == {"request_id", "takes", "telemetry"}
    assert result["request_id"] == REQUEST_ID
    assert len(result["takes"]) == 2
    take = result["takes"][0]
    assert set(take) == {
        "take",
        "audio_b64",
        "duration_seconds",
        "sample_rate",
        "channels",
        "seed",
        "words",
        "wer",
    }
    assert take["take"] == 1
    assert take["audio_b64"].startswith("UklGR")
    assert take["duration_seconds"] == pytest.approx(2.0, abs=1e-3)
    assert (take["sample_rate"], take["channels"]) == (48000, 2)
    # words — из аллайнера (мок GigaAM), а не из TTS-модели.
    assert take["words"], "words обязаны присутствовать при aligner.word_timestamps=true"
    assert all({"text", "start_ms", "end_ms"} <= set(word) for word in take["words"])
    assert take["wer"] == 0.0

    telemetry = result["telemetry"]
    assert telemetry["engine"] == "qwen3_tts"
    assert telemetry["engine_version"] == "qwen3-tts-12hz-1.7b@fake"
    assert telemetry["aligner_version"] == "gigaam-v3@fake"
    assert telemetry["model_hash"] == "sha256:" + model_sha("qwen3-tts-1.7b/model.safetensors")
    assert telemetry["gpu_seconds"] == pytest.approx(1.7)
    assert telemetry["license"] == "Apache-2.0"

    assert len(engine.calls) == 1
    assert len(aligner.calls) == 2
    assert aligner.calls[0]["language"] == "ru"
    _output_validator().validate(result)


def test_old_v1_input_has_additive_markup_telemetry(setup) -> None:
    """v1.2 аддитивно: старый input без prosody получает факты разметки, но prosody."""

    handler, engine, _aligner, normalizer = _handler(setup)
    result = _run(handler(_job()))

    telemetry = result["telemetry"]
    assert telemetry["normalization"] == {
        "version": normalizer("Швы затянуты. Следов нет.", "ru").version,
        "applied": ["fake"],
    }
    assert telemetry["stress_marks"] == 0
    assert telemetry["pauses"] == []
    assert "prosody" not in telemetry
    assert engine.calls[0]["prosody"] is None

    _output_validator().validate(result)


def test_take_facts_come_from_produced_file(setup) -> None:
    takes = [
        SynthesisTake(wav_bytes(2.0, sample_rate=24000, channels=1), 42),
        SynthesisTake(wav_bytes(2.05, sample_rate=24000, channels=1), 43),
    ]
    handler, _engine, _aligner, _normalizer = _handler(setup, engine=FakeEngine(takes=takes))
    result = _run(handler(_job(takes=2, seed=42)))
    first = result["takes"][0]
    assert (first["duration_seconds"], first["sample_rate"], first["channels"]) == (
        pytest.approx(2.0, abs=1e-3),
        24000,
        1,
    )
    assert first["seed"] == 42


def test_engine_receives_all_synthesis_parameters(setup) -> None:
    take = SynthesisTake(wav_bytes(1.0, sample_rate=24000, channels=1), 4242)
    handler, engine, _aligner, _normalizer = _handler(setup, engine=FakeEngine(takes=[take]))
    payload = _job(
        takes=1,
        speed=1.3,
        seed=4242,
        voice={"mode": "preset", "speaker_id": 3},
        render={"sample_rate": 24000, "channels": 1, "format": "wav"},
    )
    result = _run(handler(payload))

    call = engine.calls[0]
    assert call["takes"] == 1
    assert call["speed"] == 1.3
    assert call["seed"] == 4242
    assert call["voice"] == {"mode": "preset", "speaker_id": 3}
    assert (call["sample_rate"], call["channels"]) == (24000, 1)
    assert call["language"] == "ru"
    assert result["takes"][0]["seed"] == 4242


def test_window_is_tracing_only_and_not_fitted(setup) -> None:
    handler, engine, _aligner, _normalizer = _handler(setup)
    result = _run(handler(_job(window={"start": 5.0, "end": 5.1})))
    assert "window" not in engine.calls[0]
    # Длительность — факт файла, а не длина окна.
    assert result["takes"][0]["duration_seconds"] == pytest.approx(2.0, abs=1e-3)


def test_normalized_text_is_synthesized(setup) -> None:
    normalizer = FakeNormalizer(output="НОРМАЛИЗОВАННЫЙ ТЕКСТ")
    handler, engine, aligner, normalizer = _handler(setup, normalizer=normalizer)
    _run(handler(_job()))
    assert engine.calls[0]["text"] == "НОРМАЛИЗОВАННЫЙ ТЕКСТ"
    assert aligner.calls[0]["text"] == "НОРМАЛИЗОВАННЫЙ ТЕКСТ"
    assert normalizer.calls == [("Швы затянуты. Следов нет.", "ru")]


def test_real_dictionary_normalizer_end_to_end(setup) -> None:
    normalizer = DictionaryNormalizer.from_path(DATA / "normalization.ru.json")
    handler, engine, _aligner, _normalizer = _handler(setup, normalizer=normalizer)
    result = _run(handler(_job(text="ул. Ленина, д. 5")))
    assert engine.calls[0]["text"] == "улица Ленина, дом пять"
    _output_validator().validate(result)


# --- Пустой / битый результат ------------------------------------------------


@pytest.mark.parametrize(
    "takes,code",
    [
        ([], "audio_empty_result"),
        ([SynthesisTake(b"", None)], "output_empty"),
        ([SynthesisTake(b"not-an-audio", None)], "output_invalid"),
    ],
)
def test_empty_or_broken_result_is_an_error(setup, takes, code: str) -> None:
    handler, _engine, _aligner, _normalizer = _handler(
        setup, engine=FakeEngine(takes=takes)
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_job(takes=1)))
    assert error.value.code == code


def test_takes_count_mismatch_is_an_error(setup) -> None:
    handler, _engine, _aligner, _normalizer = _handler(
        setup, engine=FakeEngine(takes=[SynthesisTake(wav_bytes(), None)])
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_job(takes=2)))
    assert error.value.code == "takes_mismatch"


def test_empty_normalized_text_is_an_error(setup) -> None:
    handler, engine, _aligner, _normalizer = _handler(
        setup, normalizer=FakeNormalizer(output="   ")
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "text_empty"
    assert engine.calls == []


def test_invalid_alignment_is_not_silently_fixed(setup) -> None:
    bad = WordTiming(text="Швы", start_ms=100, end_ms=100)
    handler, _engine, _aligner, _normalizer = _handler(setup, aligner=FakeAligner(words=[bad]))
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "output_invalid"


def test_wer_out_of_range_is_an_error(setup) -> None:
    handler, _engine, _aligner, _normalizer = _handler(setup, aligner=FakeAligner(wer=1.5))
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "output_invalid"


# --- «Воркер не выбирает модель» --------------------------------------------


def test_worker_cannot_be_given_a_model_choice(setup) -> None:
    handler, engine, _aligner, _normalizer = _handler(setup)
    for field, value in (
        ("model", "qwen3-tts-0.6b"),
        ("model_slug", "chatterbox-multilingual-v3"),
        ("engine", "chatterbox"),
    ):
        with pytest.raises(WorkerError) as error:
            _run(handler(_job(**{field: value})))
        assert error.value.code == "input_invalid"
    assert engine.calls == []


def test_engine_and_versions_come_from_execution(setup) -> None:
    engine = FakeEngine(engine_version="qwen3-tts-12hz-1.7b@from-image")
    aligner = FakeAligner(aligner_version="gigaam-v3@from-image")
    handler, _engine, _aligner, _normalizer = _handler(setup, engine=engine, aligner=aligner)
    result = _run(handler(_job()))
    assert result["telemetry"]["engine_version"] == "qwen3-tts-12hz-1.7b@from-image"
    assert result["telemetry"]["aligner_version"] == "gigaam-v3@from-image"
    assert result["telemetry"]["model_hash"] == "sha256:" + model_sha(
        "qwen3-tts-1.7b/model.safetensors"
    )


# --- Aligner-режимы ----------------------------------------------------------


def test_aligner_disabled_skips_words_wer_and_aligner_model(setup) -> None:
    _config, _manifest, models_root = setup
    # aligner не исполняется — его веса не проверяются (job не должен упасть).
    (models_root / "gigaam-v3" / "pytorch_model.bin").unlink()
    handler, _engine, aligner, _normalizer = _handler(setup)
    result = _run(handler(_job(aligner={"enabled": False, "word_timestamps": True})))
    take = result["takes"][0]
    assert take["words"] == []
    assert take["wer"] is None
    assert result["telemetry"]["aligner_version"] is None
    assert aligner.calls == []
    _output_validator().validate(result)


def test_word_timestamps_false_keeps_wer_but_omits_words(setup) -> None:
    handler, _engine, aligner, _normalizer = _handler(setup)
    result = _run(handler(_job(aligner={"enabled": True, "word_timestamps": False})))
    take = result["takes"][0]
    assert take["words"] == []
    assert take["wer"] == 0.0
    assert aligner.calls, "WER меряется — aligner обязан исполниться"
    assert result["telemetry"]["aligner_version"] == "gigaam-v3@fake"


# --- Модели на томе ----------------------------------------------------------


def test_missing_engine_model_is_visible(setup) -> None:
    _config, _manifest, models_root = setup
    (models_root / "qwen3-tts-1.7b" / "model.safetensors").unlink()
    handler, engine, _aligner, _normalizer = _handler(setup)
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "model_file_missing"
    assert engine.calls == []


def test_engine_hash_mismatch_is_provision_hash_mismatch(setup) -> None:
    _config, _manifest, models_root = setup
    (models_root / "qwen3-tts-1.7b" / "model.safetensors").write_bytes(b"tampered")
    handler, _engine, _aligner, _normalizer = _handler(setup)
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "provision_hash_mismatch"


def test_aligner_hash_mismatch_is_provision_hash_mismatch(setup) -> None:
    _config, _manifest, models_root = setup
    (models_root / "gigaam-v3" / "pytorch_model.bin").write_bytes(b"tampered")
    handler, _engine, _aligner, _normalizer = _handler(setup)
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "provision_hash_mismatch"


# --- Доставка: base64 vs URL -------------------------------------------------


def _small_inline_config(config):
    return dataclasses.replace(
        config,
        output_inline_max_mb=0.0001,
        output_inline_budget_mb=0.0001,
    )


def test_large_takes_go_to_storage_url(setup) -> None:
    config, manifest, _models_root = setup
    storage = FakeStorage()
    engine = FakeEngine()
    handler = build_handler(
        _small_inline_config(config),
        manifest,
        engine,
        FakeAligner(),
        FakeNormalizer(),
        storage=storage,
        clock=FakeClock([10.0, 11.7]),
    )
    result = _run(handler(_job(takes=2)))
    assert len(result["takes"]) == 2
    for take in result["takes"]:
        assert "audio_url" in take
        assert "audio_b64" not in take
        assert take["audio_url"].startswith("https://")
    assert [content_type for _data, content_type in storage.uploads] == ["audio/wav", "audio/wav"]
    _output_validator().validate(result)


def test_large_takes_without_storage_fail_visibly(setup) -> None:
    config, manifest, _models_root = setup
    handler = build_handler(
        _small_inline_config(config),
        manifest,
        FakeEngine(),
        FakeAligner(),
        FakeNormalizer(),
        clock=FakeClock([10.0, 11.7]),
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_job(takes=1)))
    assert error.value.code == "output_storage_missing"


# --- Валидация job и пример контракта ---------------------------------------


@pytest.mark.parametrize(
    "job",
    [
        None,
        {"id": "x"},
        {"id": "x", "input": "not-an-object"},
        {"id": "x", "input": {}},
    ],
)
def test_invalid_job_is_input_invalid(setup, job: object) -> None:
    handler, _engine, _aligner, _normalizer = _handler(setup)
    with pytest.raises(WorkerError) as error:
        _run(handler(job))
    assert error.value.code == "input_invalid"


def test_contract_example_end_to_end(setup) -> None:
    payload = load_json(contract_dir() / "examples" / "request.json")["input"]
    normalizer = DictionaryNormalizer.from_path(DATA / "normalization.ru.json")
    handler, engine, aligner, _normalizer = _handler(setup, normalizer=normalizer)
    result = _run(handler({"id": "example", "input": payload}))

    _output_validator().validate(result)
    assert result["request_id"] == payload["request_id"]
    assert len(result["takes"]) == payload["takes"]
    assert result["takes"][0]["words"]
    assert engine.calls[0]["text"] == payload["text"]
    assert len(aligner.calls) == payload["takes"]


def test_import_without_runpod() -> None:
    sys.modules.pop("runpod", None)
    import tts_worker  # noqa: F401
    import tts_worker.handler  # noqa: F401

    assert "runpod" not in sys.modules


# --- Клонирование голоса (Base) ---------------------------------------------


def test_clone_job_passes_reference_to_engine_and_reports_telemetry(setup) -> None:
    downloader = FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    decoder = FakeReferenceDecoder(pcm=REFERENCE_PCM, duration_seconds=3.0)
    handler, engine, _aligner, _normalizer = _clone_handler(
        setup, downloader=downloader, decoder=decoder
    )
    result = _run(handler(_clone_job()))

    assert downloader.urls == [REFERENCE_URL]
    assert decoder.calls == [REFERENCE_BYTES]
    call = engine.calls[0]
    assert call["voice"]["mode"] == "clone"
    reference = call["reference"]
    assert reference is not None
    assert reference.pcm is REFERENCE_PCM
    assert reference.sample_rate == 24000
    assert reference.sha256 == REFERENCE_SHA
    assert reference.text == "Проверка голоса для клонирования."
    assert reference.duration_seconds == pytest.approx(3.0)

    telemetry = result["telemetry"]
    assert telemetry["voice_mode"] == "clone"
    assert telemetry["reference_hash"] == REFERENCE_SHA
    assert telemetry["engine"] == "qwen3_tts"
    _output_validator().validate(result)


def test_preset_telemetry_is_preset_without_reference_hash(setup) -> None:
    handler, engine, _aligner, _normalizer = _handler(setup)
    result = _run(handler(_job()))
    assert engine.calls[0]["reference"] is None
    assert result["telemetry"]["voice_mode"] == "preset"
    assert result["telemetry"]["reference_hash"] is None
    _output_validator().validate(result)


def test_clone_hash_mismatch_is_visible_and_engine_is_not_called(setup) -> None:
    handler, engine, _aligner, _normalizer = _clone_handler(setup)
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job(sha256="b" * 64)))
    assert error.value.code == "reference_hash"
    assert REFERENCE_SHA in error.value.message or "sha256" in error.value.message
    assert engine.calls == []


def test_clone_too_large_is_visible(setup, monkeypatch) -> None:
    import tts_worker.reference as reference_module

    monkeypatch.setattr(reference_module, "MAX_REFERENCE_BYTES", len(REFERENCE_BYTES) - 1)
    # Выше лимита режет и prepare_reference (кастомный загрузчик обходит httpx).
    handler, engine, _aligner, _normalizer = _clone_handler(
        setup, downloader=FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job()))
    assert error.value.code == "reference_too_large"
    assert engine.calls == []


@pytest.mark.parametrize("duration", [0.5, 20.0, 0.0])
def test_clone_reference_duration_out_of_range(setup, duration: float) -> None:
    handler, engine, _aligner, _normalizer = _clone_handler(
        setup, decoder=FakeReferenceDecoder(duration_seconds=duration)
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job()))
    assert error.value.code == "reference_invalid"
    assert engine.calls == []


def test_clone_download_failure_is_visible_without_url_in_message(setup) -> None:
    failure = WorkerError("reference_invalid", "скачивание референса: HTTP 403")
    downloader = FakeReferenceDownloader(error=failure)
    handler, engine, _aligner, _normalizer = _clone_handler(setup, downloader=downloader)
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job()))
    assert error.value.code == "reference_invalid"
    assert REFERENCE_URL not in str(error.value)
    assert engine.calls == []


def test_clone_decoder_failure_is_visible(setup) -> None:
    handler, engine, _aligner, _normalizer = _clone_handler(
        setup, decoder=FakeReferenceDecoder(error=ValueError("битый контейнер"))
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job()))
    assert error.value.code == "reference_invalid"
    assert engine.calls == []


def test_clone_reference_url_must_be_http_schema_rejects_ftp(setup) -> None:
    handler, engine, _aligner, _normalizer = _clone_handler(setup)
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job(url="ftp://files.example/ref.wav")))
    assert error.value.code == "input_invalid"
    assert engine.calls == []


def test_clone_without_reference_is_input_invalid(setup) -> None:
    handler, engine, _aligner, _normalizer = _clone_handler(setup)
    job = _job()
    job["input"]["voice"] = {"mode": "clone"}
    with pytest.raises(WorkerError) as error:
        _run(handler(job))
    assert error.value.code == "input_invalid"
    assert engine.calls == []


def test_prepare_reference_without_reference_is_reference_missing(setup) -> None:
    downloader = FakeReferenceDownloader()
    decoder = FakeReferenceDecoder()
    with pytest.raises(WorkerError) as error:
        _run(prepare_reference({"mode": "clone"}, downloader=downloader, decoder=decoder))
    assert error.value.code == "reference_missing"
    assert downloader.urls == []


def test_clone_logs_hash_but_not_reference_url(setup, caplog) -> None:
    handler, _engine, _aligner, _normalizer = _clone_handler(setup)
    with caplog.at_level(logging.INFO, logger="tts_worker.handler"):
        _run(handler(_clone_job()))
    assert REFERENCE_SHA in caplog.text
    assert REFERENCE_URL not in caplog.text


@pytest.mark.parametrize("mode", ["preset", "design"])
def test_engine_rejects_non_clone_modes_on_base(setup, mode: str) -> None:
    _config, _manifest, models_root = setup
    engine = Qwen3TtsEngine(models_root / "qwen3-tts-1.7b", "qwen3-tts@fake")
    handler, _engine, _aligner, _normalizer = _handler(setup, engine=engine)
    job = _job()
    job["input"]["voice"] = (
        {"mode": "preset", "speaker_id": 0} if mode == "preset" else {"mode": "clone"}
    )
    with pytest.raises(WorkerError) as error:
        _run(handler(job))
    # preset: схема валидна, движок Base обязан дать voice_mode_unsupported
    # (никакого тихого перехода на пресет); design схемой уже отсекается.
    assert error.value.code in {"voice_mode_unsupported", "input_invalid"}


def test_preset_on_base_fails_visibly_without_loading_the_model(setup) -> None:
    _config, _manifest, models_root = setup
    engine = Qwen3TtsEngine(models_root / "qwen3-tts-1.7b", "qwen3-tts@fake")
    handler, _engine, _aligner, _normalizer = _handler(setup, engine=engine)
    with pytest.raises(WorkerError) as error:
        _run(handler(_job()))
    assert error.value.code == "voice_mode_unsupported"
    assert engine._model is None


def test_clone_without_gpu_runtime_is_engine_unavailable(setup) -> None:
    _config, _manifest, models_root = setup
    engine = Qwen3TtsEngine(models_root / "qwen3-tts-1.7b", "qwen3-tts@fake")
    handler, _engine, _aligner, _normalizer = _clone_handler(setup, engine=engine)
    with pytest.raises(WorkerError) as error:
        _run(handler(_clone_job()))
    # В тестовом окружении нет torch/qwen-tts: видимый код, не ImportError и не
    # подмена пресетом.
    assert error.value.code == "engine_unavailable"
