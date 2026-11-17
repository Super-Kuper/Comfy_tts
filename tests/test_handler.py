"""Handler: контракт, факты исполнения, пустой результат, «модель не выбирается»."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import sys

import pytest
from helpers import (
    DATA,
    FakeAligner,
    FakeClock,
    FakeEngine,
    FakeNormalizer,
    FakeStorage,
    contract_dir,
    load_json,
    model_sha,
    wav_bytes,
)
from jsonschema import Draft202012Validator

from tts_worker.aligner import WordTiming
from tts_worker.engine import SynthesisTake
from tts_worker.errors import WorkerError
from tts_worker.handler import build_handler
from tts_worker.normalization import DictionaryNormalizer

REQUEST_ID = "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001"


# --- Вспомогательное ---------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _job(**input_overrides) -> dict:
    payload = copy.deepcopy(load_json(contract_dir() / "examples" / "request.json")["input"])
    payload.update(input_overrides)
    return {"id": "job-1", "input": payload}


def _handler(setup, *, engine=None, aligner=None, normalizer=None, storage=None, clock=None):
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
    )
    return handler, engine, aligner, normalizer


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
