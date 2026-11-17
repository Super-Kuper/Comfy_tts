"""Qwen3-TTS Base engine: clone-only режим, takes/speed/resample, видимые ошибки.

Тесты без GPU/torch/numpy/librosa: модель, генерация и аудио-постобработка
подменяются, а контроль потока (takes, seed, speed ≠ 1.0, resample, каналы,
WAV) проверяется на границах. Живой ``generate_voice_clone`` не запускается —
это остаётся шагом живого GPU-стенда.
"""

from __future__ import annotations

import pytest
from helpers import REFERENCE_PCM, reference_sha256

from tts_worker import engine as engine_module
from tts_worker.engine import (
    Qwen3TtsEngine,
    SynthesisTake,
    language_for_model,
)
from tts_worker.errors import WorkerError
from tts_worker.reference import ReferenceAudio

REFERENCE = ReferenceAudio(
    pcm=REFERENCE_PCM,
    sample_rate=24000,
    text="Проверка голоса для клонирования.",
    language="ru",
    sha256=reference_sha256(),
    duration_seconds=3.0,
)


def _engine(tmp_path) -> Qwen3TtsEngine:
    return Qwen3TtsEngine(tmp_path / "qwen3-tts-1.7b", "qwen3-tts-12hz-1.7b@fake")


def _synthesize(engine: Qwen3TtsEngine, **overrides):
    kwargs = {
        "language": "ru",
        "voice": {"mode": "clone"},
        "speed": 1.0,
        "seed": None,
        "takes": 1,
        "sample_rate": 24000,
        "channels": 1,
        "reference": REFERENCE,
    }
    kwargs.update(overrides)
    return engine.synthesize("Швы затянуты. Следов нет.", **kwargs)


# --- Режимы голоса -----------------------------------------------------------


def test_preset_is_unsupported_on_base_without_loading_model(tmp_path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(WorkerError) as error:
        _synthesize(engine, voice={"mode": "preset", "speaker_id": 0})
    assert error.value.code == "voice_mode_unsupported"
    assert "clone" in error.value.message
    assert engine._model is None, "preset обязан падать до загрузки модели"


@pytest.mark.parametrize("voice", [{}, {"mode": "design"}, {"mode": ""}])
def test_unknown_or_missing_mode_is_voice_mode_unsupported(tmp_path, voice: dict) -> None:
    with pytest.raises(WorkerError) as error:
        _synthesize(_engine(tmp_path), voice=voice)
    assert error.value.code == "voice_mode_unsupported"


def test_clone_without_reference_is_reference_missing(tmp_path) -> None:
    with pytest.raises(WorkerError) as error:
        _synthesize(_engine(tmp_path), reference=None)
    assert error.value.code == "reference_missing"


def test_clone_without_gpu_runtime_is_engine_unavailable(tmp_path) -> None:
    # В тестовом окружении нет qwen-tts/torch: видимый код, а не ImportError.
    with pytest.raises(WorkerError) as error:
        _synthesize(_engine(tmp_path))
    assert error.value.code == "engine_unavailable"


# --- Control flow: takes, seed, speed/resample -------------------------------


def test_clone_takes_loop_and_postprocess_parameters(tmp_path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    postprocess_calls: list[dict] = []
    generate_calls: list[dict] = []

    def fake_generate(self, model, *, text, language, reference, seed):
        generate_calls.append(
            {
                "model": model,
                "text": text,
                "language": language,
                "reference": reference,
                "seed": seed,
            }
        )
        return REFERENCE_PCM, 24000

    def fake_postprocess(pcm, **kwargs):
        postprocess_calls.append({"pcm": pcm, **kwargs})
        return b"wav-take"

    monkeypatch.setattr(engine, "_load", lambda: object())
    monkeypatch.setattr(Qwen3TtsEngine, "_generate_clone", fake_generate)
    monkeypatch.setattr(engine_module, "_postprocess", fake_postprocess)

    takes = _synthesize(engine, takes=2, seed=7, speed=1.4, sample_rate=48000, channels=2)

    assert len(takes) == 2
    assert all(isinstance(take, SynthesisTake) for take in takes)
    assert [take.audio for take in takes] == [b"wav-take", b"wav-take"]
    assert [take.seed for take in takes] == [7, 7]
    assert len(generate_calls) == 2
    assert generate_calls[0]["reference"] is REFERENCE
    assert generate_calls[0]["language"] == "ru"
    assert len(postprocess_calls) == 2
    assert postprocess_calls[0]["source_rate"] == 24000
    assert postprocess_calls[0]["target_rate"] == 48000
    assert postprocess_calls[0]["channels"] == 2
    assert postprocess_calls[0]["speed"] == 1.4


def test_generate_clone_calls_qwen_with_reference_tuple_and_language_name(tmp_path) -> None:
    engine = _engine(tmp_path)

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def generate_voice_clone(self, **kwargs):
            self.kwargs = kwargs
            return [REFERENCE_PCM], 24000

    model = FakeModel()
    pcm, sample_rate = engine._generate_clone(
        model,
        text="Швы затянуты.",
        language="ru",
        reference=REFERENCE,
        seed=None,
    )
    assert pcm is REFERENCE_PCM
    assert sample_rate == 24000
    assert model.kwargs == {
        "text": "Швы затянуты.",
        "language": "russian",
        "ref_audio": (REFERENCE_PCM, 24000),
        "ref_text": REFERENCE.text,
    }


def test_generate_clone_seed_requires_torch(tmp_path) -> None:
    engine = _engine(tmp_path)

    class FakeModel:
        def generate_voice_clone(self, **kwargs):  # pragma: no cover - не дойдёт
            raise AssertionError("генерация не должна начаться без torch")

    with pytest.raises(WorkerError) as error:
        engine._generate_clone(
            FakeModel(), text="x", language="ru", reference=REFERENCE, seed=42
        )
    assert error.value.code == "engine_unavailable"


def test_generate_clone_error_is_engine_failed(tmp_path) -> None:
    engine = _engine(tmp_path)

    class BoomModel:
        def generate_voice_clone(self, **kwargs):
            raise RuntimeError("CUDA OOM")

    with pytest.raises(WorkerError) as error:
        engine._generate_clone(BoomModel(), text="x", language="ru", reference=REFERENCE, seed=None)
    assert error.value.code == "engine_failed"
    assert "CUDA OOM" in error.value.message


def test_generate_clone_empty_result_is_audio_empty_result(tmp_path) -> None:
    engine = _engine(tmp_path)

    class EmptyModel:
        def generate_voice_clone(self, **kwargs):
            return [], 24000

    with pytest.raises(WorkerError) as error:
        engine._generate_clone(EmptyModel(), text="x", language="ru", reference=REFERENCE, seed=None)
    assert error.value.code == "audio_empty_result"


# --- Постобработка: speed → resample → channels → WAV -----------------------


def test_postprocess_applies_speed_resample_and_channels(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_time_stretch(pcm, *, rate):
        calls.append(("time_stretch", rate))
        return "stretched"

    def fake_resample(pcm, *, orig_sr, target_sr):
        calls.append(("resample", orig_sr, target_sr))
        return "resampled"

    def fake_channels(pcm, *, channels):
        calls.append(("channels", channels))
        return "channels"

    def fake_encode(pcm, *, sample_rate):
        calls.append(("encode", sample_rate))
        return b"RIFF..."

    monkeypatch.setattr(engine_module, "_time_stretch", fake_time_stretch)
    monkeypatch.setattr(engine_module, "_resample", fake_resample)
    monkeypatch.setattr(engine_module, "_to_channels", fake_channels)
    monkeypatch.setattr(engine_module, "_encode_wav", fake_encode)

    audio = engine_module._postprocess(
        object(), source_rate=24000, target_rate=48000, channels=2, speed=1.5
    )
    assert audio == b"RIFF..."
    assert calls == [
        ("time_stretch", 1.5),
        ("resample", 24000, 48000),
        ("channels", 2),
        ("encode", 48000),
    ]


def test_postprocess_speed_one_skips_time_stretch_and_same_rate_skips_resample(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        engine_module, "_time_stretch", lambda pcm, *, rate: calls.append("stretch")
    )
    monkeypatch.setattr(
        engine_module, "_resample", lambda pcm, *, orig_sr, target_sr: calls.append("resample")
    )
    monkeypatch.setattr(engine_module, "_to_channels", lambda pcm, *, channels: "mono")
    monkeypatch.setattr(engine_module, "_encode_wav", lambda pcm, *, sample_rate: b"wav")

    assert engine_module._postprocess(
        object(), source_rate=24000, target_rate=24000, channels=1, speed=1.0
    ) == b"wav"
    assert calls == []


@pytest.mark.parametrize(
    "language,expected",
    [("ru", "russian"), ("RU", "russian"), ("en", "english"), ("auto", "auto"), ("", "auto")],
)
def test_language_for_model_maps_iso_to_qwen_names(language: str, expected: str) -> None:
    assert language_for_model(language) == expected
