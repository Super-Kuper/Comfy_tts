"""Просодия v1.2: авторская разметка, сегменты, DSP и телеметрия воркера.

GPU-free: словарь — данные из ``data/``, движок/аллайнер — инъекции, DSP-функции
librosa/numpy/soundfile подменяются, склейка сегментов — stdlib ``wave``.
"""

from __future__ import annotations

import asyncio
import copy
import math

import pytest
from helpers import (
    DATA,
    REFERENCE_PCM,
    FakeAligner,
    FakeClock,
    FakeEngine,
    FakeReferenceDecoder,
    FakeReferenceDownloader,
    contract_dir,
    load_json,
    reference_bytes,
    reference_sha256,
    wav_bytes,
)

from tts_worker import engine as engine_module
from tts_worker.aligner import Alignment, WordTiming
from tts_worker.audio import read_audio_info
from tts_worker.dsp import (
    combine_wav_segments,
    effective_rate,
    gain_scale,
    whisper_kernel,
)
from tts_worker.engine import Qwen3TtsEngine, SynthesisTake
from tts_worker.errors import WorkerError
from tts_worker.handler import build_handler
from tts_worker.normalization import DictionaryNormalizer, apply_stress_markup
from tts_worker.prosody import (
    MAX_PAUSES_PER_PHRASE,
    PAUSE_MAX_SECONDS,
    PAUSE_MIN_SECONDS,
    build_segments,
    parse_prosody,
    sentence_pieces,
    split_pause_markup,
)

REQUEST_ID = "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001"
REFERENCE_URL = "https://files.example/voice/reference-001.wav?X-Amz-Signature=test"
REFERENCE_BYTES = reference_bytes(3.0)
REFERENCE_SHA = reference_sha256(REFERENCE_BYTES)

VALID_PROSODY = {
    "style_id": "confident",
    "style_catalog_version": "voice-styles@2026-11-18",
    "rate": 1.2,
    "gain_db": 1.0,
    "pitch_semitones": 0.0,
    "sentence_pause_ms": 80,
    "whisper": False,
}


# --- Вспомогательное ---------------------------------------------------------


def _normalizer() -> DictionaryNormalizer:
    return DictionaryNormalizer.from_path(DATA / "normalization.ru.json")


def _job(**input_overrides) -> dict:
    payload = copy.deepcopy(load_json(contract_dir() / "examples" / "request.json")["input"])
    payload.update(input_overrides)
    return {"id": "job-prosody", "input": payload}


def _clone_job(**input_overrides) -> dict:
    job = _job()
    job["input"]["voice"] = {
        "mode": "clone",
        "reference": {
            "url": REFERENCE_URL,
            "sha256": REFERENCE_SHA,
            "text": "Проверка голоса для клонирования.",
            "language": "ru",
        },
    }
    job["input"].update(input_overrides)
    return job


def _handler(setup, *, engine=None, aligner=None, normalizer=None, downloader=None, decoder=None):
    config, manifest, _models_root = setup
    handler = build_handler(
        config,
        manifest,
        engine or FakeEngine(),
        aligner or FakeAligner(),
        normalizer or _normalizer(),
        clock=FakeClock([10.0, 11.7]),
        reference_downloader=downloader,
        reference_decoder=decoder,
    )
    return handler


def _run(coro):
    return asyncio.run(coro)


def _prosody(**overrides) -> dict:
    payload = dict(VALID_PROSODY)
    payload.update(overrides)
    return payload


class _PerSegmentAligner:
    """Аллергейт GigaAM: WER/слова зависят от сегмента (проверка агрегата)."""

    aligner_version = "gigaam-v3@fake"

    def __init__(self, wer_by_text: dict[str, float], words_by_text: dict[str, int] | None = None):
        self._wer_by_text = wer_by_text
        self._words_by_text = words_by_text or {}
        self.calls: list[dict] = []

    def word_timestamps(self, audio: bytes, text: str, *, language: str) -> Alignment:
        self.calls.append({"audio": audio, "text": text, "language": language})
        count = self._words_by_text.get(text, 1)
        words = tuple(
            WordTiming(text=f"w{index}", start_ms=40 + 160 * index, end_ms=200 + 160 * index)
            for index in range(count)
        )
        return Alignment(words=words, wer=self._wer_by_text[text])


# --- Ударение `+` ------------------------------------------------------------


def test_stress_mark_converted_after_vowel() -> None:
    text, applied, marks = apply_stress_markup("звон+ит")
    assert text == "звони\u0301т"
    assert applied == ("stress:звонит",)
    assert marks == 1


def test_stress_mark_uppercase_and_first_syllable() -> None:
    text, applied, marks = apply_stress_markup("+Это и Зам+ок")
    assert text == "Э\u0301то и Замо\u0301к"
    assert marks == 2
    assert applied == ("stress:Это", "stress:Замок")


def test_stress_plus_outside_cyrillic_word_is_literal() -> None:
    text, applied, marks = apply_stress_markup("C++ и 5+5, a+b")
    assert text == "C++ и 5+5, a+b"
    assert applied == ()
    assert marks == 0


def test_stress_yo_is_invalid() -> None:
    with pytest.raises(WorkerError) as error:
        apply_stress_markup("+ёлка")
    assert error.value.code == "stress_invalid"
    assert "ё" in error.value.message


@pytest.mark.parametrize("text", ["звон+т", "дом+", "+дом", "++и"])
def test_stress_invalid_marks(text: str) -> None:
    with pytest.raises(WorkerError) as error:
        apply_stress_markup(text)
    assert error.value.code == "stress_invalid"


def test_stress_one_mark_per_word() -> None:
    with pytest.raises(WorkerError) as error:
        apply_stress_markup("зв+он+ит")
    assert error.value.code == "stress_invalid"
    assert "одной метки" in error.value.message


def test_explicit_stress_wins_over_dictionary_and_is_protected() -> None:
    normalizer = _normalizer()
    marked = normalizer("зам+ок", "ru")
    assert marked.text == "замо\u0301к"
    assert marked.applied == ("stress:замок",)
    assert marked.stress_marks == 1
    # Словарная замена не перетирает явную метку, но слово без метки — меняет.
    mixed = normalizer("зам+ок и замок", "ru")
    assert mixed.text == "замо\u0301к и за\u0301мок"
    assert mixed.applied.count("stress:замок") == 2
    assert mixed.stress_marks == 1


def test_normalizer_reports_stress_marks_and_data_version() -> None:
    normalizer = _normalizer()
    result = normalizer("звон+ит [[pause:0.4]] крас+ивее", "ru")
    # pause-метка защищена от правил числа/словаря; ударения — факт.
    assert "[[pause:0.4]]" in result.text
    assert result.stress_marks == 2
    assert result.applied == ("pause:0.4", "stress:звонит", "stress:красивее")
    assert result.version == "ru-normalization@2026-11-18"


# --- Паузы `[[pause]]` -------------------------------------------------------


def test_pause_markers_split_segments() -> None:
    fragments, pauses, applied = split_pause_markup("до [[pause:0.4]] после")
    assert fragments == ["до", "после"]
    assert pauses == [0.4]
    assert applied == ["pause:0.4"]


def test_pause_boundaries_and_step_are_valid() -> None:
    _fragments, pauses, _applied = split_pause_markup(
        "a [[pause:0.05]] b [[pause:2.0]] c"
    )
    assert pauses == [PAUSE_MIN_SECONDS, PAUSE_MAX_SECONDS]
    with pytest.raises(WorkerError) as error:
        split_pause_markup("a [[pause:0.03]] b")
    assert error.value.code == "pause_invalid"
    with pytest.raises(WorkerError) as error:
        split_pause_markup("a [[pause:0.07]] b")
    assert error.value.code == "pause_invalid"


@pytest.mark.parametrize(
    "text",
    [
        "[[pause:0.4]] после",
        "до [[pause:0.4]]",
        "a [[pause:0.4]][[pause:0.5]] b",
        "a [[pause:0.4] b",
        "a [[pause:]] b",
        "a [[pause:400ms]] b",
        "a [[pause:0.4]] b [[pause:0.4]] c [[pause:0.4]] d [[pause:0.4]] "
        "e [[pause:0.4]] f [[pause:0.4]] g [[pause:0.4]] h",
        "a [[pause:2.0]] b [[pause:2.0]] c [[pause:2.0]] d [[pause:2.0]] e",
    ],
)
def test_pause_invalid_forms(text: str) -> None:
    with pytest.raises(WorkerError) as error:
        split_pause_markup(text)
    assert error.value.code == "pause_invalid"


def test_pause_max_total_is_six_seconds() -> None:
    # 6 × 1.0 = 6.0 — граница; 6 × 1.5 > 6 — ошибка.
    fragments, pauses, _ = split_pause_markup(
        "a [[pause:1.0]] b [[pause:1.0]] c [[pause:1.0]] d "
        "[[pause:1.0]] e [[pause:1.0]] f [[pause:1.0]] g"
    )
    assert len(fragments) == 7
    assert pauses == [1.0] * MAX_PAUSES_PER_PHRASE
    with pytest.raises(WorkerError) as error:
        split_pause_markup(
            "a [[pause:1.5]] b [[pause:1.5]] c [[pause:1.5]] d "
            "[[pause:1.5]] e [[pause:1.5]] f"
        )
    assert error.value.code == "pause_invalid"


def test_no_markup_is_single_segment() -> None:
    segments, pauses, applied = build_segments("Швы затянуты. Следов нет.")
    assert [(segment.text, segment.pause_after_seconds) for segment in segments] == [
        ("Швы затянуты. Следов нет.", 0.0)
    ]
    assert pauses == []
    assert applied == []


# --- Паузы предложений -------------------------------------------------------


def test_sentence_pause_segments_and_exact_silence() -> None:
    segments, pauses, _applied = build_segments(
        "Раз. Два? Три!", sentence_pause_ms=100
    )
    assert [(segment.text, segment.pause_after_seconds) for segment in segments] == [
        ("Раз.", 0.1),
        ("Два?", 0.1),
        ("Три!", 0.0),
    ]
    assert pauses == []


def test_sentence_pause_combines_with_pause_marker() -> None:
    segments, pauses, _applied = build_segments(
        "Раз. Два [[pause:0.4]] Три.", sentence_pause_ms=100
    )
    assert [(segment.text, segment.pause_after_seconds) for segment in segments] == [
        ("Раз.", 0.1),
        ("Два", 0.4),
        ("Три.", 0.0),
    ]
    assert pauses == [0.4]


def test_sentence_split_requires_whitespace_and_keeps_punctuation() -> None:
    assert sentence_pieces("Раз. Два?! Три…") == ["Раз.", "Два?!", "Три…"]
    # Без пробела границы нет: опечатка/десятичная запись не рождает паузу.
    assert sentence_pieces("Раз.Два") == ["Раз.Два"]


def test_sentence_pause_zero_does_not_split() -> None:
    segments, _pauses, _applied = build_segments("Раз. Два.", sentence_pause_ms=0)
    assert [segment.text for segment in segments] == ["Раз. Два."]


# --- Prosody-параметры -------------------------------------------------------


def test_prosody_absent_is_none() -> None:
    assert parse_prosody(None) is None


def test_prosody_valid_fields_parsed() -> None:
    params = parse_prosody(_prosody())
    assert params is not None
    assert params.style_id == "confident"
    assert params.style_catalog_version == "voice-styles@2026-11-18"
    assert (params.rate, params.gain_db, params.pitch_semitones) == (1.2, 1.0, 0.0)
    assert params.sentence_pause_ms == 80
    assert params.whisper is False


def test_prosody_sentence_pause_accepts_integral_float() -> None:
    assert parse_prosody(_prosody(sentence_pause_ms=80.0)).sentence_pause_ms == 80


@pytest.mark.parametrize(
    "patch",
    [
        {"rate": 0.4},
        {"rate": 2.1},
        {"gain_db": -7},
        {"gain_db": 7},
        {"pitch_semitones": -2.1},
        {"pitch_semitones": 2.1},
        {"sentence_pause_ms": -1},
        {"sentence_pause_ms": 501},
        {"sentence_pause_ms": 80.5},
        {"style_id": "  "},
        {"style_catalog_version": ""},
        {"whisper": "yes"},
        {"rate": True},
    ],
)
def test_prosody_invalid_values(patch: dict) -> None:
    payload = _prosody()
    payload.update(patch)
    with pytest.raises(WorkerError) as error:
        parse_prosody(payload)
    assert error.value.code == "prosody_invalid"


@pytest.mark.parametrize("missing", sorted(VALID_PROSODY))
def test_prosody_missing_field(missing: str) -> None:
    payload = _prosody()
    payload.pop(missing)
    with pytest.raises(WorkerError) as error:
        parse_prosody(payload)
    assert error.value.code == "prosody_invalid"


def test_prosody_unknown_field_is_invalid() -> None:
    with pytest.raises(WorkerError) as error:
        parse_prosody(_prosody(extremely=False))
    assert error.value.code == "prosody_invalid"
    assert "неизвестные поля" in error.value.message


# --- DSP: effective rate / gain / whisper ------------------------------------


@pytest.mark.parametrize(
    "speed,rate,expected",
    [(1.0, 1.0, 1.0), (1.5, 1.2, 1.8), (2.0, 0.5, 1.0), (0.5, 2.0, 1.0), (1.25, 0.8, 1.0)],
)
def test_effective_rate(speed: float, rate: float, expected: float) -> None:
    assert effective_rate(speed, rate) == pytest.approx(expected)


@pytest.mark.parametrize("speed,rate", [(1.5, 1.5), (0.5, 0.5), (2.0, 1.1), (0.9, 0.5)])
def test_effective_rate_out_of_range_is_prosody_rate(speed: float, rate: float) -> None:
    with pytest.raises(WorkerError) as error:
        effective_rate(speed, rate)
    assert error.value.code == "prosody_rate"


def test_gain_scale_never_clips() -> None:
    # +6 dB при пике 0.9 упирается в потолок 0.999, фактическое усиление меньше.
    scale = gain_scale(0.9, 6.0)
    assert 0.9 * scale == pytest.approx(0.999)
    assert scale < 10 ** (6 / 20)
    # Без риска клиппинга — ровно запрошенный gain.
    assert gain_scale(0.25, 6.0) == pytest.approx(10 ** (6 / 20))
    assert gain_scale(0.5, -6.0) == pytest.approx(10 ** (-6 / 20))
    assert gain_scale(0.0, 3.0) == pytest.approx(10 ** (3 / 20))


def test_whisper_kernel_is_normalized_lowpass() -> None:
    kernel = whisper_kernel(48000)
    assert len(kernel) % 2 == 1
    assert sum(kernel) == pytest.approx(1.0)
    middle = len(kernel) // 2
    for offset in range(1, middle + 1):
        assert kernel[middle - offset] == pytest.approx(kernel[middle + offset])

    def response(hz: float) -> float:
        return abs(
            sum(
                value
                * complex(
                    math.cos(-2 * math.pi * hz * index / 48000),
                    math.sin(-2 * math.pi * hz * index / 48000),
                )
                for index, value in enumerate(kernel)
            )
        )

    # Шёпот — lowpass: высокие частоты гасятся заметно сильнее низких.
    assert response(1000.0) > 0.8
    assert response(10000.0) < response(1000.0) / 4


@pytest.mark.parametrize("sample_rate,cutoff", [(0, 3500.0), (48000, 30000.0), (48000, 0.0)])
def test_whisper_kernel_rejects_impossible_cutoff(sample_rate: int, cutoff: float) -> None:
    with pytest.raises(WorkerError) as error:
        whisper_kernel(sample_rate, cutoff_hz=cutoff)
    assert error.value.code in {"output_invalid"}


# --- Склейка сегментов -------------------------------------------------------


def test_combine_wav_segments_inserts_exact_silence() -> None:
    combined = combine_wav_segments(
        [wav_bytes(1.0, sample_rate=48000, channels=2), wav_bytes(0.5, sample_rate=48000, channels=2)],
        [0.4],
    )
    info = read_audio_info(combined)
    assert (info.sample_rate, info.channels) == (48000, 2)
    # 1.0 + 0.4 + 0.5 = 1.9 с ровно (тишина = round(0.4 × 48000) кадров).
    assert info.duration_seconds == pytest.approx(1.9, abs=1e-9)


def test_combine_single_segment_is_byte_identical() -> None:
    single = wav_bytes(1.0)
    assert combine_wav_segments([single], []) == single


@pytest.mark.parametrize(
    "segments,pauses",
    [
        ([wav_bytes(1.0)], [0.4]),
        (
            [wav_bytes(1.0, sample_rate=48000), wav_bytes(0.5, sample_rate=24000)],
            [0.4],
        ),
        ([b"not-a-wav", wav_bytes(1.0)], [0.4]),
        ([wav_bytes(1.0), wav_bytes(0.5)], [-0.4]),
        ([], []),
    ],
)
def test_combine_invalid_inputs_are_visible(segments: list, pauses: list) -> None:
    with pytest.raises(WorkerError) as error:
        combine_wav_segments(segments, pauses)
    assert error.value.code in {"output_invalid", "output_empty", "audio_empty_result"}


# --- Движок: DSP в _postprocess и effective rate -----------------------------


def test_postprocess_applies_prosody_dsp_in_order(monkeypatch) -> None:
    calls: list[tuple] = []

    monkeypatch.setattr(
        engine_module, "_whisper_filter", lambda pcm, *, sample_rate: calls.append(("whisper", sample_rate)) or "w"
    )
    monkeypatch.setattr(
        engine_module,
        "_pitch_shift",
        lambda pcm, *, sample_rate, semitones: calls.append(("pitch", sample_rate, semitones)) or "p",
    )
    monkeypatch.setattr(
        engine_module, "_time_stretch", lambda pcm, *, rate: calls.append(("stretch", rate)) or "s"
    )
    monkeypatch.setattr(
        engine_module,
        "_resample",
        lambda pcm, *, orig_sr, target_sr: calls.append(("resample", orig_sr, target_sr)) or "r",
    )
    monkeypatch.setattr(
        engine_module, "_apply_gain", lambda pcm, *, gain_db: calls.append(("gain", gain_db)) or "g"
    )
    monkeypatch.setattr(engine_module, "_to_channels", lambda pcm, *, channels: "c")
    monkeypatch.setattr(
        engine_module, "_encode_wav", lambda pcm, *, sample_rate: calls.append(("encode", sample_rate)) or b"wav"
    )

    audio = engine_module._postprocess(
        object(),
        source_rate=24000,
        target_rate=48000,
        channels=2,
        speed=1.5,
        gain_db=1.0,
        pitch_semitones=1.0,
        whisper=True,
    )
    assert audio == b"wav"
    assert calls == [
        ("whisper", 24000),
        ("pitch", 24000, 1.0),
        ("stretch", 1.5),
        ("resample", 24000, 48000),
        ("gain", 1.0),
        ("encode", 48000),
    ]


def test_postprocess_skips_pitch_and_whisper_when_off(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(engine_module, "_whisper_filter", lambda pcm, *, sample_rate: calls.append("whisper"))
    monkeypatch.setattr(
        engine_module, "_pitch_shift", lambda pcm, *, sample_rate, semitones: calls.append("pitch")
    )
    monkeypatch.setattr(engine_module, "_apply_gain", lambda pcm, *, gain_db: calls.append("gain"))
    monkeypatch.setattr(engine_module, "_to_channels", lambda pcm, *, channels: "mono")
    monkeypatch.setattr(engine_module, "_encode_wav", lambda pcm, *, sample_rate: b"wav")

    assert (
        engine_module._postprocess(
            object(),
            source_rate=24000,
            target_rate=24000,
            channels=1,
            speed=1.0,
            gain_db=0.0,
            pitch_semitones=0.0,
            whisper=False,
        )
        == b"wav"
    )
    assert calls == []


def test_engine_synthesize_uses_effective_rate_and_prosody(tmp_path, monkeypatch) -> None:
    from tts_worker.prosody import ProsodyParams

    engine = Qwen3TtsEngine(tmp_path / "qwen3-tts-1.7b", "qwen3-tts@fake")
    monkeypatch.setattr(engine, "_load", lambda: object())
    monkeypatch.setattr(
        Qwen3TtsEngine,
        "_generate_clone",
        lambda self, model, *, text, language, reference, seed: (REFERENCE_PCM, 24000),
    )
    postprocess_calls: list[dict] = []
    monkeypatch.setattr(
        engine_module,
        "_postprocess",
        lambda pcm, **kwargs: postprocess_calls.append(kwargs) or b"wav",
    )
    prosody = ProsodyParams(
        style_id="confident",
        style_catalog_version="voice-styles@2026-11-18",
        rate=1.2,
        gain_db=1.5,
        pitch_semitones=0.0,
        sentence_pause_ms=80,
        whisper=True,
    )
    from tts_worker.reference import ReferenceAudio

    reference = ReferenceAudio(
        pcm=REFERENCE_PCM,
        sample_rate=24000,
        text="Проверка голоса для клонирования.",
        language="ru",
        sha256=REFERENCE_SHA,
        duration_seconds=3.0,
    )
    takes = engine.synthesize(
        "фраза",
        language="ru",
        voice={"mode": "clone"},
        speed=1.25,
        seed=None,
        takes=1,
        sample_rate=24000,
        channels=1,
        reference=reference,
        prosody=prosody,
    )
    assert len(takes) == 1
    call = postprocess_calls[0]
    assert call["speed"] == pytest.approx(1.5)  # 1.25 × 1.2, один time_stretch
    assert call["gain_db"] == 1.5
    assert call["pitch_semitones"] == 0.0
    assert call["whisper"] is True


def test_engine_synthesize_rejects_effective_rate_out_of_range(tmp_path, monkeypatch) -> None:
    from tts_worker.prosody import ProsodyParams

    engine = Qwen3TtsEngine(tmp_path / "qwen3-tts-1.7b", "qwen3-tts@fake")
    proton = ProsodyParams(
        style_id="fast",
        style_catalog_version="voice-styles@2026-11-18",
        rate=2.0,
        gain_db=0.0,
        pitch_semitones=0.0,
        sentence_pause_ms=0,
        whisper=False,
    )
    with pytest.raises(WorkerError) as error:
        engine.synthesize(
            "фраза",
            language="ru",
            voice={"mode": "clone"},
            speed=1.5,
            seed=None,
            takes=1,
            sample_rate=24000,
            channels=1,
            prosody=proton,
        )
    assert error.value.code == "prosody_rate"
    assert engine._model is None, "ошибка просодии — до загрузки модели"


# --- Handler: сегменты, слова, телеметрия ------------------------------------


def _prosody_job(**overrides) -> dict:
    payload = {
        "text": "Звон+ит [[pause:0.4]] телеф+он? Следов нет.",
        "takes": 1,
        "speed": 1.0,
        "prosody": _prosody(),
    }
    payload.update(overrides)
    return _job(**payload)


def _segment_engine() -> FakeEngine:
    def takes_for(text: str, **kwargs):
        duration = {"Звони́т": 1.0, "телефо́н?": 0.5, "Следов нет.": 2.0}.get(text, 1.0)
        return [
            SynthesisTake(audio=wav_bytes(duration), seed=kwargs.get("seed"))
            for _ in range(int(kwargs.get("takes", 1)))
        ]

    return FakeEngine(takes=takes_for)


def test_handler_prosody_segments_words_and_telemetry(setup) -> None:
    engine = _segment_engine()
    aligner = FakeAligner()
    handler = _handler(setup, engine=engine, aligner=aligner)
    result = _run(handler(_prosody_job()))

    # 3 сегмента (пауза + конец предложения), один синтез на сегмент.
    assert [call["text"] for call in engine.calls] == [
        "Звони\u0301т",
        "телефо\u0301н?",
        "Следов нет.",
    ]
    assert all(call["takes"] == 1 for call in engine.calls)
    assert all(call["prosody"].style_id == "confident" for call in engine.calls)
    assert len(aligner.calls) == 3

    take = result["takes"][0]
    # 1.0 + 0.4 (метка) + 0.5 + 0.08 (конец предложения) + 2.0 = 3.98
    assert take["duration_seconds"] == pytest.approx(3.98, abs=1e-6)
    assert (take["sample_rate"], take["channels"]) == (48000, 2)
    # Слова смещены на фактические длительности+паузы предыдущих сегментов.
    starts = [(word["text"], word["start_ms"]) for word in take["words"]]
    assert starts == [
        ("Звони\u0301т", 40),
        ("телефо\u0301н?", 1440),  # 1000 + 400
        ("Следов", 2020),  # +500 +80
        ("нет.", 2180),
    ]
    assert take["wer"] == 0.0

    telemetry = result["telemetry"]
    assert telemetry["normalization"] == {
        "version": "ru-normalization@2026-11-18",
        "applied": ["pause:0.4", "stress:Звонит", "stress:телефон"],
    }
    assert telemetry["stress_marks"] == 2
    assert telemetry["pauses"] == [0.4]
    assert telemetry["prosody"] == {
        "style_id": "confident",
        "effective_rate": 1.2,
        "gain_db": 1.0,
        "pitch_semitones": 0.0,
        "sentence_pause_ms": 80,
        "whisper": False,
    }
    assert result["request_id"] == REQUEST_ID

    from jsonschema import Draft202012Validator

    Draft202012Validator(load_json(contract_dir() / "output.schema.json")).validate(result)


def test_handler_prosody_effective_rate_is_speed_times_rate(setup) -> None:
    handler = _handler(setup, engine=_segment_engine())
    result = _run(handler(_prosody_job(speed=1.5, prosody=_prosody(rate=1.2))))
    assert result["telemetry"]["prosody"]["effective_rate"] == pytest.approx(1.8)
    assert result["telemetry"]["prosody"]["style_id"] == "confident"
    assert result["takes"]


def test_handler_prosody_rate_error_before_engine_and_aligner(setup) -> None:
    engine = FakeEngine()
    aligner = FakeAligner()
    handler = _handler(setup, engine=engine, aligner=aligner)
    with pytest.raises(WorkerError) as error:
        _run(handler(_prosody_job(speed=2.0, prosody=_prosody(rate=2.0))))
    assert error.value.code == "prosody_rate"
    assert engine.calls == []
    assert aligner.calls == []


def test_handler_stress_invalid_before_engine(setup) -> None:
    engine = FakeEngine()
    handler = _handler(setup, engine=engine)
    with pytest.raises(WorkerError) as error:
        _run(handler(_job(text="+ёлка")))
    assert error.value.code == "stress_invalid"
    assert engine.calls == []


def test_handler_pause_invalid_before_engine(setup) -> None:
    engine = FakeEngine()
    handler = _handler(setup, engine=engine)
    with pytest.raises(WorkerError) as error:
        _run(handler(_job(text="до [[pause:0.4]]")))
    assert error.value.code == "pause_invalid"
    assert engine.calls == []


def test_handler_markup_without_prosody_has_no_prosody_telemetry(setup) -> None:
    handler = _handler(setup, engine=_segment_engine())
    result = _run(handler(_job(text="до [[pause:0.4]] после")))
    telemetry = result["telemetry"]
    assert "prosody" not in telemetry
    assert telemetry["pauses"] == [0.4]
    assert telemetry["stress_marks"] == 0
    assert telemetry["normalization"]["version"] == "ru-normalization@2026-11-18"
    assert telemetry["normalization"]["applied"] == ["pause:0.4"]
    # Пауза без стиля всё равно материализуется: 1.0 + 0.4 + 1.0 = 2.4 с.
    assert result["takes"][0]["duration_seconds"] == pytest.approx(2.4, abs=1e-6)


def test_handler_takes_combined_per_index_across_segments(setup) -> None:
    engine = _segment_engine()
    handler = _handler(setup, engine=engine)
    result = _run(handler(_prosody_job(takes=2)))
    assert len(result["takes"]) == 2
    assert [take["take"] for take in result["takes"]] == [1, 2]
    for take in result["takes"]:
        assert take["duration_seconds"] == pytest.approx(3.98, abs=1e-6)
    assert len(engine.calls) == 3  # по одному синтезу на сегмент (takes внутри)


def test_handler_wer_is_word_weighted_mean(setup) -> None:
    aligner = _PerSegmentAligner(
        {"Звони\u0301т": 0.0, "телефо\u0301н?": 0.5, "Следов нет.": 1.0},
        {"Звони\u0301т": 1, "телефо\u0301н?": 1, "Следов нет.": 3},
    )
    handler = _handler(setup, engine=_segment_engine(), aligner=aligner)
    result = _run(handler(_prosody_job()))
    # (0*1 + 0.5*1 + 1.0*3) / 5 = 0.7 — на длинном сегменте вес больше.
    assert result["takes"][0]["wer"] == pytest.approx(0.7)


def test_handler_null_segment_wer_aggregates_to_null(setup) -> None:
    handler = _handler(setup, engine=_segment_engine(), aligner=FakeAligner(wer=None))
    result = _run(handler(_prosody_job()))
    take = result["takes"][0]
    assert take["wer"] is None
    assert len(take["words"]) == 4  # слова по-прежнему факт по сегментам


def test_handler_segment_takes_mismatch_is_visible(setup) -> None:
    def takes_for(text: str, **kwargs):
        count = 1 if text == "телефо\u0301н?" else int(kwargs.get("takes", 1))
        return [SynthesisTake(audio=wav_bytes(0.5), seed=kwargs.get("seed")) for _ in range(count)]

    handler = _handler(setup, engine=FakeEngine(takes=takes_for))
    with pytest.raises(WorkerError) as error:
        _run(handler(_prosody_job(takes=2)))
    assert error.value.code == "takes_mismatch"


def test_handler_clone_aligner_disabled_words_empty_with_pauses(setup) -> None:
    downloader = FakeReferenceDownloader({REFERENCE_URL: REFERENCE_BYTES})
    decoder = FakeReferenceDecoder(pcm=REFERENCE_PCM, duration_seconds=3.0)
    config, manifest, _models_root = setup
    handler = build_handler(
        config,
        manifest,
        _segment_engine(),
        FakeAligner(),
        _normalizer(),
        clock=FakeClock([10.0, 11.7]),
        reference_downloader=downloader,
        reference_decoder=decoder,
    )
    result = _run(
        handler(
            _clone_job(
                text="до [[pause:0.4]] после",
                aligner={"enabled": False, "word_timestamps": True},
            )
        )
    )
    take = result["takes"][0]
    assert take["words"] == []
    assert take["wer"] is None
    assert result["telemetry"]["aligner_version"] is None
    assert result["telemetry"]["voice_mode"] == "clone"
    assert result["telemetry"]["pauses"] == [0.4]
    # 1.0 + 0.4 + 1.0 = 2.4: тишина реально в файле, а не только в телеметрии.
    assert take["duration_seconds"] == pytest.approx(2.4, abs=1e-6)


# --- Контракт v1.2 -----------------------------------------------------------


def test_contract_prosody_examples_are_valid() -> None:
    from jsonschema import Draft202012Validator

    input_validator = Draft202012Validator(
        load_json(contract_dir() / "input.schema.json")
    )
    output_validator = Draft202012Validator(
        load_json(contract_dir() / "output.schema.json")
    )
    input_validator.validate(load_json(contract_dir() / "examples" / "request-prosody.json"))
    output_validator.validate(load_json(contract_dir() / "examples" / "response-prosody.json"))


def test_contract_prosody_is_optional_and_strict() -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    validator = Draft202012Validator(load_json(contract_dir() / "input.schema.json"))
    payload = copy.deepcopy(load_json(contract_dir() / "examples" / "request.json"))
    payload["input"].pop("prosody", None)
    validator.validate(payload)  # старый v1 input без prosody валиден

    with pytest.raises(ValidationError):
        validator.validate(
            {
                "input": {
                    **copy.deepcopy(payload["input"]),
                    "prosody": {**_prosody(), "extra": 1},
                }
            }
        )
    with pytest.raises(ValidationError):
        validator.validate(
            {
                "input": {
                    **copy.deepcopy(payload["input"]),
                    "prosody": {**_prosody(), "rate": 2.5},
                }
            }
        )
