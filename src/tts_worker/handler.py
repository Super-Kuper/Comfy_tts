"""RunPod handler контракта `tts-serverless/v1.2` (worker 0.4.0).

Поток: валидация job/input по схеме → разрешение авторской просодии (``prosody``
из сервиса, ``+``/``[[pause]]`` в тексте) → pinned-модели на томе
(engine/aligner, ``model_file_missing`` / ``provision_hash_mismatch``) → голос:
для ``voice.mode=clone`` референс скачивается по presigned URL, проверяется по
sha256/размеру/длительности и декодируется в PCM 24 кГц mono
(``reference_*``); для preset — как раньше → нормализация текста по словарю
(авторские ударения приоритетнее, паузы защищены) → сегментация по
``[[pause]]`` и ``sentence_pause_ms`` → синтез Qwen3-TTS Base каждого сегмента
тем же голосом/референсом/дублями/seed → точная склейка дубля #i с тишиной
(``dsp.combine_wav_segments``) → факты WAV из заголовка → GigaAM word
timestamps/WER по сегментам со смещением на фактические длительности и паузы →
телеметрия из исполнения (``voice_mode``/``reference_hash``/``normalization``/
``stress_marks``/``pauses``/``prosody`` — аддитивно) → доставка base64/URL →
финальный гейт output-схемой.

Воркер **не выбирает модель и стиль**: модель — свойство образа/тома, стиль
разрешает сервис в явные DSP-параметры; поля выбора нет. Внешние зависимости
инъектируются (``build_handler``/``Dependencies``): тесты идут без GPU, сети,
torch и runpod.

Ссылки: `contract/worker-contract.md` §1/§2/§4.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aligner import Alignment, WordAligner, WordTiming
from .audio import read_audio_info
from .config import Config
from .contract import validate_input, validate_output
from .dsp import combine_wav_segments, effective_rate
from .engine import SynthesisTake, TtsEngine
from .errors import WorkerError
from .health import health_payload
from .manifest import ModelFile, ModelManifest, sha256_file
from .normalization import Normalizer
from .prosody import ProsodyParams, build_segments, parse_prosody
from .reference import (
    AudioDecoder,
    HttpxReferenceDownloader,
    ReferenceAudio,
    ReferenceDownloader,
    SoundFileDecoder,
    prepare_reference,
)
from .storage import AudioStorage, S3Storage, deliver_audio

_LOG = logging.getLogger(__name__)

ModelHasher = Callable[[Path], str]
Clock = Callable[[], float]


@dataclass
class Dependencies:
    """Всё внешнее, что нужно handler'у (инъекция для тестов)."""

    config: Config
    manifest: ModelManifest
    engine: TtsEngine
    aligner: WordAligner
    normalizer: Normalizer
    storage: AudioStorage | None = None
    model_hasher: ModelHasher = sha256_file
    clock: Clock = time.monotonic
    reference_downloader: ReferenceDownloader | None = None
    reference_decoder: AudioDecoder | None = None


def _job_id(job: Any) -> str:
    if isinstance(job, Mapping) and job.get("id") is not None:
        return str(job.get("id"))
    return "<без id>"


def _validate_job(job: Any) -> Mapping[str, Any]:
    if not isinstance(job, Mapping):
        raise WorkerError("input_invalid", "job должен быть объектом RunPod")
    payload = job.get("input")
    if not isinstance(payload, Mapping):
        raise WorkerError("input_invalid", "job.input отсутствует или не является объектом")
    return payload


def _measure_model(entry: ModelFile, deps: Dependencies) -> str:
    """sha256 файла из манифеста, сверенный с pinned-значением (§7.3 плана)."""

    path = deps.config.models_root / entry.relative_path
    if not path.is_file():
        raise WorkerError(
            "model_file_missing",
            f"{path} нет на томе: прогнать populate-volume.sh для {entry.path}",
        )
    actual = deps.model_hasher(path)
    if actual != entry.sha256:
        raise WorkerError(
            "provision_hash_mismatch",
            f"{entry.path}: sha256 {actual} != pinned {entry.sha256}",
        )
    return actual


def _words_payload(words: Sequence[WordTiming]) -> list[dict[str, Any]]:
    """Слова GigaAM — факт; нарушенный порядок/интервалы не «чиним» молча."""

    payload: list[dict[str, Any]] = []
    for word in words:
        if word.start_ms < 0 or word.end_ms <= word.start_ms:
            raise WorkerError(
                "output_invalid",
                f"aligner: слово {word.text!r} с интервалом {word.start_ms}..{word.end_ms} мс",
            )
        payload.append(
            {"text": word.text, "start_ms": word.start_ms, "end_ms": word.end_ms}
        )
    return payload


def _offset_words(alignment: Alignment, offset_ms: int) -> list[WordTiming]:
    """Слова сегмента, смещённые на фактические длительности+паузы слева."""

    words: list[WordTiming] = []
    for word in alignment.words:
        if word.start_ms < 0 or word.end_ms <= word.start_ms:
            raise WorkerError(
                "output_invalid",
                f"aligner: слово {word.text!r} с интервалом {word.start_ms}..{word.end_ms} мс",
            )
        words.append(
            WordTiming(
                text=word.text,
                start_ms=word.start_ms + offset_ms,
                end_ms=word.end_ms + offset_ms,
            )
        )
    return words


def _aggregate_wer(
    alignments: Sequence[Alignment], segment_texts: Sequence[str]
) -> float | None:
    """WER склейки: **взвешенное среднее по числу слов сегментов**.

    Каждый сегмент даёт свой WER от GigaAM; объединённый WER — сумма
    ``wer_i × words_i / Σ words_i``, где ``words_i`` — число слов выравнивания
    (fallback — число токенов текста сегмента). Так длинный сегмент не теряется
    за коротким; при ``None`` у всех сегментов WER = ``None``. Значение вне
    [0, 1] у любого сегмента — видимый ``output_invalid``, не тихая обрезка.
    """

    weighted: list[tuple[float, int]] = []
    for alignment, segment_text in zip(alignments, segment_texts):
        if alignment.wer is None:
            continue
        wer = float(alignment.wer)
        if not 0.0 <= wer <= 1.0:
            raise WorkerError("output_invalid", f"aligner: WER {wer} вне [0, 1]")
        weight = len(alignment.words) or len(segment_text.split())
        weighted.append((wer, max(weight, 1)))
    if not weighted:
        return None
    total_weight = sum(weight for _wer, weight in weighted)
    return round(sum(wer * weight for wer, weight in weighted) / total_weight, 4)


def _synthesize_segments(
    deps: Dependencies,
    segments: Sequence[Any],
    *,
    language: str,
    voice: Mapping[str, Any],
    speed: float,
    seed: int | None,
    takes: int,
    sample_rate: int,
    channels: int,
    reference: ReferenceAudio | None,
    prosody: ProsodyParams | None,
) -> list[list[SynthesisTake]]:
    """Синтез каждого сегмента **тем же** голосом/референсом/дублями/seed."""

    per_segment: list[list[SynthesisTake]] = []
    for index, segment in enumerate(segments, start=1):
        result = deps.engine.synthesize(
            segment.text,
            language=language,
            voice=dict(voice),
            speed=speed,
            seed=seed,
            takes=takes,
            sample_rate=sample_rate,
            channels=channels,
            reference=reference,
            prosody=prosody,
        )
        if not result:
            raise WorkerError(
                "audio_empty_result", f"движок вернул 0 дублей на сегменте {index}"
            )
        if len(result) != takes:
            raise WorkerError(
                "takes_mismatch",
                f"сегмент {index}: запрошено дублей {takes}, движок вернул {len(result)}",
            )
        per_segment.append(result)
    return per_segment


async def execute(job: Any, deps: Dependencies) -> dict[str, Any]:
    """Выполнить один job; ошибки — ``WorkerError`` наружу (job FAILED)."""

    payload = _validate_job(job)
    # Схема описывает тело job'а: {"input": ..., "policy": ...}; RunPod добавляет
    # к нему ``id`` (top-level additionalProperties: true).
    validate_input(job, deps.config.contract_dir)
    inp = payload

    if payload.get("action") == "health":
        return health_payload(deps.config, deps.manifest)

    text = inp.get("text")
    if not isinstance(text, str) or not text:
        raise WorkerError("input_invalid", "input.text отсутствует")
    if len(text) > deps.config.max_text_chars:
        raise WorkerError(
            "input_invalid",
            f"input.text: {len(text)} > TTS_MAX_TEXT_CHARS={deps.config.max_text_chars}",
        )
    language = str(inp.get("language"))
    voice = inp.get("voice")
    if not isinstance(voice, Mapping):
        raise WorkerError("input_invalid", "input.voice отсутствует")
    render = inp.get("render")
    if not isinstance(render, Mapping):
        raise WorkerError("input_invalid", "input.render отсутствует")
    aligner_config: dict[str, Any] = {"enabled": True, "word_timestamps": True}
    raw_aligner = inp.get("aligner")
    if isinstance(raw_aligner, Mapping):
        aligner_config.update(raw_aligner)
    aligner_enabled = bool(aligner_config["enabled"])
    want_words = bool(aligner_config["word_timestamps"])
    takes_requested = int(inp.get("takes") or 1)
    seed = inp.get("seed")
    speed = float(inp.get("speed") or 1.0)
    sample_rate = int(render["sample_rate"])
    channels = int(render["channels"])

    # v1.2: явные DSP-параметры стиля (сервис уже разрешил стиль). Отсутствие
    # ``prosody`` — старый путь v1 без изменений. Ошибки просодии/разметки
    # видимы и возникают до траты GPU.
    prosody = parse_prosody(inp.get("prosody"))
    if prosody is not None:
        effective_speed = effective_rate(speed, prosody.rate)
    else:
        effective_speed = speed

    # Нормализация + разбор авторской разметки до загрузки модели/скачивания
    # референса: битый `+`/`[[pause]]` не должен стоить GPU или сети.
    normalized = deps.normalizer(text, language)
    if not normalized.text.strip():
        raise WorkerError("text_empty", "после нормализации текст пуст")
    segments, marker_pauses, _markup_applied = build_segments(
        normalized.text,
        sentence_pause_ms=prosody.sentence_pause_ms if prosody is not None else 0,
    )
    pauses_between = [segment.pause_after_seconds for segment in segments[:-1]]

    engine_entry = deps.manifest.entry("engine")
    engine_sha256 = _measure_model(engine_entry, deps)
    aligner_entry = deps.manifest.entry("aligner")
    if aligner_enabled:
        _measure_model(aligner_entry, deps)

    # Голос: Base умеет только клонирование. Референс скачивается/проверяется
    # здесь (видимые reference_*), движку уходит уже PCM 24 кГц mono; URL и
    # содержимое референса в лог не попадают — только sha256.
    voice_mode = str(voice.get("mode") or "")
    reference: ReferenceAudio | None = None
    if voice_mode == "clone":
        reference = await prepare_reference(
            voice,
            downloader=deps.reference_downloader or HttpxReferenceDownloader(),
            decoder=deps.reference_decoder or SoundFileDecoder(),
        )
    elif voice_mode != "preset":
        # Схема уже ограничивает mode, но defensive-код остаётся видимым:
        # незнакомый режим не должен тихо превращаться в пресет.
        raise WorkerError(
            "voice_mode_unsupported",
            f"voice.mode={voice_mode!r}: ожидалось preset|clone",
        )

    _LOG.info(
        "job %s: takes=%d, language=%s, voice_mode=%s, reference_hash=%s, "
        "aligner=%s, сегментов=%d, паузы=%s, stress_marks=%d, "
        "normalization=%s, prosody=%s",
        _job_id(job),
        takes_requested,
        language,
        voice_mode,
        reference.sha256 if reference is not None else None,
        aligner_enabled,
        len(segments),
        marker_pauses,
        normalized.stress_marks,
        list(normalized.applied),
        prosody.style_id if prosody is not None else None,
    )

    started = deps.clock()
    segment_takes = _synthesize_segments(
        deps,
        segments,
        language=language,
        voice=voice,
        speed=speed,
        seed=seed,
        takes=takes_requested,
        sample_rate=sample_rate,
        channels=channels,
        reference=reference,
        prosody=prosody,
    )

    out_takes: list[dict[str, Any]] = []
    for index in range(takes_requested):
        # Дубль #i = сегмент #1 take i + точная тишина + сегмент #2 take i ...
        combined = combine_wav_segments(
            [segment_takes[segment][index].audio for segment in range(len(segments))],
            pauses_between,
        )
        if not combined:
            raise WorkerError("output_empty", f"дубль {index + 1}: 0 байт аудио")
        info = read_audio_info(combined)
        take_seed = next(
            (
                segment_takes[segment][index].seed
                for segment in range(len(segments))
                if segment_takes[segment][index].seed is not None
            ),
            seed,
        )
        words: list[dict[str, Any]] = []
        wer: float | None = None
        if aligner_enabled:
            alignments: list[Alignment] = []
            offset_ms = 0
            collected: list[WordTiming] = []
            for segment_index, segment in enumerate(segments):
                segment_audio = segment_takes[segment_index][index].audio
                alignment = deps.aligner.word_timestamps(
                    segment_audio, segment.text, language=language
                )
                alignments.append(alignment)
                if want_words:
                    collected.extend(_offset_words(alignment, offset_ms))
                segment_info = read_audio_info(segment_audio)
                offset_ms += int(round(segment_info.duration_seconds * 1000))
                offset_ms += int(round(segment.pause_after_seconds * 1000))
            if want_words:
                words = _words_payload(collected)
            wer = _aggregate_wer(alignments, [segment.text for segment in segments])
        delivery = await deliver_audio(combined, info, deps.config, deps.storage)
        out_takes.append(
            {
                "take": index + 1,
                **delivery,
                "duration_seconds": round(info.duration_seconds, 3),
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "seed": take_seed,
                "words": words,
                "wer": wer,
            }
        )

    gpu_seconds = max(round(deps.clock() - started, 3), 0.001)
    telemetry: dict[str, Any] = {
        "engine": deps.engine.engine_id,
        "engine_version": deps.engine.engine_version,
        "aligner_version": deps.aligner.aligner_version if aligner_enabled else None,
        "model_hash": f"sha256:{engine_sha256}",
        "gpu_seconds": gpu_seconds,
        "license": engine_entry.license,
        # Аддитивные факты голоса (contract §2.1): режим исполнения и sha256
        # референса (та же форма, что ``media_items.content_hash``; для preset —
        # null). Значение — проверенный sha256, а не поле запроса.
        "voice_mode": voice_mode,
        "reference_hash": reference.sha256 if reference is not None else None,
        # v1.2: факты авторской разметки и нормализатора (аддитивно).
        "normalization": {
            "version": normalized.version,
            "applied": list(normalized.applied),
        },
        "stress_marks": int(normalized.stress_marks),
        "pauses": [float(pause) for pause in marker_pauses],
    }
    if prosody is not None:
        telemetry["prosody"] = {
            "style_id": prosody.style_id,
            "effective_rate": effective_speed,
            "gain_db": float(prosody.gain_db),
            "pitch_semitones": float(prosody.pitch_semitones),
            "sentence_pause_ms": int(prosody.sentence_pause_ms),
            "whisper": bool(prosody.whisper),
        }
    result: dict[str, Any] = {
        "request_id": inp["request_id"],
        "takes": out_takes,
        "telemetry": telemetry,
    }
    validate_output(result, deps.config.contract_dir)
    _LOG.info(
        "job %s: готово, дублей=%d, words=%s, wer=%s, %.2f с",
        _job_id(job),
        len(out_takes),
        sum(len(take["words"]) for take in out_takes),
        [take["wer"] for take in out_takes],
        gpu_seconds,
    )
    return result


def build_handler(
    config: Config,
    manifest: ModelManifest,
    engine: TtsEngine,
    aligner: WordAligner,
    normalizer: Normalizer,
    storage: AudioStorage | None = None,
    *,
    model_hasher: ModelHasher = sha256_file,
    clock: Clock = time.monotonic,
    reference_downloader: ReferenceDownloader | None = None,
    reference_decoder: AudioDecoder | None = None,
) -> Callable[[Any], Awaitable[dict[str, Any]]]:
    """Собрать handler с зафиксированными зависимостями (для тестов и main)."""

    deps = Dependencies(
        config=config,
        manifest=manifest,
        engine=engine,
        aligner=aligner,
        normalizer=normalizer,
        storage=storage,
        model_hasher=model_hasher,
        clock=clock,
        reference_downloader=reference_downloader,
        reference_decoder=reference_decoder,
    )

    async def handler(job: Any) -> dict[str, Any]:
        return await execute(job, deps)

    return handler


# Ленивый дефолт для RunPod: config/manifest/engine создаются при первом job,
# чтобы импорт модуля (и тесты) не требовали GPU, том, torch и S3.
_default_deps: Dependencies | None = None


def default_dependencies() -> Dependencies:
    """Зависимости по умолчанию из env (том, манифест, Qwen3-TTS, GigaAM-v3)."""

    global _default_deps
    if _default_deps is None:
        from .aligner import GigaamAligner
        from .engine import Qwen3TtsEngine
        from .normalization import DictionaryNormalizer

        config = Config.from_env()
        manifest = ModelManifest.load(config.manifest_path)
        engine_entry = manifest.entry("engine")
        aligner_entry = manifest.entry("aligner")
        engine = Qwen3TtsEngine(
            config.models_root / engine_entry.relative_path.parent,
            engine_entry.version,
            device=config.device,
        )
        aligner = GigaamAligner(
            config.models_root / aligner_entry.relative_path.parent,
            aligner_entry.version,
            device=config.device,
        )
        normalizer = DictionaryNormalizer.from_path(config.dictionaries_path)
        storage = S3Storage(config) if config.storage_configured else None
        _default_deps = Dependencies(
            config=config,
            manifest=manifest,
            engine=engine,
            aligner=aligner,
            normalizer=normalizer,
            storage=storage,
            reference_downloader=HttpxReferenceDownloader(),
            reference_decoder=SoundFileDecoder(),
        )
    return _default_deps


async def handler(job: Any) -> dict[str, Any]:
    """Точка входа RunPod (`handler.py` в корне); тесты подменяют зависимости."""

    return await execute(job, default_dependencies())


__all__ = [
    "Dependencies",
    "build_handler",
    "default_dependencies",
    "execute",
    "handler",
]
