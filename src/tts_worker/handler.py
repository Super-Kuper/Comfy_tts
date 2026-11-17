"""RunPod handler контракта `tts-serverless/v1`.

Поток: валидация job/input по схеме → pinned-модели на томе (engine/aligner,
``model_file_missing`` / ``provision_hash_mismatch``) → нормализация текста по
словарю → синтез Qwen3-TTS (1–2 дубля) → факты WAV из заголовка → GigaAM
word timestamps/WER → телеметрия из исполнения → доставка base64/URL →
финальный гейт output-схемой.

Воркер **не выбирает модель**: модель — свойство образа/тома, поля выбора в
контракте нет; ``engine_version``/``aligner_version`` — факты исполнения.
Внешние зависимости инъектируются (``build_handler``/``Dependencies``): тесты
идут без GPU, сети, torch и runpod.

Ссылки: `contract/worker-contract.md` §1/§2/§3.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aligner import Alignment, WordAligner
from .audio import read_audio_info
from .config import Config
from .contract import validate_input, validate_output
from .engine import SynthesisTake, TtsEngine
from .errors import WorkerError
from .manifest import ModelFile, ModelManifest, sha256_file
from .normalization import Normalizer
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


def _words_payload(alignment: Alignment) -> list[dict[str, Any]]:
    """Слова GigaAM — факт; нарушенный порядок/интервалы не «чиним» молча."""

    words: list[dict[str, Any]] = []
    for word in alignment.words:
        if word.start_ms < 0 or word.end_ms <= word.start_ms:
            raise WorkerError(
                "output_invalid",
                f"aligner: слово {word.text!r} с интервалом {word.start_ms}..{word.end_ms} мс",
            )
        words.append({"text": word.text, "start_ms": word.start_ms, "end_ms": word.end_ms})
    return words


async def execute(job: Any, deps: Dependencies) -> dict[str, Any]:
    """Выполнить один job; ошибки — ``WorkerError`` наружу (job FAILED)."""

    payload = _validate_job(job)
    # Схема описывает тело job'а: {"input": ..., "policy": ...}; RunPod добавляет
    # к нему ``id`` (top-level additionalProperties: true).
    validate_input(job, deps.config.contract_dir)
    inp = payload

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

    engine_entry = deps.manifest.entry("engine")
    engine_sha256 = _measure_model(engine_entry, deps)
    aligner_entry = deps.manifest.entry("aligner")
    if aligner_enabled:
        _measure_model(aligner_entry, deps)

    normalized = deps.normalizer(text, language)
    if not normalized.text.strip():
        raise WorkerError("text_empty", "после нормализации текст пуст")
    _LOG.info(
        "job %s: takes=%d, language=%s, aligner=%s, нормализация=%s",
        _job_id(job),
        takes_requested,
        language,
        aligner_enabled,
        list(normalized.applied),
    )

    started = deps.clock()
    synthesis: list[SynthesisTake] = deps.engine.synthesize(
        normalized.text,
        language=language,
        voice=dict(voice),
        speed=speed,
        seed=seed,
        takes=takes_requested,
        sample_rate=sample_rate,
        channels=channels,
    )
    if not synthesis:
        raise WorkerError("audio_empty_result", "движок вернул 0 дублей — это не успех")
    if len(synthesis) != takes_requested:
        raise WorkerError(
            "takes_mismatch",
            f"запрошено дублей {takes_requested}, движок вернул {len(synthesis)}",
        )

    out_takes: list[dict[str, Any]] = []
    for index, take in enumerate(synthesis, start=1):
        if not take.audio:
            raise WorkerError("output_empty", f"дубль {index}: 0 байт аудио")
        info = read_audio_info(take.audio)
        words: list[dict[str, Any]] = []
        wer: float | None = None
        if aligner_enabled:
            alignment = deps.aligner.word_timestamps(
                take.audio, normalized.text, language=language
            )
            if want_words:
                words = _words_payload(alignment)
            if alignment.wer is not None:
                wer = round(float(alignment.wer), 4)
                if not 0.0 <= wer <= 1.0:
                    raise WorkerError("output_invalid", f"aligner: WER {wer} вне [0, 1]")
        delivery = await deliver_audio(take.audio, info, deps.config, deps.storage)
        out_takes.append(
            {
                "take": index,
                **delivery,
                "duration_seconds": round(info.duration_seconds, 3),
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "seed": take.seed if take.seed is not None else seed,
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
