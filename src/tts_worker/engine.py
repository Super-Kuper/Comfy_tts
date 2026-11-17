"""Qwen3-TTS callable: граница исполнения движка (мок/инъекция для тестов).

Живой код модели **не подключён в этом репозитории** (нет torch/GPU в CI и
образ собирается без весов): ``TODO(live)`` ниже — куда вставить pinned
Qwen3-TTS. Важно, что импорты тяжёлых пакетов ленивые: воркер и тесты
запускаются без GPU и без установленного torch; отсутствие движка даёт
видимый код ``engine_unavailable``, а не ImportError и не тихий фолбэк.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .errors import WorkerError

ENGINE_ID = "qwen3_tts"


@dataclass(frozen=True)
class SynthesisTake:
    """Факт одного дубля: байты WAV и фактический seed (None — движок не сообщил)."""

    audio: bytes
    seed: int | None = None


class TtsEngine(Protocol):
    """Контракт движка: фраза + параметры → N дублей WAV."""

    engine_id: str
    engine_version: str

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: Mapping[str, Any],
        speed: float,
        seed: int | None,
        takes: int,
        sample_rate: int,
        channels: int,
    ) -> list[SynthesisTake]:
        ...


class Qwen3TtsEngine:
    """Pinned Qwen3-TTS 1.7B на томе; callable-обвязка без выбора модели.

    Модель — свойство образа/тома, а не job'а: параметра ``model`` нигде нет.
    ``engine_version`` приходит из pinned-манифеста при сборке зависимостей
    (факт загруженного образа); после реальной загрузки весов её можно уточнить
    из конфига модели — это по-прежнему факт исполнения, не запроса.
    """

    engine_id = ENGINE_ID

    def __init__(self, model_dir: Path, engine_version: str, *, device: str = "cuda") -> None:
        self._model_dir = Path(model_dir)
        self.engine_version = engine_version
        self._device = device
        self._loaded = False

    def _load(self) -> None:
        # TODO(live): подключить pinned исполнение Qwen3-TTS:
        #   1. lazi import torch + пакет Qwen3-TTS (зафиксировать версию в образе);
        #   2. загрузить веса из self._model_dir (model.safetensors + speech_tokenizer)
        #      на self._device; Native 24 kHz, 1.7B, Apache-2.0;
        #   3. прогреть модель при старте воркера (загрузка при старте — plan §7.2),
        #      чтобы первый job не платил холодный старт (3–10 мин амортизируются батчем);
        #   4. ресемплинг 24 kHz → render.sample_rate и каналы тут же на границе,
        #      в WAV PCM s16le.
        # Пока кода исполнения нет — видимая ошибка, не подмена.
        self._loaded = True
        raise WorkerError(
            "engine_unavailable",
            "Qwen3-TTS исполнение не подключено (TODO(live)): модель "
            f"{self._model_dir} на устройстве {self._device}; тесты/стенд — fake/injection",
        )

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: Mapping[str, Any],
        speed: float,
        seed: int | None,
        takes: int,
        sample_rate: int,
        channels: int,
    ) -> list[SynthesisTake]:
        self._load()  # TODO(live): после подключения — реальный синтез
        raise AssertionError("unreachable: _load() always raises until TODO(live) is done")


__all__ = ["ENGINE_ID", "Qwen3TtsEngine", "SynthesisTake", "TtsEngine"]
