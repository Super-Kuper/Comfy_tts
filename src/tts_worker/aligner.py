"""GigaAM-v3 aligner/QC hook: word timestamps + WER (контракт §2).

``words`` и ``wer`` в ответе воркера — из GigaAM, **не** из TTS-модели: у
Qwen3-TTS нет нативных пословных таймингов (proposal tts-ru.md, «Тайминги»).
Живой код ASR/forced alignment не подключён (``TODO(live)``); hook
инъектируется и мокается в тестах, тяжёлые импорты ленивые.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import WorkerError


@dataclass(frozen=True)
class WordTiming:
    """Слово с абсолютными миллисекундами от начала дубля."""

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Alignment:
    """Результат выравнивания: слова и WER против нормализованной фразы."""

    words: tuple[WordTiming, ...]
    wer: float | None


class WordAligner(Protocol):
    """Контракт aligner'а: байты WAV + фраза → слова и WER."""

    aligner_version: str

    def word_timestamps(self, audio: bytes, text: str, *, language: str) -> Alignment:
        ...


class GigaamAligner:
    """GigaAM-v3 (MIT) c CTC-выравниванием; callable-граница для QC.

    ``aligner_version`` — из pinned-манифеста (``gigaam-v3@2026-04``); после
    реальной загрузки её можно уточнить из конфига модели (факт исполнения).
    """

    def __init__(self, model_dir: Path, aligner_version: str, *, device: str = "cuda") -> None:
        self._model_dir = Path(model_dir)
        self.aligner_version = aligner_version
        self._device = device

    def word_timestamps(self, audio: bytes, text: str, *, language: str) -> Alignment:
        # TODO(live): подключить pinned GigaAM-v3:
        #   1. lazi import torch + загрузка pytorch_model.bin/tokenizer.model из
        #      self._model_dir (trust_remote_code: modeling_gigaam.py) на self._device;
        #   2. ASR по WAV: гипотеза + пословные/посимвольные эмиссии CTC;
        #   3. forced alignment нормализованной фразы по эмиссиям → WordTiming
        #      (абсолютные ms от старта дубля);
        #   4. WER = edit_distance(гипотеза, нормализованный текст) / len(words);
        #   5. resample 48/32/24 kHz → частоту GigaAM, если нужно.
        # Пока кода нет — видимая ошибка, не подмена words из TTS-модели.
        raise WorkerError(
            "aligner_unavailable",
            "GigaAM-v3 выравнивание не подключено (TODO(live)): модель "
            f"{self._model_dir} на устройстве {self._device}; тесты/стенд — fake/injection",
        )


__all__ = ["Alignment", "GigaamAligner", "WordAligner", "WordTiming"]
