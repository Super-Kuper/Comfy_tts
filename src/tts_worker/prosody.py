"""Авторская просодия фразы: параметры DSP-стиля и разметка `[[pause]]`.

Контракт ``tts-serverless/v1.2`` (аддитивно к v1, worker 0.4.0):

* необязательный вход ``prosody`` — **явные** детерминированные параметры
  (стиль уже разрешён сервисом; воркер не выбирает стиль и не знает каталога);
* в ``text`` допустимы авторские метки: ``+`` перед ударной кириллической
  гласной (обрабатывает нормализатор, см. ``normalization.py``) и
  ``[[pause:<секунды>]]`` — разбор здесь.

Паузы: 0.05–2.0 с, шаг 0.05, ≤ 6 меток на фразу, суммарно ≤ 6 с, не в
начале/конце и не подряд (пустые сегменты запрещены). Метка не «срезается»
молча: сегменты синтезируются отдельно, между ними вставляется точная тишина
(``dsp.combine_wav_segments``). ``sentence_pause_ms`` добавляет тишину после
конца предложения (``.!?…``) уже на этапе сегментации — окно фразы при этом не
подгоняется.

Стиль-тег ``[[style:<id>]]`` сервис вырезает до отправки job'а; его в воркере
нет. Неизвестные/невалидные значения — видимые коды (``prosody_invalid``,
``pause_invalid``, ``prosody_rate``); тихого clamp/fallback нет.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .dsp import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    PITCH_MAX_SEMITONES,
    PITCH_MIN_SEMITONES,
    RATE_MAX,
    RATE_MIN,
    SENTENCE_PAUSE_MAX_MS,
)
from .errors import WorkerError

#: Пределы авторских пауз (plan §3.2).
PAUSE_MIN_SECONDS = 0.05
PAUSE_MAX_SECONDS = 2.0
PAUSE_STEP_SECONDS = 0.05
MAX_PAUSES_PER_PHRASE = 6
MAX_TOTAL_PAUSE_SECONDS = 6.0

#: Концы предложений, после которых ``sentence_pause_ms`` вставляет тишину.
SENTENCE_END_CHARS = ".!?…"

#: Поля входного ``prosody`` (строгая схема входа; порядок — как в контракте).
PROSODY_FIELDS = (
    "style_id",
    "style_catalog_version",
    "rate",
    "gain_db",
    "pitch_semitones",
    "sentence_pause_ms",
    "whisper",
)
PROSODY_KEYS = frozenset(PROSODY_FIELDS)

_PAUSE_MARKER_RE = re.compile(r"\[\[pause(?P<body>[^\]]*)\]\]")
_PAUSE_BODY_RE = re.compile(r"^:(?P<value>[0-9]+(?:\.[0-9]+)?)$")
_PROTECTED_PAUSE_RE = re.compile(r"\[\[pause:[^\]]*\]\]")
_PAUSE_APPLIED_RE = re.compile(r"\[\[pause:([0-9]+(?:\.[0-9]+)?)\]\]")


@dataclass(frozen=True)
class ProsodyParams:
    """Разрешённые сервисом параметры DSP-стиля (факт входа job'а)."""

    style_id: str
    style_catalog_version: str
    rate: float
    gain_db: float
    pitch_semitones: float
    sentence_pause_ms: int
    whisper: bool


@dataclass(frozen=True)
class SpeechSegment:
    """Кусок фразы для отдельного синтеза + тишина после него."""

    text: str
    pause_after_seconds: float = 0.0


def _require_number(raw: Mapping[str, Any], field: str, minimum: float, maximum: float) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerError(
            "prosody_invalid", f"prosody.{field}: ожидалось число {minimum}–{maximum}"
        )
    number = float(value)
    if not math.isfinite(number) or not minimum - 1e-9 <= number <= maximum + 1e-9:
        raise WorkerError(
            "prosody_invalid",
            f"prosody.{field}={value!r}: ожидалось число {minimum}–{maximum}",
        )
    return number


def _require_text(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise WorkerError("prosody_invalid", f"prosody.{field}: ожидалась непустая строка")
    return value.strip()


def parse_prosody(raw: Any) -> ProsodyParams | None:
    """Разобрать и проверить ``input.prosody`` (``None`` — старый путь v1).

    Схема входа уже strict и проверяет диапазоны; эта функция — вторая линия
    защиты (и публичная точка для прямых вызовов): неизвестные поля, bool
    вместо числа, пустые style_id/version дают видимый ``prosody_invalid``.
    """

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise WorkerError("prosody_invalid", "prosody: ожидался объект")
    unknown = sorted(key for key in raw if key not in PROSODY_KEYS)
    if unknown:
        raise WorkerError("prosody_invalid", f"prosody: неизвестные поля {unknown}")
    missing = [field for field in PROSODY_FIELDS if field not in raw]
    if missing:
        raise WorkerError("prosody_invalid", f"prosody: нет полей {missing}")
    whisper = raw.get("whisper")
    if not isinstance(whisper, bool):
        raise WorkerError("prosody_invalid", "prosody.whisper: ожидался boolean")
    raw_sentence_pause = raw.get("sentence_pause_ms")
    if (
        isinstance(raw_sentence_pause, bool)
        or not isinstance(raw_sentence_pause, (int, float))
        or not float(raw_sentence_pause).is_integer()
    ):
        raise WorkerError(
            "prosody_invalid", "prosody.sentence_pause_ms: ожидалось целое 0–500"
        )
    sentence_pause = int(raw_sentence_pause)
    if not 0 <= sentence_pause <= SENTENCE_PAUSE_MAX_MS:
        raise WorkerError(
            "prosody_invalid",
            f"prosody.sentence_pause_ms={sentence_pause}: ожидалось 0–{SENTENCE_PAUSE_MAX_MS}",
        )
    return ProsodyParams(
        style_id=_require_text(raw, "style_id"),
        style_catalog_version=_require_text(raw, "style_catalog_version"),
        rate=_require_number(raw, "rate", RATE_MIN, RATE_MAX),
        gain_db=_require_number(raw, "gain_db", GAIN_MIN_DB, GAIN_MAX_DB),
        pitch_semitones=_require_number(
            raw, "pitch_semitones", PITCH_MIN_SEMITONES, PITCH_MAX_SEMITONES
        ),
        sentence_pause_ms=sentence_pause,
        whisper=whisper,
    )


def format_seconds(value: float) -> str:
    """Канонический вид секунд для ``applied``/метки: ``0.4``, ``1``, ``0.05``."""

    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _validate_pause_seconds(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or not (
        PAUSE_MIN_SECONDS - 1e-9 <= value <= PAUSE_MAX_SECONDS + 1e-9
    ):
        raise WorkerError(
            "pause_invalid",
            f"[[pause:{raw_value}]]: ожидалось {PAUSE_MIN_SECONDS}–{PAUSE_MAX_SECONDS} с",
        )
    steps = value / PAUSE_STEP_SECONDS
    if abs(steps - round(steps)) > 1e-6:
        raise WorkerError(
            "pause_invalid",
            f"[[pause:{raw_value}]]: шаг паузы {PAUSE_STEP_SECONDS} с",
        )
    return round(value, 2)


def split_pause_markup(text: str) -> tuple[list[str], list[float], list[str]]:
    """Разрезать текст по меткам ``[[pause:...]]`` без срезания меток.

    Возвращает ``(fragments, pauses, applied)``: сегменты произносимого текста,
    паузы после каждого сегмента кроме последнего (``len(pauses) ==
    len(fragments) - 1``) и факты ``applied`` вида ``pause:0.4``.

    Ошибки ``pause_invalid``: кривая запись/шаг/диапазон, > 6 меток, сумма
    > 6 с, пустой сегмент (метка в начале/конце или две метки подряд).
    """

    markers = list(_PAUSE_MARKER_RE.finditer(text))
    if len(markers) != text.count("[[pause"):
        raise WorkerError(
            "pause_invalid",
            "в тексте есть незакрытая/битая метка [[pause:...]]; ожидалось "
            "[[pause:<секунды>]]",
        )
    fragments_raw: list[str] = []
    pauses: list[float] = []
    applied: list[str] = []
    cursor = 0
    for marker in markers:
        body = marker.group("body")
        match = _PAUSE_BODY_RE.fullmatch(body)
        if match is None:
            raise WorkerError(
                "pause_invalid",
                f"[[pause{body}]]: ожидалось [[pause:<секунды>]], например [[pause:0.4]]",
            )
        value = _validate_pause_seconds(match.group("value"))
        fragments_raw.append(text[cursor : marker.start()])
        pauses.append(value)
        applied.append(f"pause:{format_seconds(value)}")
        cursor = marker.end()
    fragments_raw.append(text[cursor:])
    if len(pauses) > MAX_PAUSES_PER_PHRASE:
        raise WorkerError(
            "pause_invalid",
            f"{len(pauses)} пауз во фразе: максимум {MAX_PAUSES_PER_PHRASE}",
        )
    if sum(pauses) > MAX_TOTAL_PAUSE_SECONDS + 1e-9:
        raise WorkerError(
            "pause_invalid",
            f"суммарная пауза {sum(pauses):g} с: максимум {MAX_TOTAL_PAUSE_SECONDS} с",
        )
    fragments = [fragment.strip() for fragment in fragments_raw]
    if any(not fragment for fragment in fragments):
        raise WorkerError(
            "pause_invalid",
            "пустой сегмент: метка [[pause]] не может стоять в начале/конце или "
            "две метки подряд",
        )
    return fragments, pauses, applied


def sentence_pieces(fragment: str) -> list[str]:
    """Разбить кусок на предложения по ``.!?…`` (смежная пунктуация — одна).

    Граница — только там, где после пунктуации есть пробел и следующий текст;
    точка без пробела (опечатка/десятичная запись) остаётся внутри сегмента,
    чтобы лишняя тишина не вставлялась. Пунктуация остаётся в предложении.
    """

    pieces: list[str] = []
    start = 0
    index = 0
    length = len(fragment)
    while index < length:
        if fragment[index] not in SENTENCE_END_CHARS:
            index += 1
            continue
        end = index
        while end < length and fragment[end] in SENTENCE_END_CHARS:
            end += 1
        after = end
        while after < length and fragment[after].isspace():
            after += 1
        if after >= length or after == end:
            # Конец фразы или пунктуация без пробела (опечатка/десятичная
            # запись): границы предложения нет, тишину не вставляем.
            break
        piece = fragment[start:end].strip()
        if piece:
            pieces.append(piece)
        start = after
        index = after
    tail = fragment[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [fragment.strip()]


def build_segments(
    text: str, *, sentence_pause_ms: int = 0
) -> tuple[list[SpeechSegment], list[float], list[str]]:
    """Построить план сегментов фразы: паузы меток + паузы конца предложения.

    ``pauses`` в результате — только авторские паузы ``[[pause]]`` (для
    телеметрии); паузы предложений входят в ``SpeechSegment.pause_after_seconds``
    и видны через ``prosody.sentence_pause_ms``. После последнего сегмента
    паузы нет (конец фразы уже даёт тишину окна), поэтому ``sum(pauses)`` и
    суммы сегментов не складываются дважды.
    """

    if not 0 <= int(sentence_pause_ms) <= SENTENCE_PAUSE_MAX_MS:
        raise WorkerError(
            "prosody_invalid",
            f"sentence_pause_ms={sentence_pause_ms}: ожидалось 0–{SENTENCE_PAUSE_MAX_MS}",
        )
    fragments, pauses, applied = split_pause_markup(text)
    segments: list[SpeechSegment] = []
    for fragment_index, fragment in enumerate(fragments):
        pieces = sentence_pieces(fragment) if sentence_pause_ms > 0 else [fragment]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            raise WorkerError(
                "pause_invalid", "пустой сегмент после разбиения по предложениям"
            )
        for piece_index, piece in enumerate(pieces):
            if piece_index < len(pieces) - 1:
                pause_after = int(sentence_pause_ms) / 1000.0
            else:
                pause_after = (
                    pauses[fragment_index] if fragment_index < len(pauses) else 0.0
                )
            segments.append(SpeechSegment(text=piece, pause_after_seconds=pause_after))
    return segments, pauses, applied


__all__ = [
    "MAX_PAUSES_PER_PHRASE",
    "MAX_TOTAL_PAUSE_SECONDS",
    "PAUSE_MAX_SECONDS",
    "PAUSE_MIN_SECONDS",
    "PAUSE_STEP_SECONDS",
    "PROSODY_FIELDS",
    "PROSODY_KEYS",
    "ProsodyParams",
    "SENTENCE_END_CHARS",
    "SpeechSegment",
    "build_segments",
    "format_seconds",
    "parse_prosody",
    "sentence_pieces",
    "split_pause_markup",
]
