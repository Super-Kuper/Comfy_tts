"""DSP просодии: детерминированные эффекты над PCM, без GPU и без model-инструкций.

Модуль границы «пост-обработка» контракта ``tts-serverless/v1.2`` (worker
0.4.0). Здесь только измеримые эффекты варианта A (DSP-only):

* ``effective_rate`` — единая проверка ``speed × prosody.rate`` в 0.5–2.0;
* ``time_stretch``/``pitch_shift``/``resample`` — librosa, ленивый импорт;
* ``apply_gain`` — усиление с пик-лимитером (клиппинга нет: пик ≤ 0.999);
* ``whisper_filter`` — экспериментальный lowpass (оконный sinc) + gain;
* ``combine_wav_segments`` — точная склейка сегментов с тишиной (stdlib
  ``wave``, без numpy): длительность паузы считается в кадрах запрошенной
  частоты, поэтому silence не «плывёт».

Тяжёлые зависимости (numpy/librosa/soundfile) импортируются **внутри**
функций: GPU-free тесты подменяют эти функции и не требуют их установки.
Чистые решения (коэффициент усиления, ядро lowpass, формат WAV) вынесены в
функции без зависимостей — их можно проверять математикой в обычном pytest.
"""

from __future__ import annotations

import io
import math
import wave
from collections.abc import Sequence
from typing import Any

from .errors import WorkerError

#: Продуктовый диапазон скорости (contract §1: speed 0.5–2.0). Effective rate —
#: ``speed × prosody.rate`` и тоже обязан остаться в этом диапазоне.
RATE_MIN = 0.5
RATE_MAX = 2.0

#: Параметры каталога DSP-стилей (plan §4); схема входа дублирует их для
#: строгого отказа до GPU, здесь — defensive-проверка перед DSP.
GAIN_MIN_DB = -6.0
GAIN_MAX_DB = 6.0
PITCH_MIN_SEMITONES = -2.0
PITCH_MAX_SEMITONES = 2.0
SENTENCE_PAUSE_MAX_MS = 500

#: Пик-лимитер: gain не загоняет сигнал в клиппинг, а упирается в 0.999 FS.
#: Фактическое усиление может быть меньше запрошенного — это видимое поведение
#: limiter'а, а не тихая подмена стиля.
PEAK_CEILING = 0.999

#: Экспериментальный шёпот: lowpass + собственный gain (плюс gain стиля).
WHISPER_LOWPASS_HZ = 3500.0
WHISPER_GAIN_DB = 1.5


def effective_rate(speed: float, prosody_rate: float) -> float:
    """``speed × prosody.rate`` в 0.5–2.0; иначе видимый ``prosody_rate``.

    Один множитель на всю фразу (никакого авто-atempo и подгонки под окно):
    time_stretch применяется ровно один раз.
    """

    try:
        value = float(speed) * float(prosody_rate)
    except (TypeError, ValueError) as exc:  # defensive: схема уже требует числа
        raise WorkerError(
            "prosody_rate",
            f"speed={speed!r} × prosody.rate={prosody_rate!r}: ожидались числа",
        ) from exc
    if not math.isfinite(value) or not (
        RATE_MIN - 1e-9 <= value <= RATE_MAX + 1e-9
    ):
        raise WorkerError(
            "prosody_rate",
            f"effective_rate = speed {float(speed):g} × prosody.rate "
            f"{float(prosody_rate):g} = {value:g}: ожидалось {RATE_MIN}–{RATE_MAX}; "
            "авто-подгонки нет",
        )
    return round(value, 6)


def gain_scale(peak: float, gain_db: float) -> float:
    """Линейный множитель gain с пик-лимитером; клиппинга не допускает.

    ``peak`` — максимальный модуль сэмпла на входе (0..1). Если после усиления
    пик превысил ``PEAK_CEILING``, множитель уменьшается ровно до потолка —
    форма сигнала сохраняется (без hard-clip), слышимое усиление меньше
    запрошенного. Чистая функция: тестируется без numpy.
    """

    factor = 10.0 ** (float(gain_db) / 20.0)
    if peak <= 0.0 or not math.isfinite(peak):
        return factor
    if peak * factor > PEAK_CEILING:
        return PEAK_CEILING / peak
    return factor


def whisper_kernel(
    sample_rate: int, *, cutoff_hz: float = WHISPER_LOWPASS_HZ
) -> tuple[float, ...]:
    """Оконное sinc-ядро lowpass (Hann), нормированное на единичный DC.

    Нормировка ``sum(kernel) == 1`` сохраняет уровень; длина нечётная
    (линейная фаза), ограничена сверху, чтобы свёртка была дешёвой. Чистая
    функция: тестируется без numpy.
    """

    rate = int(sample_rate)
    if rate <= 0:
        raise WorkerError("output_invalid", f"whisper: sample_rate={sample_rate}")
    cutoff = float(cutoff_hz)
    if not 0.0 < cutoff < rate / 2.0:
        raise WorkerError(
            "output_invalid",
            f"whisper: cutoff={cutoff_hz} Гц вне (0, {rate / 2:g}) при {rate} Гц",
        )
    per_hz = 2 * cutoff / rate
    taps = int(round(2.0 / per_hz)) | 1
    taps = max(3, min(taps, 2049))
    middle = (taps - 1) / 2.0
    values: list[float] = []
    for index in range(taps):
        offset = index - middle
        if offset == 0:
            sinc = 2.0 * per_hz
        else:
            x = math.pi * per_hz * offset
            sinc = math.sin(x) / (math.pi * offset)
        window = 0.5 - 0.5 * math.cos(2.0 * math.pi * index / (taps - 1))
        values.append(sinc * window)
    total = sum(values)
    if total <= 0.0:  # pragma: no cover - защита от экзотических частот
        raise WorkerError("output_invalid", f"whisper: пустое ядро при {rate} Гц")
    return tuple(value / total for value in values)


# --- Ленивые DSP-примитивы (инъекция/подмена в тестах) -----------------------


def time_stretch(pcm: Any, *, rate: float) -> Any:
    """Явная скорость: один ``librosa.effects.time_stretch`` (без atempo)."""

    import librosa

    return librosa.effects.time_stretch(pcm, rate=float(rate))


def pitch_shift(pcm: Any, *, sample_rate: int, semitones: float) -> Any:
    """Сдвиг высоты голоса (``librosa.effects.pitch_shift``), только если ≠ 0."""

    import librosa

    return librosa.effects.pitch_shift(y=pcm, sr=int(sample_rate), n_steps=float(semitones))


def resample(pcm: Any, *, orig_sr: int, target_sr: int) -> Any:
    """Смена частоты дискретизации (``librosa.resample``)."""

    import librosa

    return librosa.resample(pcm, orig_sr=int(orig_sr), target_sr=int(target_sr))


def to_channels(pcm: Any, *, channels: int) -> Any:
    """Mono PCM → запрошенное число каналов (1/2 по контракту)."""

    import numpy as np

    data = np.asarray(pcm, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    raise WorkerError(
        "output_invalid",
        f"каналы движка {data.shape[1]} не сводятся к запрошенным {channels}",
    )


def apply_gain(pcm: Any, *, gain_db: float) -> Any:
    """Усиление на ``gain_db`` с пик-лимитером (``gain_scale``)."""

    import numpy as np

    data = np.asarray(pcm, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return data * np.float32(gain_scale(peak, float(gain_db)))


def lowpass(pcm: Any, *, sample_rate: int, cutoff_hz: float = WHISPER_LOWPASS_HZ) -> Any:
    """FIR lowpass по каждому каналу (ядро ``whisper_kernel``)."""

    import numpy as np

    kernel = np.asarray(whisper_kernel(sample_rate, cutoff_hz=cutoff_hz), dtype=np.float32)
    data = np.asarray(pcm, dtype=np.float32)
    if data.ndim == 1:
        return np.convolve(data, kernel, mode="same").astype(np.float32)
    filtered = [
        np.convolve(data[:, channel], kernel, mode="same")
        for channel in range(data.shape[1])
    ]
    return np.stack(filtered, axis=1).astype(np.float32)


def whisper_filter(pcm: Any, *, sample_rate: int) -> Any:
    """Экспериментальный «шёпот»: lowpass + gain (с пик-лимитером)."""

    filtered = lowpass(pcm, sample_rate=int(sample_rate))
    return apply_gain(filtered, gain_db=WHISPER_GAIN_DB)


def encode_wav(pcm: Any, *, sample_rate: int) -> bytes:
    """WAV PCM s16le (контракт §2: отдаётся только WAV)."""

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, pcm, int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# --- Склейка сегментов с точной тишиной (stdlib, без numpy) ------------------


def wav_params(data: bytes) -> tuple[int, int, int]:
    """``(sample_rate, channels, sampwidth)`` из заголовка WAV-сегмента."""

    if not data:
        raise WorkerError("output_empty", "сегмент синтеза пуст (0 байт)")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WorkerError("output_invalid", "склейка ожидает WAV-сегменты движка")
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return (
                int(handle.getframerate()),
                int(handle.getnchannels()),
                int(handle.getsampwidth()),
            )
    except Exception as exc:  # noqa: BLE001 - любой сбой разбора = output_invalid
        raise WorkerError("output_invalid", f"WAV-сегмент не читается: {exc}") from exc


def _wav_frames(data: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return handle.readframes(handle.getnframes())
    except Exception as exc:  # noqa: BLE001 - любой сбой разбора = output_invalid
        raise WorkerError("output_invalid", f"WAV-сегмент не читается: {exc}") from exc


def combine_wav_segments(segments: Sequence[bytes], pauses_seconds: Sequence[float]) -> bytes:
    """Склеить дубль #i из сегментов, вставив точную тишину между ними.

    ``pauses_seconds[k]`` — тишина **после** ``segments[k]``; длина — на единицу
    меньше числа сегментов. Формат всех сегментов обязан совпадать (движок
    пишет WAV по ``render``): расхождение — видимый ``output_invalid``, а не
    тихое ресемплирование. Одиночный сегмент без пауз возвращается байт-в-байт
    (старый путь не меняется). Тишина = ровно ``round(pause × sample_rate)``
    кадров нулей; длительность в телеметрии/словах смещается на фактические
    кадры, а не на «запрошенные» секунды.
    """

    if not segments:
        raise WorkerError("audio_empty_result", "нет сегментов для склейки")
    if len(pauses_seconds) != len(segments) - 1:
        raise WorkerError(
            "output_invalid",
            f"склейка: {len(segments)} сегментов и {len(pauses_seconds)} пауз",
        )
    if len(segments) == 1 and not pauses_seconds:
        return segments[0]
    rate, channels, width = wav_params(segments[0])
    frame_width = channels * width
    output = bytearray()
    for index, segment in enumerate(segments):
        segment_rate, segment_channels, segment_width = wav_params(segment)
        if (segment_rate, segment_channels, segment_width) != (rate, channels, width):
            raise WorkerError(
                "output_invalid",
                f"сегмент {index + 1}: формат "
                f"{segment_rate}/{segment_channels}/{segment_width} != "
                f"{rate}/{channels}/{width}",
            )
        frames = _wav_frames(segment)
        if not frames:
            raise WorkerError("output_empty", f"сегмент {index + 1}: 0 кадров")
        output += frames
        if index < len(pauses_seconds):
            pause = float(pauses_seconds[index])
            if not math.isfinite(pause) or pause < 0.0:
                raise WorkerError(
                    "output_invalid", f"пауза {index + 1}: {pause} с не может быть вставлена"
                )
            silence_frames = int(round(pause * rate))
            output += b"\x00" * (silence_frames * frame_width)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(bytes(output))
    return buffer.getvalue()


__all__ = [
    "GAIN_MAX_DB",
    "GAIN_MIN_DB",
    "PEAK_CEILING",
    "PITCH_MAX_SEMITONES",
    "PITCH_MIN_SEMITONES",
    "RATE_MAX",
    "RATE_MIN",
    "SENTENCE_PAUSE_MAX_MS",
    "WHISPER_GAIN_DB",
    "WHISPER_LOWPASS_HZ",
    "apply_gain",
    "combine_wav_segments",
    "effective_rate",
    "encode_wav",
    "gain_scale",
    "lowpass",
    "pitch_shift",
    "resample",
    "time_stretch",
    "to_channels",
    "wav_params",
    "whisper_filter",
    "whisper_kernel",
]
