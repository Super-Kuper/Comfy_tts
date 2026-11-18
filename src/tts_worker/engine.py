"""Qwen3-TTS callable: граница исполнения движка (мок/инъекция для тестов).

Живой код модели **ленивый**: torch/qwen-tts импортируются только при первом
job'е (``_load``), поэтому воркер и тесты запускаются без GPU и без
установленного torch; отсутствие runtime даёт видимый код
``engine_unavailable``, а не ImportError и не тихий фолбэк.

Модель — pinned **Qwen3-TTS 1.7B Base** (Apache-2.0): она умеет только
клонирование голоса (``generate_voice_clone``), пресетных голосов у Base нет.
Поэтому ``voice.mode="preset"`` на этом образе — видимый
``voice_mode_unsupported``, а не подмена пресетом. Reference PCM приходит от
``tts_worker.reference`` (24 кГц mono): воркер скачал и проверил его сам.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from . import dsp
from .errors import WorkerError
from .prosody import ProsodyParams
from .reference import ReferenceAudio

# Алиасы с прежними именами: старые тесты подменяют ``engine._time_stretch`` и
# т.п., а сами DSP-примитивы (ленивые numpy/librosa/soundfile) живут в dsp.py.
_time_stretch = dsp.time_stretch
_resample = dsp.resample
_to_channels = dsp.to_channels
_encode_wav = dsp.encode_wav
_apply_gain = dsp.apply_gain
_pitch_shift = dsp.pitch_shift
_whisper_filter = dsp.whisper_filter

ENGINE_ID = "qwen3_tts"

#: Контрактный язык (``ru``) → имя языка модели (``russian``).
#: Qwen3-TTS ищет язык в ``talker_config.codec_language_id``; ключи там —
#: полные английские имена (``russian``, ``english``, ...), не ISO-коды.
LANGUAGE_NAMES: dict[str, str] = {
    "ru": "russian",
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
}


def language_for_model(language: str) -> str:
    """Имя языка для Qwen3-TTS: ISO-код из контракта → ключ ``codec_language_id``."""

    key = str(language or "").strip().lower()
    return LANGUAGE_NAMES.get(key, key or "auto")


@dataclass(frozen=True)
class SynthesisTake:
    """Факт одного дубля: байты WAV и фактический seed (None — движок не сообщил)."""

    audio: bytes
    seed: int | None = None


class TtsEngine(Protocol):
    """Контракт движка: фраза + параметры + референс → N дублей WAV."""

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
        reference: ReferenceAudio | None = None,
        prosody: ProsodyParams | None = None,
    ) -> list[SynthesisTake]:
        ...


# --- Граница аудио-постобработки ---------------------------------------------
# Примитивы DSP живут в ``tts_worker.dsp`` (ленивые numpy/librosa/soundfile), а
# здесь — тонкие алиасы: старые тесты подменяют ``engine._time_stretch`` и т.п.,
# поэтому ``_postprocess`` обращается к модульным именам, а не к ``dsp.*``.
# Порядок: whisper → pitch → time_stretch (один) → resample → gain → channels → WAV.


def _postprocess(
    pcm: Any,
    *,
    source_rate: int,
    target_rate: int,
    channels: int,
    speed: float,
    gain_db: float = 0.0,
    pitch_semitones: float = 0.0,
    whisper: bool = False,
) -> bytes:
    """Привести выход модели к контракту: DSP-просодия, частота, каналы, WAV.

    ``speed`` — **effective_rate** (уже ``input.speed × prosody.rate``, проверен
    ``dsp.effective_rate``): один time_stretch, никакого atempo. ``gain_db``
    применяется с пик-лимитером (клиппинга нет); ``whisper`` — экспериментальный
    lowpass+gain; ``pitch_semitones`` — только при ненулевом значении.
    """

    if whisper:
        pcm = _whisper_filter(pcm, sample_rate=int(source_rate))
    if abs(float(pitch_semitones)) > 1e-9:
        pcm = _pitch_shift(
            pcm, sample_rate=int(source_rate), semitones=float(pitch_semitones)
        )
    if abs(float(speed) - 1.0) > 1e-9:
        pcm = _time_stretch(pcm, rate=float(speed))
    if int(source_rate) != int(target_rate):
        pcm = _resample(pcm, orig_sr=int(source_rate), target_sr=int(target_rate))
    if abs(float(gain_db)) > 1e-9:
        pcm = _apply_gain(pcm, gain_db=float(gain_db))
    pcm = _to_channels(pcm, channels=int(channels))
    return _encode_wav(pcm, sample_rate=int(target_rate))


class Qwen3TtsEngine:
    """Pinned Qwen3-TTS 1.7B Base на томе; callable-обвязка без выбора модели.

    Модель — свойство образа/тома, а не job'а: параметра ``model`` нигде нет.
    ``engine_version`` приходит из pinned-манифеста при сборке зависимостей
    (факт загруженного образа). Base поддерживает только клонирование:
    ``generate_voice_clone(text, language, ref_audio=(np, sr), ref_text=...)``.
    """

    engine_id = ENGINE_ID

    def __init__(self, model_dir: Path, engine_version: str, *, device: str = "cuda") -> None:
        self._model_dir = Path(model_dir)
        self.engine_version = engine_version
        self._device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        """Ленивая загрузка pinned Qwen3-TTS Base с тома (один раз на процесс)."""

        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise WorkerError(
                "engine_unavailable",
                "Qwen3-TTS runtime (torch/qwen-tts) не установлен в образе: "
                f"{exc}; тесты/стенд — fake/injection",
            ) from exc
        device = str(self._device or "cuda").strip()
        kwargs: dict[str, Any] = {"device_map": "cuda:0" if device in {"cuda", "cuda:0"} else device}
        if device.startswith("cuda") or device == "cuda":
            kwargs["dtype"] = torch.bfloat16
        try:
            model = Qwen3TTSModel.from_pretrained(str(self._model_dir), **kwargs)
        except Exception as exc:  # noqa: BLE001 - видимая ошибка загрузки модели
            raise WorkerError(
                "engine_unavailable",
                f"Qwen3-TTS Base не загрузился из {self._model_dir}: {exc}",
            ) from exc
        self._model = model
        return model

    def _generate_clone(
        self,
        model: Any,
        *,
        text: str,
        language: str,
        reference: ReferenceAudio,
        seed: int | None,
    ) -> tuple[Any, int]:
        """Один вызов ``generate_voice_clone``: (numpy PCM mono, частота)."""

        if seed is not None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - зависимость образа
                raise WorkerError(
                    "engine_unavailable",
                    f"torch не установлен в образе воркера: {exc}",
                ) from exc
            torch.manual_seed(int(seed))
        try:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language_for_model(language),
                ref_audio=(reference.pcm, reference.sample_rate),
                ref_text=reference.text,
            )
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - сбой генерации — видимый engine_failed
            raise WorkerError(
                "engine_failed",
                f"Qwen3-TTS generate_voice_clone: {type(exc).__name__}: {exc}",
            ) from exc
        if isinstance(wavs, (list, tuple)):
            if not wavs:
                raise WorkerError("audio_empty_result", "Qwen3-TTS вернул 0 дублей")
            pcm = wavs[0]
        else:
            pcm = wavs
        if pcm is None:
            raise WorkerError("audio_empty_result", "Qwen3-TTS вернул пустой PCM")
        return pcm, int(sample_rate)

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
        reference: ReferenceAudio | None = None,
        prosody: ProsodyParams | None = None,
    ) -> list[SynthesisTake]:
        """Синтезировать 1–2 дубля клонированным голосом; preset на Base — ошибка.

        ``prosody`` (v1.2) — явные DSP-параметры уже разрешённого стиля. Effective
        rate считается здесь (``speed × prosody.rate``, 0.5–2.0, иначе видимый
        ``prosody_rate``) и применяется одним time_stretch; gain/whisper/pitch
        уходят в ``_postprocess``. Без ``prosody`` поведение v1 не меняется.
        """

        mode = str(voice.get("mode") or "") if isinstance(voice, Mapping) else ""
        if mode == "preset":
            raise WorkerError(
                "voice_mode_unsupported",
                "Qwen3-TTS 1.7B Base не имеет пресетных голосов: нужен "
                "voice.mode='clone' + reference; тихого перехода на пресет нет",
            )
        if mode != "clone":
            raise WorkerError(
                "voice_mode_unsupported",
                f"voice.mode={mode!r}: Qwen3-TTS Base поддерживает только clone",
            )

        # Effective rate проверяется до референса/загрузки модели: ошибка
        # просодии не должна стоить ни сети, ни GPU (видимый prosody_rate).
        if prosody is None:
            effective_rate = float(speed)
            gain_db = 0.0
            pitch_semitones = 0.0
            whisper = False
        else:
            effective_rate = dsp.effective_rate(speed, prosody.rate)
            gain_db = float(prosody.gain_db)
            pitch_semitones = float(prosody.pitch_semitones)
            whisper = bool(prosody.whisper)

        if reference is None:
            raise WorkerError(
                "reference_missing",
                "voice.mode='clone' без подготовленного референса",
            )

        model = self._load()
        count = max(1, int(takes))
        result: list[SynthesisTake] = []
        for _index in range(count):
            pcm, source_rate = self._generate_clone(
                model,
                text=text,
                language=language,
                reference=reference,
                seed=seed,
            )
            audio = _postprocess(
                pcm,
                source_rate=source_rate,
                target_rate=int(sample_rate),
                channels=int(channels),
                speed=effective_rate,
                gain_db=gain_db,
                pitch_semitones=pitch_semitones,
                whisper=whisper,
            )
            result.append(SynthesisTake(audio=audio, seed=seed))
        return result


__all__ = [
    "ENGINE_ID",
    "LANGUAGE_NAMES",
    "Qwen3TtsEngine",
    "SynthesisTake",
    "TtsEngine",
    "language_for_model",
]
