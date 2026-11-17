"""Test helpers: WAV bytes, pinned manifest, fake engine/aligner/normalizer.

Tests need no GPU, no network, no torch and no runpod SDK: the engine, aligner,
normalizer and storage are injected fakes, exactly like the live worker injects
the pinned model callables.
"""

from __future__ import annotations

import hashlib
import io
import json
import wave
from pathlib import Path
from typing import Any, Callable, Sequence

from tts_worker.aligner import Alignment, WordTiming
from tts_worker.engine import SynthesisTake
from tts_worker.normalization import NormalizationResult

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contract"
DATA = ROOT / "data"
PIN = "a" * 40

# path under MODELS_ROOT -> (version, license, role)
MODELS: dict[str, tuple[str, str, str]] = {
    "qwen3-tts-1.7b/model.safetensors": ("qwen3-tts-12hz-1.7b@2026-01", "Apache-2.0", "engine"),
    "qwen3-tts-1.7b/config.json": ("qwen3-tts-12hz-1.7b@2026-01", "Apache-2.0", "support"),
    "gigaam-v3/pytorch_model.bin": ("gigaam-v3@2026-04", "MIT", "aligner"),
    "gigaam-v3/config.json": ("gigaam-v3@2026-04", "MIT", "support"),
}


def contract_dir() -> Path:
    return CONTRACT


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wav_bytes(
    duration_seconds: float = 2.0,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
) -> bytes:
    """Valid silent PCM s16le WAV with the requested facts."""

    frames = max(0, int(round(float(duration_seconds) * sample_rate)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * channels * frames)
    return buffer.getvalue()


def flac_bytes(
    duration_seconds: float = 2.0,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
) -> bytes:
    """Minimal FLAC container with a STREAMINFO block (header facts only)."""

    total_samples = max(1, int(round(float(duration_seconds) * sample_rate)))
    packed = (sample_rate << 44) | ((channels - 1) << 41) | ((16 - 1) << 36) | total_samples
    streaminfo = (
        (4096).to_bytes(2, "big")
        + (4096).to_bytes(2, "big")
        + (0).to_bytes(3, "big")
        + (0).to_bytes(3, "big")
        + packed.to_bytes(8, "big")
        + b"\x00" * 16
    )
    header = bytes([0x80 | 0x00]) + len(streaminfo).to_bytes(3, "big")
    return b"fLaC" + header + streaminfo


def model_content(path: str) -> bytes:
    return f"weights:{path}".encode()


def model_sha(path: str) -> str:
    return hashlib.sha256(model_content(path)).hexdigest()


def manifest_line(path: str) -> str:
    version, license_id, role = MODELS[path]
    url = f"https://huggingface.co/example/repo/resolve/{PIN}/{path}"
    return "\t".join([path, url, model_sha(path), version, license_id, role])


def write_models(models_root: Path) -> Path:
    """Write fake model files and a matching pinned manifest; return manifest path."""

    lines = ["# test manifest", *[manifest_line(path) for path in MODELS]]
    for path in MODELS:
        target = models_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(model_content(path))
    manifest_path = models_root.parent / "models-manifest.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


class FakeEngine:
    """Подменяет Qwen3-TTS: без GPU, записывает аргументы вызова."""

    engine_id = "qwen3_tts"

    def __init__(
        self,
        *,
        engine_version: str = "qwen3-tts-12hz-1.7b@fake",
        takes: Sequence[SynthesisTake] | Callable[..., Sequence[SynthesisTake]] | None = None,
    ) -> None:
        self.engine_version = engine_version
        self._takes = takes
        self.calls: list[dict[str, Any]] = []

    def synthesize(self, text: str, **kwargs: Any) -> list[SynthesisTake]:
        self.calls.append({"text": text, **kwargs})
        if callable(self._takes):
            return list(self._takes(text, **kwargs))
        if self._takes is not None:
            return list(self._takes)
        seed = kwargs.get("seed")
        return [
            SynthesisTake(audio=wav_bytes(2.0), seed=seed)
            for _ in range(int(kwargs.get("takes", 1)))
        ]


class FakeAligner:
    """Подменяет GigaAM-v3: слова из фразы, заданный WER, запись вызовов."""

    def __init__(
        self,
        *,
        aligner_version: str = "gigaam-v3@fake",
        words: Sequence[WordTiming] | None = None,
        wer: float | None = 0.0,
    ) -> None:
        self.aligner_version = aligner_version
        self._words = words
        self.wer = wer
        self.calls: list[dict[str, Any]] = []

    def word_timestamps(self, audio: bytes, text: str, *, language: str) -> Alignment:
        self.calls.append({"audio": audio, "text": text, "language": language})
        if self._words is not None:
            words = tuple(self._words)
        else:
            words = tuple(
                WordTiming(text=word, start_ms=40 + 160 * index, end_ms=200 + 160 * index)
                for index, word in enumerate(text.split()[:4])
            )
        return Alignment(words=words, wer=self.wer)


class FakeNormalizer:
    """Подменяет словарную нормализацию: фиксированный текст, запись вызовов."""

    def __init__(
        self,
        output: str | Callable[[str], str] | None = None,
    ) -> None:
        self._output = output
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, language: str) -> NormalizationResult:
        self.calls.append((text, language))
        if callable(self._output):
            result = self._output(text)
        elif self._output is not None:
            result = self._output
        else:
            result = text
        return NormalizationResult(text=result, version="normalizer@fake", applied=("fake",))


class FakeStorage:
    """Подменяет S3: запоминает загруженные байты и отдаёт presigned URL."""

    def __init__(self) -> None:
        self.uploads: list[tuple[bytes, str]] = []

    async def upload(self, data: bytes, content_type: str) -> str:
        self.uploads.append((data, content_type))
        return f"https://storage.example/voiceover/take-{len(self.uploads)}.wav?signature=test"


class FakeClock:
    """Детерминированные секунды вместо time.monotonic."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return value
