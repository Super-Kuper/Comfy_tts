"""Общие фикстуры: конфиг без GPU/сети, том с pinned-моделями и манифест."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helpers import CONTRACT, DATA, write_models  # noqa: E402

from tts_worker.config import Config  # noqa: E402
from tts_worker.manifest import ModelManifest  # noqa: E402


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Config, ModelManifest, Path]:
    """Том с моделями, pinned-манифест и конфиг с быстрыми таймаутами."""

    models_root = tmp_path / "models"
    manifest_path = write_models(models_root)
    config = Config.from_env(
        {
            "MODELS_ROOT": str(models_root),
            "MODELS_MANIFEST": str(manifest_path),
            "CONTRACT_DIR": str(CONTRACT),
            "TTS_DICTIONARIES": str(DATA / "normalization.ru.json"),
            "JOB_TIMEOUT_SECONDS": "2",
            "OUTPUT_INLINE_MAX_MB": "1",
            "OUTPUT_INLINE_BUDGET_MB": "2",
        }
    )
    return config, ModelManifest.load(manifest_path), models_root
