"""Health: ok/starting/degraded по тому и pinned-файлам, без синтеза/выравнивания."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from tts_worker.handler import Dependencies, execute
from tts_worker.health import health_payload


def _volume(tmp_path: Path) -> Path:
    volume = tmp_path / "volume"
    volume.mkdir(parents=True, exist_ok=True)
    return volume


def test_ready_when_volume_and_models_are_present(tmp_path: Path, setup) -> None:
    config, manifest, _ = setup
    result = health_payload(replace(config, volume_root=_volume(tmp_path)), manifest)
    assert result == {
        "status": "ok",
        "volume_mounted": True,
        "engine": "ready",
        "aligner": "ready",
    }


def test_missing_aligner_file_is_starting(tmp_path: Path, setup) -> None:
    config, manifest, models_root = setup
    (models_root / manifest.entry("aligner").relative_path).unlink()
    result = health_payload(replace(config, volume_root=_volume(tmp_path)), manifest)
    assert result["status"] == "starting"
    assert result["engine"] == "ready"
    assert result["aligner"] == "missing"


def test_missing_volume_is_degraded(tmp_path: Path, setup) -> None:
    config, manifest, _ = setup
    result = health_payload(replace(config, volume_root=tmp_path / "absent"), manifest)
    assert result["status"] == "degraded"
    assert result["volume_mounted"] is False
    assert result["engine"] == "ready"
    assert result["aligner"] == "ready"


def test_health_action_does_not_synthesize(tmp_path: Path, setup) -> None:
    config, manifest, _ = setup
    volume = _volume(tmp_path)

    class Boom:
        def synthesize(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("health не должен синтезировать")

        def word_timestamps(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("health не должен выравнивать")

    deps = Dependencies(
        config=replace(config, volume_root=volume),
        manifest=manifest,
        engine=Boom(),  # type: ignore[arg-type]
        aligner=Boom(),  # type: ignore[arg-type]
        normalizer=Boom(),  # type: ignore[arg-type]
    )
    result = asyncio.run(execute({"id": "h1", "input": {"action": "health"}}, deps))
    assert result == {
        "status": "ok",
        "volume_mounted": True,
        "engine": "ready",
        "aligner": "ready",
    }
