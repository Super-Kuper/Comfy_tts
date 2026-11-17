"""models-manifest.txt: commit-pinned URL, sha256, роли engine/aligner, layout."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tts_worker.errors import WorkerError
from tts_worker.manifest import ModelManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models-manifest.txt"

PIN_RE = re.compile(r"/resolve/([0-9a-f]{40})/")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _rows() -> list[list[str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(line.split("\t"))
    return rows


def test_manifest_parses_with_roles() -> None:
    manifest = ModelManifest.load(MANIFEST)
    engine = manifest.entry("engine")
    aligner = manifest.entry("aligner")
    assert engine.path == "qwen3-tts-1.7b/model.safetensors"
    assert engine.version == "qwen3-tts-12hz-1.7b@2026-01"
    assert engine.license == "Apache-2.0"
    assert aligner.path == "gigaam-v3/pytorch_model.bin"
    assert aligner.version == "gigaam-v3@2026-04"
    assert aligner.license == "MIT"
    assert len(manifest.files_for("support")) >= 2


def test_all_rows_have_six_columns() -> None:
    rows = _rows()
    assert rows, "манифест пуст"
    for row in rows:
        assert len(row) == 6, (
            "ожидалось <path>\\t<URL>\\t<sha256>\\t<version>\\t<license>\\t<role>: "
            f"{row!r}"
        )


def test_urls_are_commit_pinned_and_named() -> None:
    for path, url, sha, version, license_id, role in _rows():
        assert "/resolve/main/" not in url and "/resolve/master/" not in url, (path, url)
        assert PIN_RE.search(url) is not None, f"{path}: URL не запинен на commit: {url}"
        assert Path(path).name in url, f"{path}: имя файла не совпадает с URL {url}"
        assert SHA_RE.fullmatch(sha), f"{path}: sha256 не 64 hex"
        assert "@" in version, f"{path}: version без @: {version!r}"
        assert license_id, f"{path}: пустая license"
        assert role in {"engine", "aligner", "support"}, f"{path}: role={role!r}"


def test_exactly_one_engine_and_aligner() -> None:
    roles = [row[5] for row in _rows()]
    assert roles.count("engine") == 1
    assert roles.count("aligner") == 1


def test_expected_model_files() -> None:
    paths = {row[0] for row in _rows()}
    assert "qwen3-tts-1.7b/model.safetensors" in paths
    assert "qwen3-tts-1.7b/speech_tokenizer/model.safetensors" in paths
    assert "gigaam-v3/pytorch_model.bin" in paths
    assert "gigaam-v3/tokenizer.model" in paths


def test_weights_are_not_in_git() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "models/" in gitignore
    weights = list(ROOT.glob("**/*.safetensors")) + list(ROOT.glob("**/*.bin"))
    assert not weights, "веса не должны лежать в репозитории"


def test_populate_volume_script_exists() -> None:
    script = ROOT / "populate-volume.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "MODELS_MANIFEST" in text and "provision_hash_mismatch" in text


# --- Невалидные манифесты ----------------------------------------------------


def _line(path: str = "m/model.safetensors", **overrides: str) -> str:
    values = {
        "path": path,
        "url": f"https://huggingface.co/example/repo/resolve/{'a' * 40}/{path}",
        "sha256": "b" * 64,
        "version": "engine@2026-01",
        "license": "MIT",
        "role": "engine",
    }
    values.update(overrides)
    return "\t".join(values.values())


def _text(*roles: str) -> str:
    lines = []
    for index, role in enumerate(roles or ("engine", "aligner")):
        path = f"model-{index}/file-{index}.bin"
        lines.append(
            _line(
                path=path,
                url=f"https://huggingface.co/example/repo/resolve/{'a' * 40}/{path}",
                role=role,
            )
        )
    return "\n".join(lines)


@pytest.mark.parametrize(
    "text",
    [
        # URL не запинен на commit.
        _line(url="https://huggingface.co/example/repo/resolve/main/m/model.safetensors")
        + "\n"
        + _line(path="a/aligner.bin", role="aligner"),
        # sha256 не 64 hex.
        _line(sha256="ABC") + "\n" + _line(path="a/aligner.bin", role="aligner"),
        # Неизвестная роль.
        _line(role="other") + "\n" + _line(path="a/aligner.bin", role="aligner"),
        # Выход за MODELS_ROOT.
        _line(path="../escape.bin") + "\n" + _line(path="a/aligner.bin", role="aligner"),
        # Не шесть колонок.
        "only\ttwo\tcolumns",
        # Две строки engine.
        _text("engine", "engine"),
    ],
)
def test_invalid_manifest_is_config_invalid(text: str) -> None:
    with pytest.raises(WorkerError) as error:
        ModelManifest.parse(text)
    assert error.value.code == "config_invalid"


def test_manifest_without_aligner_is_config_invalid() -> None:
    with pytest.raises(WorkerError) as error:
        ModelManifest.parse(_text("engine", "support"))
    assert error.value.code == "config_invalid"
