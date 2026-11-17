"""Pinned-манифест весов TTS-стенда (`models-manifest.txt`).

Формат строки (tab-separated):

    <path under MODELS_ROOT>\t<URL>\t<sha256>\t<version>\t<license>\t<role>

role:
* ``engine``  — веса TTS-движка: их sha256 становятся telemetry.model_hash,
  license — telemetry.license; ровно одна строка;
* ``aligner`` — веса GigaAM-v3 (words/WER); ровно одна строка;
* ``support`` — config/tokenizer-файлы обеих моделей (0..n строк).

URL обязан быть pinned на immutable commit Hugging Face
(`/resolve/<40-hex-commit>/`); `main`/`master` запрещены. Веса в git не
хранятся; скачивание/проверка — ``populate-volume.sh``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import WorkerError

PIN_RE = re.compile(r"/resolve/([0-9a-f]{40})/")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"^https://")
ROLES = frozenset({"engine", "aligner", "support"})

_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class ModelFile:
    """Одна pinned-запись манифеста."""

    path: str
    url: str
    sha256: str
    version: str
    license: str
    role: str

    @property
    def relative_path(self) -> Path:
        return Path(self.path)


def sha256_file(path: Path) -> str:
    """sha256 файла; результат кешируется по (путь, mtime, размер) в процессе."""

    stat = path.stat()
    key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _CACHE[key] = value
    return value


def _validate_path(raw: str, number: int) -> str:
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != raw.replace("\\", "/")
        or raw.endswith("/")
    ):
        raise WorkerError(
            "config_invalid",
            f"models-manifest.txt:{number}: путь относительный без '..', получено {raw!r}",
        )
    return raw


def parse_manifest(text: str) -> tuple[ModelFile, ...]:
    """Разобрать и провалидировать манифест; ошибка — ``config_invalid``."""

    entries: list[ModelFile] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 6:
            raise WorkerError(
                "config_invalid",
                f"models-manifest.txt:{number}: ожидалось 6 колонок "
                "<path>\\t<URL>\\t<sha256>\\t<version>\\t<license>\\t<role>",
            )
        raw_path, url, sha256, version, license_id, role = (
            column.strip() for column in columns
        )
        path = _validate_path(raw_path, number)
        if not URL_RE.match(url) or PIN_RE.search(url) is None or "/resolve/main/" in url:
            raise WorkerError(
                "config_invalid",
                f"models-manifest.txt:{number}: URL не запинен на commit: {url}",
            )
        if SHA_RE.fullmatch(sha256) is None:
            raise WorkerError("config_invalid", f"models-manifest.txt:{number}: sha256 не 64 hex")
        if not version or "@" not in version:
            raise WorkerError(
                "config_invalid", f"models-manifest.txt:{number}: version без @: {version!r}"
            )
        if not license_id:
            raise WorkerError("config_invalid", f"models-manifest.txt:{number}: пустая license")
        if role not in ROLES:
            raise WorkerError(
                "config_invalid",
                f"models-manifest.txt:{number}: role={role!r}, ожидалось одно из {sorted(ROLES)}",
            )
        entries.append(ModelFile(path, url, sha256, version, license_id, role))
    if not entries:
        raise WorkerError("config_invalid", "models-manifest.txt: нет ни одной записи")
    paths = [entry.path for entry in entries]
    if len(paths) != len(set(paths)):
        raise WorkerError("config_invalid", "models-manifest.txt: дубликаты путей")
    for role in ("engine", "aligner"):
        count = sum(1 for entry in entries if entry.role == role)
        if count != 1:
            raise WorkerError(
                "config_invalid",
                f"models-manifest.txt: ровно одна строка role={role}, найдено {count}",
            )
    return tuple(entries)


class ModelManifest:
    """Упорядоченный pinned-манифест с ролями engine/aligner/support."""

    def __init__(self, entries: tuple[ModelFile, ...]) -> None:
        self._entries = entries
        self._by_role: dict[str, list[ModelFile]] = {}
        for entry in entries:
            self._by_role.setdefault(entry.role, []).append(entry)

    @classmethod
    def parse(cls, text: str) -> "ModelManifest":
        return cls(parse_manifest(text))

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkerError(
                "config_invalid", f"models-manifest.txt не читается: {path}: {exc}"
            ) from exc
        return cls.parse(text)

    @property
    def entries(self) -> tuple[ModelFile, ...]:
        return self._entries

    def entry(self, role: str) -> ModelFile:
        """Единственная запись роли (engine/aligner); иначе ``config_invalid``."""

        entries = self._by_role.get(role, [])
        if len(entries) != 1:
            raise WorkerError(
                "config_invalid",
                f"models-manifest.txt: role={role}: ожидалась одна строка, найдено {len(entries)}",
            )
        return entries[0]

    def files_for(self, role: str) -> tuple[ModelFile, ...]:
        return tuple(self._by_role.get(role, []))


__all__ = ["ModelFile", "ModelManifest", "parse_manifest", "sha256_file"]
