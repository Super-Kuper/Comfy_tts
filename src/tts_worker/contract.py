"""Валидация job по JSON Schemas контракта (input/output)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import WorkerError

_CACHE: dict[Path, Any] = {}


def _validator(path: Path) -> Any:
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - зависимость образа
        raise WorkerError("config_invalid", "jsonschema не установлен в образе воркера") from exc
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError("config_invalid", f"схема не читается: {path}: {exc}") from exc
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise WorkerError("config_invalid", f"невалидная JSON Schema {path.name}: {exc.message}") from exc
    validator = jsonschema.Draft202012Validator(schema)
    _CACHE[path] = validator
    return validator


def _format_error(error: Any) -> str:
    location = "/".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{location}: {error.message}"


def validate_input(payload: Mapping[str, Any], contract_dir: Path) -> None:
    """Проверить job payload по input.schema.json; нарушение — ``input_invalid``."""

    validator = _validator(Path(contract_dir) / "input.schema.json")
    error = next(validator.iter_errors(payload), None)
    if error is not None:
        raise WorkerError("input_invalid", _format_error(error))


def validate_output(payload: Mapping[str, Any], contract_dir: Path) -> None:
    """Финальный гейт: handler возвращает ровно контракт, иначе ``output_invalid``."""

    validator = _validator(Path(contract_dir) / "output.schema.json")
    error = next(validator.iter_errors(payload), None)
    if error is not None:
        raise WorkerError("output_invalid", _format_error(error))


__all__ = ["validate_input", "validate_output"]
