"""Нормализация русского текста: правила — данные, код только применяет.

Словари лежат в ``data/normalization.ru.json`` (версия в файле, правится новой
версией файла, не правкой кода). Порядок: сокращения → бренды → числа
(0–999 по таблицам) → ударения. Непокрытые формы остаются как написаны — это
видно в QC, а не «тихо исправляется» (contract §1.2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .errors import WorkerError

_NUMBER_RE = re.compile(r"\b\d{1,3}\b")


@dataclass(frozen=True)
class NormalizationResult:
    """Результат нормализации: текст, версия словаря и применённые правила."""

    text: str
    version: str
    applied: tuple[str, ...] = ()


class Normalizer(Protocol):
    """Инъекция для тестов: текст + язык → нормализованный результат."""

    def __call__(self, text: str, language: str) -> NormalizationResult:
        ...


def _number_words(value: int, numbers: Mapping[str, Any]) -> str | None:
    """0–999 словами по таблицам словаря; None — форма не покрыта данными."""

    if not 0 <= value <= 999:
        return None
    units = numbers.get("units") or []
    teens = numbers.get("teens") or []
    tens = numbers.get("tens") or []
    hundreds = numbers.get("hundreds") or []
    high, rest = divmod(value, 100)
    parts: list[str] = []
    if high:
        if high >= len(hundreds) or not hundreds[high]:
            return None
        parts.append(str(hundreds[high]))
    if rest >= 20:
        ten, unit = divmod(rest, 10)
        if ten >= len(tens) or not tens[ten]:
            return None
        parts.append(str(tens[ten]))
        if unit:
            if unit >= len(units) or not units[unit]:
                return None
            parts.append(str(units[unit]))
    elif rest >= 10:
        if rest - 10 >= len(teens) or not teens[rest - 10]:
            return None
        parts.append(str(teens[rest - 10]))
    else:
        if rest >= len(units) or not units[rest]:
            return None
        parts.append(str(units[rest]))
    return " ".join(parts)


class DictionaryNormalizer:
    """Нормализатор по pinned-словарю (данные, не код)."""

    def __init__(self, rules: Mapping[str, Any]) -> None:
        version = rules.get("version")
        if not isinstance(version, str) or not version:
            raise WorkerError(
                "config_invalid", "normalization dictionaries: нет строкового поля version"
            )
        self._version = version
        self._abbreviations = _string_map(rules.get("abbreviations"))
        self._brands = _string_map(rules.get("brands"))
        self._stress = _string_map(rules.get("stress"))
        numbers = rules.get("numbers")
        self._numbers: Mapping[str, Any] = numbers if isinstance(numbers, Mapping) else {}
        # Сокращения с точками/дефисами длиннее — применяются первыми.
        self._abbreviation_items = sorted(
            self._abbreviations.items(), key=lambda item: len(item[0]), reverse=True
        )
        self._brand_patterns = [
            (re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE), source, target)
            for source, target in self._brands.items()
        ]
        self._stress_patterns = [
            (re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE), source, target)
            for source, target in self._stress.items()
        ]

    @property
    def version(self) -> str:
        return self._version

    @classmethod
    def from_path(cls, path: Any) -> "DictionaryNormalizer":
        from pathlib import Path

        target = Path(path)
        try:
            rules = json.loads(target.read_text(encoding="utf-8"))
        except OSError as exc:
            raise WorkerError(
                "config_invalid", f"словарь нормализации не читается: {target}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise WorkerError(
                "config_invalid", f"словарь нормализации не JSON: {target}: {exc}"
            ) from exc
        if not isinstance(rules, Mapping):
            raise WorkerError(
                "config_invalid", f"словарь нормализации: ожидался объект: {target}"
            )
        return cls(rules)

    def __call__(self, text: str, language: str) -> NormalizationResult:
        if language != "ru":
            # v1 product voices are ru; другие языки не нормализуются молча.
            raise WorkerError(
                "input_invalid", f"language={language!r}: нормализация v1 только для ru"
            )
        applied: list[str] = []
        result = text
        for source, target in self._abbreviation_items:
            if source in result:
                result = result.replace(source, target)
                applied.append(f"abbreviation:{source}")
        for pattern, source, target in self._brand_patterns:
            result, count = pattern.subn(target, result)
            if count:
                applied.append(f"brand:{source}")
        number_applied: list[str] = []

        def replace_number(match: re.Match[str]) -> str:
            word = _number_words(int(match.group(0)), self._numbers)
            if word is None:
                return match.group(0)
            number_applied.append(f"number:{match.group(0)}")
            return word

        result = _NUMBER_RE.sub(replace_number, result)
        applied.extend(number_applied)
        for pattern, source, target in self._stress_patterns:
            result, count = pattern.subn(target, result)
            if count:
                applied.append(f"stress:{source}")
        return NormalizationResult(
            text=result,
            version=self._version,
            applied=tuple(applied),
        )


def _string_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkerError(
            "config_invalid", "normalization dictionaries: секция должна быть объектом"
        )
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise WorkerError(
                "config_invalid", "normalization dictionaries: ключ и значение — строки"
            )
        result[key] = item
    return result


__all__ = ["DictionaryNormalizer", "NormalizationResult", "Normalizer"]
