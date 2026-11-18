"""Нормализация русского текста: правила — данные, код только применяет.

Словари лежат в ``data/normalization.ru.json`` (версия в файле, правится новой
версией файла, не правкой кода). Порядок: авторские метки ударения → сокращения
→ бренды → числа (0–999 по таблицам) → ударения словаря. Непокрытые формы
остаются как написаны — это видно в QC, а не «тихо исправляется» (contract
§1.2).

Авторская разметка (contract §1.5, plan §3):

* ``+`` непосредственно перед ударной кириллической гласной превращается в
  комбинирующий знак U+0301 **после** гласной (``звон+ит`` → ``звони́т``);
  явная метка приоритетнее словаря ударений и защищена от его замены;
* ``+`` вне кириллического слова (``C++``, ``5+5``, ``a+b``) остаётся
  литералом; ``+`` в кириллическом слове не перед гласной, ``+ё`` и вторая
  метка в слове — видимый ``stress_invalid``; ``stress_marks`` — факт числа
  меток, применённые правила видны в ``applied``;
* метки пауз ``[[pause:...]]`` проходят через нормализацию как защищённые
  спаны (числа внутри них не превращаются в слова) и попадают в ``applied``
  как ``pause:<секунды>``; их лимиты и ошибки — ``prosody.split_pause_markup``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .errors import WorkerError

_NUMBER_RE = re.compile(r"\b\d{1,3}\b")
_STRESS_MARK = "\u0301"
_CYRILLIC_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
_CYRILLIC_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_YO_LETTERS = "ёЁ"
#: Защищённые спаны авторских меток: правила словаря не должны трогать их
#: содержимое (например, число в ``[[pause:0.4]]``).
_PROTECTED_MARKUP_RE = re.compile(r"\[\[(?:pause|style)[^\]]*\]\]")
_PAUSE_APPLIED_RE = re.compile(r"\[\[pause:([0-9]+(?:\.[0-9]+)?)\]\]")


@dataclass(frozen=True)
class NormalizationResult:
    """Результат нормализации: текст, версия словаря, применённые правила, метки."""

    text: str
    version: str
    applied: tuple[str, ...] = ()
    stress_marks: int = 0


class Normalizer(Protocol):
    """Инъекция для тестов: текст + язык → нормализованный результат."""

    def __call__(self, text: str, language: str) -> NormalizationResult:
        ...


def _is_cyrillic(char: str) -> bool:
    return bool(char) and char in _CYRILLIC_LETTERS


def _trailing_cyrillic_word(text: str) -> str:
    """Кириллическое слово в конце строки (с комбинирующими знаками)."""

    index = len(text)
    chars: list[str] = []
    while index > 0:
        char = text[index - 1]
        if char in _CYRILLIC_LETTERS or char == _STRESS_MARK:
            chars.append(char)
            index -= 1
        else:
            break
    return "".join(reversed(chars))


def _marked_word(text: str, vowel_index: int) -> str:
    """Слово автора вокруг гласной: без ``+`` и без U+0301 (для ``applied``)."""

    start = vowel_index
    while start > 0 and (text[start - 1] in _CYRILLIC_LETTERS or text[start - 1] == "+"):
        start -= 1
    end = vowel_index + 1
    while end < len(text) and (text[end] in _CYRILLIC_LETTERS or text[end] == "+"):
        end += 1
    return text[start:end].replace("+", "").replace(_STRESS_MARK, "")


def apply_stress_markup(text: str) -> tuple[str, tuple[str, ...], int]:
    """Разобрать авторские ``+``-метки; вернуть текст, ``applied``, число меток."""

    output: list[str] = []
    applied: list[str] = []
    marks = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "+":
            output.append(char)
            index += 1
            continue
        next_char = text[index + 1] if index + 1 < length else ""
        if next_char == "+":
            run_end = index
            while run_end < length and text[run_end] == "+":
                run_end += 1
            previous = text[index - 1] if index > 0 else ""
            after = text[run_end] if run_end < length else ""
            if _is_cyrillic(previous) or _is_cyrillic(after):
                raise WorkerError(
                    "stress_invalid",
                    f"несколько меток подряд у кириллического слова: {text[index:run_end]!r}",
                )
            output.append(char)
            index += 1
            continue
        if next_char and next_char in _CYRILLIC_VOWELS:
            if next_char in _YO_LETTERS:
                raise WorkerError(
                    "stress_invalid",
                    f"+{next_char}: «ё» уже несёт ударение, дублировать знак нельзя",
                )
            marked_word = _trailing_cyrillic_word("".join(output))
            if _STRESS_MARK in marked_word:
                raise WorkerError(
                    "stress_invalid",
                    f"в слове {marked_word!r} больше одной метки ударения",
                )
            word = _marked_word(text, index + 1)
            output.append(next_char)
            output.append(_STRESS_MARK)
            applied.append(f"stress:{word}")
            marks += 1
            index += 2
            continue
        if _is_cyrillic(next_char):
            raise WorkerError(
                "stress_invalid",
                f"+{next_char}: метка должна стоять перед ударной гласной "
                "кириллического слова",
            )
        previous = text[index - 1] if index > 0 else ""
        if _is_cyrillic(previous):
            raise WorkerError(
                "stress_invalid",
                "«+» стоит после кириллической буквы, но не перед гласной; "
                "вне кириллического слова знак остаётся литералом",
            )
        output.append(char)
        index += 1
    return "".join(output), tuple(applied), marks


def _protected_spans(text: str) -> list[tuple[bool, str]]:
    """Разбить текст на (защищённый_спан, кусок) для правил словаря."""

    parts: list[tuple[bool, str]] = []
    cursor = 0
    for match in _PROTECTED_MARKUP_RE.finditer(text):
        if match.start() > cursor:
            parts.append((False, text[cursor : match.start()]))
        parts.append((True, match.group(0)))
        cursor = match.end()
    if cursor < len(text):
        parts.append((False, text[cursor:]))
    return parts


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

    def _apply_dictionary_rules(self, text: str, applied: list[str]) -> str:
        """Сокращения → бренды → числа → словарные ударения для одного куска."""

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
            # Явная авторская метка (U+0301) приоритетнее словаря: заменяем
            # только формы без комбинирующего знака, чтобы не перетереть её.
            def replace_stress(match: re.Match[str]) -> str:
                if _STRESS_MARK in match.group(0):
                    return match.group(0)
                return target

            result, count = pattern.subn(replace_stress, result)
            if count:
                applied.append(f"stress:{source}")
        return result

    def __call__(self, text: str, language: str) -> NormalizationResult:
        if language != "ru":
            # v1 product voices are ru; другие языки не нормализуются молча.
            raise WorkerError(
                "input_invalid", f"language={language!r}: нормализация v1 только для ru"
            )
        applied: list[str] = []
        # 1. Авторские метки ударения — до словаря: явная метка приоритетнее.
        result, stress_applied, stress_marks = apply_stress_markup(text)
        # 2. Правила словаря применяются к тексту, но не к защищённым меткам
        #    пауз (иначе число из [[pause:0.4]] стало бы словом).
        rebuilt: list[str] = []
        for is_protected, chunk in _protected_spans(result):
            if is_protected:
                rebuilt.append(chunk)
            else:
                rebuilt.append(self._apply_dictionary_rules(chunk, applied))
        result = "".join(rebuilt)
        # 3. Факты авторской разметки: pause:<секунды> в порядке текста.
        for match in _PAUSE_APPLIED_RE.finditer(result):
            applied.append(f"pause:{match.group(1)}")
        applied.extend(stress_applied)
        return NormalizationResult(
            text=result,
            version=self._version,
            applied=tuple(applied),
            stress_marks=stress_marks,
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


__all__ = [
    "DictionaryNormalizer",
    "NormalizationResult",
    "Normalizer",
    "apply_stress_markup",
]
