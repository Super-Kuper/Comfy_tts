"""Нормализация: правила — данные из pinned-словаря, код только применяет."""

from __future__ import annotations

import json

import pytest
from helpers import DATA

from tts_worker.errors import WorkerError
from tts_worker.normalization import DictionaryNormalizer


@pytest.fixture(scope="module")
def normalizer() -> DictionaryNormalizer:
    return DictionaryNormalizer.from_path(DATA / "normalization.ru.json")


def test_version_is_from_data(normalizer: DictionaryNormalizer) -> None:
    assert normalizer.version.startswith("ru-normalization@")


def test_numbers_become_words(normalizer: DictionaryNormalizer) -> None:
    assert normalizer("5", "ru").text == "пять"
    assert normalizer("21", "ru").text == "двадцать один"
    assert normalizer("999", "ru").text == "девятьсот девяносто девять"


def test_abbreviations(normalizer: DictionaryNormalizer) -> None:
    result = normalizer("ул. Ленина, д. 5", "ru")
    assert result.text == "улица Ленина, дом пять"
    assert "abbreviation:ул." in result.applied


def test_brands(normalizer: DictionaryNormalizer) -> None:
    assert normalizer("пост в VK", "ru").text == "пост в ВКонтакте"


def test_stress_marks(normalizer: DictionaryNormalizer) -> None:
    result = normalizer("замок", "ru")
    assert result.text != "замок"
    assert "\u0301" in result.text
    assert "stress:замок" in result.applied


def test_uncovered_forms_stay_visible(normalizer: DictionaryNormalizer) -> None:
    # 1000 вне таблицы 0–999: форма не выдумывается, остаётся как написана.
    assert normalizer("1000", "ru").text == "1000"


def test_plain_text_is_preserved(normalizer: DictionaryNormalizer) -> None:
    text = "Швы затянуты. Следов нет."
    assert normalizer(text, "ru").text == text


def test_non_ru_language_is_not_silently_normalized(normalizer: DictionaryNormalizer) -> None:
    with pytest.raises(WorkerError) as error:
        normalizer("hello", "en")
    assert error.value.code == "input_invalid"


def test_rules_are_data(tmp_path) -> None:
    rules = {
        "version": "test-ru@1",
        "abbreviations": {"т.д.": "так далее"},
        "brands": {"VK": "ВКонтакте"},
        "stress": {"замок": "за́мок"},
        "numbers": {
            "units": ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
            "teens": [],
            "tens": [],
            "hundreds": [],
        },
    }
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    normalizer = DictionaryNormalizer.from_path(path)
    assert normalizer.version == "test-ru@1"
    assert normalizer("7", "ru").text == "семь"
    assert normalizer("т.д.", "ru").text == "так далее"


def test_missing_dictionary_is_config_invalid(tmp_path) -> None:
    with pytest.raises(WorkerError) as error:
        DictionaryNormalizer.from_path(tmp_path / "nope.json")
    assert error.value.code == "config_invalid"


def test_broken_dictionary_is_config_invalid(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkerError) as error:
        DictionaryNormalizer.from_path(path)
    assert error.value.code == "config_invalid"


def test_dictionary_without_version_is_config_invalid(tmp_path) -> None:
    path = tmp_path / "noversion.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkerError) as error:
        DictionaryNormalizer.from_path(path)
    assert error.value.code == "config_invalid"
