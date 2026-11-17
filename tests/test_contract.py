"""Схемы контракта: примеры валидны, нарушения видны, схемы — draft 2020-12."""

from __future__ import annotations

import copy

import pytest
from helpers import contract_dir, load_json
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


@pytest.fixture(scope="module")
def input_schema() -> dict:
    return load_json(contract_dir() / "input.schema.json")


@pytest.fixture(scope="module")
def output_schema() -> dict:
    return load_json(contract_dir() / "output.schema.json")


@pytest.fixture(scope="module")
def input_validator(input_schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(input_schema)
    return Draft202012Validator(input_schema)


@pytest.fixture(scope="module")
def output_validator(output_schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(output_schema)
    return Draft202012Validator(output_schema)


def _assert_invalid(validator: Draft202012Validator, payload: object) -> None:
    with pytest.raises(ValidationError):
        validator.validate(payload)


def _request_input() -> dict:
    return copy.deepcopy(load_json(contract_dir() / "examples" / "request.json")["input"])


# --- Примеры -----------------------------------------------------------------


def test_request_example_is_valid(input_validator: Draft202012Validator) -> None:
    input_validator.validate(load_json(contract_dir() / "examples" / "request.json"))


@pytest.mark.parametrize("name", ["response.json", "response-url.json"])
def test_output_examples_are_valid(output_validator: Draft202012Validator, name: str) -> None:
    output_validator.validate(load_json(contract_dir() / "examples" / name))


# --- Вход --------------------------------------------------------------------


@pytest.mark.parametrize("field", ["request_id", "text", "language", "voice", "render"])
def test_required_input_fields(input_validator: Draft202012Validator, field: str) -> None:
    payload = _request_input()
    payload.pop(field)
    _assert_invalid(input_validator, {"input": payload})


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "not-a-uuid"),
        ("text", ""),
        ("text", "x" * 1001),
        ("language", "en"),
        ("speed", 0.4),
        ("speed", 2.1),
        ("takes", 0),
        ("takes", 3),
        ("seed", -1),
        ("aligner", {"enabled": True, "word_timestamps": True, "extra": 1}),
    ],
)
def test_bad_input_values(
    input_validator: Draft202012Validator, field: str, value: object
) -> None:
    payload = _request_input()
    payload[field] = value
    _assert_invalid(input_validator, {"input": payload})


def test_the_worker_cannot_be_given_a_model_choice(input_validator: Draft202012Validator) -> None:
    """Модель — свойство образа/тома: поля выбора модели во входе невалидны."""

    for field, value in (
        ("model", "qwen3-tts-0.6b"),
        ("model_slug", "chatterbox-multilingual-v3"),
        ("engine", "chatterbox"),
    ):
        payload = _request_input()
        payload[field] = value
        _assert_invalid(input_validator, {"input": payload})


@pytest.mark.parametrize(
    "voice",
    [
        {"mode": "clone", "speaker_id": 0},
        {"mode": "preset"},
        {"mode": "preset", "speaker_id": -1},
        {"mode": "preset", "speaker_id": 0, "reference": "https://files.example/ref.wav"},
    ],
)
def test_voice_shape(input_validator: Draft202012Validator, voice: dict) -> None:
    payload = _request_input()
    payload["voice"] = voice
    _assert_invalid(input_validator, {"input": payload})


@pytest.mark.parametrize(
    "render",
    [
        {"sample_rate": 16000, "channels": 2, "format": "wav"},
        {"sample_rate": 48000, "channels": 3, "format": "wav"},
        {"sample_rate": 48000, "channels": 2, "format": "mp3"},
        {"sample_rate": 48000, "channels": 2},
    ],
)
def test_render_shape(input_validator: Draft202012Validator, render: dict) -> None:
    payload = _request_input()
    payload["render"] = render
    _assert_invalid(input_validator, {"input": payload})


@pytest.mark.parametrize(
    "window",
    [
        {"start": -1, "end": 2.0},
        {"start": 1.0, "end": 0},
        {"start": 1.0},
        {"start": 1.0, "end": 2.0, "fit": True},
    ],
)
def test_window_shape_is_tracing_only(input_validator: Draft202012Validator, window: dict) -> None:
    payload = _request_input()
    payload["window"] = window
    _assert_invalid(input_validator, {"input": payload})


def test_window_is_optional(input_validator: Draft202012Validator) -> None:
    payload = _request_input()
    payload.pop("window")
    input_validator.validate({"input": payload})


def test_policy_is_accepted(input_validator: Draft202012Validator) -> None:
    payload = _request_input()
    input_validator.validate(
        {"input": payload, "policy": {"executionTimeout": 120000, "ttl": 3600000}}
    )


def test_aligner_is_optional_with_defaults(input_validator: Draft202012Validator) -> None:
    payload = _request_input()
    payload.pop("aligner")
    input_validator.validate({"input": payload})


# --- Выход -------------------------------------------------------------------


VALID_WORDS = [
    {"text": "Швы", "start_ms": 40, "end_ms": 230},
    {"text": "Следов", "start_ms": 700, "end_ms": 1020},
]
VALID_TAKE = {
    "take": 1,
    "audio_b64": "aGVsbG8=",
    "duration_seconds": 2.1,
    "sample_rate": 48000,
    "channels": 2,
    "seed": None,
    "words": copy.deepcopy(VALID_WORDS),
    "wer": 0.0,
}
VALID_TELEMETRY = {
    "engine": "qwen3_tts",
    "engine_version": "qwen3-tts-12hz-1.7b@2026-01",
    "aligner_version": "gigaam-v3@2026-04",
    "model_hash": "sha256:" + "f" * 64,
    "gpu_seconds": 1.7,
    "license": "Apache-2.0",
}


def _output(**overrides: object) -> dict:
    payload = {
        "request_id": "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001",
        "takes": [copy.deepcopy(VALID_TAKE)],
        "telemetry": copy.deepcopy(VALID_TELEMETRY),
    }
    payload.update(overrides)
    return payload


def test_valid_outputs(output_validator: Draft202012Validator) -> None:
    output_validator.validate(_output())
    url_take = copy.deepcopy(VALID_TAKE)
    url_take.pop("audio_b64")
    url_take["audio_url"] = "https://storage.example/voiceover/take-1.wav"
    output_validator.validate(_output(takes=[url_take]))


@pytest.mark.parametrize("field", ["request_id", "takes", "telemetry"])
def test_required_output_fields(output_validator: Draft202012Validator, field: str) -> None:
    payload = _output()
    payload.pop(field)
    _assert_invalid(output_validator, payload)


def test_take_audio_delivery_is_exclusive(output_validator: Draft202012Validator) -> None:
    take = copy.deepcopy(VALID_TAKE)
    take["audio_url"] = "https://storage.example/x.wav"
    _assert_invalid(output_validator, _output(takes=[take]))
    take = copy.deepcopy(VALID_TAKE)
    take.pop("audio_b64")
    _assert_invalid(output_validator, _output(takes=[take]))


@pytest.mark.parametrize("count", [0, 3])
def test_takes_count_matches_plan(output_validator: Draft202012Validator, count: int) -> None:
    _assert_invalid(output_validator, _output(takes=[copy.deepcopy(VALID_TAKE)] * count))


@pytest.mark.parametrize(
    "patch",
    [
        {"words": [{"text": "Швы", "start_ms": 40}]},
        {"words": [{"text": "", "start_ms": 40, "end_ms": 230}]},
        {"words": [{"text": "Швы", "start_ms": -1, "end_ms": 230}]},
        {"words": [{"text": "Швы", "start_ms": 40, "end_ms": 0}]},
        {"wer": 1.5},
        {"duration_seconds": 0},
        {"sample_rate": 0},
        {"channels": 9},
        {"take": 3},
    ],
)
def test_take_facts_validation(output_validator: Draft202012Validator, patch: dict) -> None:
    take = copy.deepcopy(VALID_TAKE)
    take.update(patch)
    _assert_invalid(output_validator, _output(takes=[take]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine", "unknown_tts"),
        ("engine_version", ""),
        ("model_hash", "abc"),
        ("gpu_seconds", 0),
        ("license", ""),
        ("aligner_version", 5),
    ],
)
def test_telemetry_bad_values(
    output_validator: Draft202012Validator, field: str, value: object
) -> None:
    telemetry = copy.deepcopy(VALID_TELEMETRY)
    telemetry[field] = value
    _assert_invalid(output_validator, _output(telemetry=telemetry))


@pytest.mark.parametrize("missing", sorted(VALID_TELEMETRY))
def test_telemetry_required_fields(
    output_validator: Draft202012Validator, missing: str
) -> None:
    telemetry = copy.deepcopy(VALID_TELEMETRY)
    telemetry.pop(missing)
    _assert_invalid(output_validator, _output(telemetry=telemetry))


def test_aligner_version_may_be_null_when_disabled(output_validator: Draft202012Validator) -> None:
    telemetry = copy.deepcopy(VALID_TELEMETRY)
    telemetry["aligner_version"] = None
    take = copy.deepcopy(VALID_TAKE)
    take["words"] = []
    take["wer"] = None
    output_validator.validate(_output(takes=[take], telemetry=telemetry))


def test_unknown_output_field_is_rejected(output_validator: Draft202012Validator) -> None:
    _assert_invalid(output_validator, _output(image="aGVsbG8="))
    take = copy.deepcopy(VALID_TAKE)
    take["model"] = "qwen3-tts"
    _assert_invalid(output_validator, _output(takes=[take]))


def test_health_request_example_is_valid(input_validator: Draft202012Validator) -> None:
    input_validator.validate(load_json(contract_dir() / "examples" / "health.request.json"))


def test_health_allows_no_phrase_fields(input_validator: Draft202012Validator) -> None:
    input_validator.validate({"input": {"action": "health"}})


def test_health_response_shape() -> None:
    payload = load_json(contract_dir() / "examples" / "health.response.json")
    assert payload["status"] in {"ok", "starting", "degraded"}
    assert isinstance(payload["volume_mounted"], bool)
    assert payload["engine"] in {"ready", "missing"}
    assert payload["aligner"] in {"ready", "missing"}
