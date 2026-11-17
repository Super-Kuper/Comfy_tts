# Worker contract — tts-serverless/v1

This document is the source of truth for the job input, output and errors of the
worker in this repository. The contract is versioned separately from the worker
release: an incompatible change is a new contract version (`v2`). The platform
plan is `docs/tasks/2026-11-16-audio-sfx-tts.md` §4.1/§7.2 in the `video_gen`
repository.

The worker loads the **pinned Qwen3-TTS 1.7B** model from its own image/volume,
synthesizes one Russian phrase into 1–2 takes, aligns the takes with
**GigaAM-v3** (words + WER) and returns audio + execution-fact telemetry. It
**never chooses a model**: there is no model field in the contract, the image is
the model choice, and a different engine (e.g. Chatterbox) is a separate
image/stand — not a routing flag in this worker. An empty/broken audio file or a
different number of takes is never a successful result. JSON Schemas and
examples: [`input.schema.json`](input.schema.json),
[`output.schema.json`](output.schema.json), [`examples/`](examples/).

## 1. Job input

`POST {RUNPOD_SERVERLESS_API_BASE}/v2/{endpoint_id}/run`

```json
{
  "input": {
    "request_id": "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001",
    "text": "Швы затянуты. Следов нет.",
    "language": "ru",
    "voice": {"mode": "preset", "speaker_id": 0},
    "speed": 1.0,
    "takes": 2,
    "seed": null,
    "window": {"start": 1.2, "end": 3.4},
    "render": {"sample_rate": 48000, "channels": 2, "format": "wav"},
    "aligner": {"enabled": true, "word_timestamps": true}
  },
  "policy": {"executionTimeout": 120000, "ttl": 3600000}
}
```

| Field | Meaning |
|---|---|
| `request_id` | Application UUID of the phrase request; echoed in the output so a batched phrase run can be matched. |
| `text` | Phrase text (1–1000 chars). Normalized inside the worker (§1.2). |
| `language` | `ru` in v1 (product voices are Russian). |
| `voice` | `{mode, speaker_id}`: `preset` + preset index in the pinned model's catalog. v1 has presets only. |
| `speed` | Speech speed factor 0.5–2.0 (default 1.0). |
| `takes` | 1–2 takes (plan §7.2). Default 1. The worker returns exactly this many takes or fails. |
| `seed` | Optional seed or `null` (engine default sampling). The reported take seed is the execution fact. |
| `window` | `{start, end}` phrase window on the scenario timeline. **Tracing only**: the worker does not fit, stretch or trim the phrase to the window; overflow is handled by the service (`voiceover_phrase_overflow`). |
| `render` | `{sample_rate, channels, format}`: `22050/24000/32000/44100/48000`, `1/2`, `wav`. Qwen3-TTS is 24 kHz native — resampling to the requested rate is the engine boundary's job. |
| `aligner` | `{enabled, word_timestamps}`, defaults `true/true`. `enabled=false` → no GigaAM run, output `words=[]`, `wer=null`; `word_timestamps=false` → GigaAM still measures WER, output `words=[]`. |
| `policy` | Optional RunPod policy (`executionTimeout`, `ttl`). The worker does not read it; RunPod applies it before the handler runs. |

`input` is strict (`additionalProperties: false`): fields such as `model`,
`model_slug`, `engine` or `voice.reference` are rejected with `input_invalid`.

### 1.1. Worker and model are one image

The model is a property of the image/volume, not of a job: the image ships
pinned Qwen3-TTS 1.7B weights (or reads them from the network volume) and the
contract has no model selector. The plan allows a second engine (Chatterbox) only
as a **separate endpoint with a separate image/volume**, never as routing inside
one worker. If a future product needs engine switching, it is a new contract
field and a matching service change — not a silent fallback.

### 1.2. Text normalization

Normalization (numbers, abbreviations, brands, stress) happens **inside the
worker** and is data-driven: `data/normalization.ru.json` is versioned in the
image, code only applies the rules. Rules that are not covered by the pinned
dictionary stay as written (they are visible, not silently "fixed"). The service
snapshot should pin the dictionary/worker version to make reproduction
auditable; that reference is out of scope for contract v1 (the worker's
`normalization` version is not part of the telemetry — `engine_version`,
`aligner_version` and `model_hash` are).

## 2. Job output

The handler returns one object that RunPod places in `output` of the status
response.

```json
{
  "request_id": "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001",
  "takes": [{
    "take": 1,
    "audio_b64": "<base64 wav>",
    "duration_seconds": 2.1,
    "sample_rate": 48000,
    "channels": 2,
    "seed": null,
    "words": [{"text": "Швы", "start_ms": 40, "end_ms": 230}],
    "wer": 0.0
  }],
  "telemetry": {
    "engine": "qwen3_tts",
    "engine_version": "qwen3-tts-12hz-1.7b@2026-01",
    "aligner_version": "gigaam-v3@2026-04",
    "model_hash": "sha256:<64 hex>",
    "gpu_seconds": 1.7,
    "license": "Apache-2.0"
  }
}
```

- `request_id` is an exact echo of the input.
- `takes` has exactly `input.takes` entries (1–2); a different count is
  `takes_mismatch`. Each take has exactly one of `audio_b64` / `audio_url`
  (inline vs S3 delivery, `OUTPUT_INLINE_*` / `S3_*`); no silent truncation.
- `words` are produced by **GigaAM-v3**, not by the TTS model, with absolute
  milliseconds from the take start. They are empty when the aligner is disabled
  or `word_timestamps=false`.
- `wer` is the GigaAM-v3 word error rate against the normalized phrase; `null`
  when `aligner.enabled=false`.
- Audio bytes are WAV (PCM). The produced file facts (`duration_seconds`,
  `sample_rate`, `channels`) are read from the take's WAV header, not from
  `window` or `render`.

### 2.1. Telemetry facts

| Field | Measured from |
|---|---|
| `engine` | The TTS engine callable that executed the job (`qwen3_tts` in this image). |
| `engine_version` | The loaded engine/model version (pinned engine object; e.g. `qwen3-tts-12hz-1.7b@2026-01`). |
| `aligner_version` | The aligner that actually ran (`gigaam-v3@2026-04`); `null` when disabled. |
| `model_hash` | Measured sha256 of the executed TTS weight file, verified equal to the pinned manifest sha256 (`provision_hash_mismatch` otherwise), prefixed with `sha256:`. |
| `gpu_seconds` | Wall clock of synthesis + alignment inside the handler. |
| `license` | License identifier of the executed TTS weights from the pinned manifest row (`Apache-2.0`). |

A fact that cannot be measured fails the job with `telemetry_missing`; expected
values are never substituted. The aligner weight file is verified against the
pinned manifest too (a mismatch is `provision_hash_mismatch`).

## 3. Health

Request (`POST /run`):

```json
{"input": {"action": "health"}}
```

Response:

```json
{"status": "ok", "volume_mounted": true, "engine": "ready", "aligner": "ready"}
```

- `status`: `ok` (network volume mounted and both pinned files present),
  `starting` (volume mounted but the engine/aligner files are still being
  populated — normal cold-start state) or `degraded` (volume not mounted).
- `volume_mounted`: whether `VOLUME_ROOT` exists.
- `engine` / `aligner`: `ready` when the pinned file from `models-manifest.txt`
  is on the volume, `missing` otherwise.
- Health does not hash weights, does not load a model and does not run
  synthesis or alignment, so the phrase fields are not required with
  `action: health`; it is safe as an endpoint smoke test and for diagnostics.
  RunPod Hub runs it on build ([`.runpod/tests.json`](../.runpod/tests.json)).

## 4. Errors and cancellation

Any handler error fails the RunPod job (`FAILED`); error codes are stable
lowercase identifiers:

| Group | Codes |
|---|---|
| Input/config | `input_invalid`, `config_invalid`, `text_empty` |
| Model manifest | `model_file_missing`, `provision_hash_mismatch` |
| Engine/aligner | `engine_unavailable`, `aligner_unavailable` (visible when the live model code is not wired yet) |
| Result | `audio_empty_result`, `takes_mismatch`, `output_empty`, `output_invalid` |
| Output | `output_storage_missing`, `output_storage_failed` |

- Cancellation (`POST /v2/{endpoint_id}/cancel/{run_id}`) fails the job; the
  handler does not swallow `asyncio.CancelledError`. `policy.executionTimeout`
  must be greater than the endpoint timeout used for synthesis.
- The handler validates its own output against
  [`output.schema.json`](output.schema.json) before returning; a schema
  violation is `output_invalid`.

## 5. Environment

The worker reads its settings from endpoint environment variables. Invalid values
fail the job with `config_invalid` and the variable name.

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_ROOT` / `VOLUME_ROOT` | `/runpod-volume/models` / `/runpod-volume` | Volume root and model root. |
| `MODELS_MANIFEST` | `/opt/tts-worker/models-manifest.txt` | Pinned model manifest shipped in the image. |
| `TTS_DEVICE` | `cuda` | Torch device for the engine/aligner (`cpu` is not supported by Qwen3-TTS but useful for diagnostics). |
| `TTS_MAX_TEXT_CHARS` | `1000` | Hard cap on normalized input length (schema max is 1000). |
| `TTS_DICTIONARIES` | `/opt/tts-worker/data/normalization.ru.json` | Versioned normalization dictionaries (data, not code). |
| `JOB_TIMEOUT_SECONDS` | `120` | Synthesis/alignment time bound used by the engine call (endpoint timeout must be ≥). |
| `OUTPUT_INLINE_MAX_MB` / `OUTPUT_INLINE_BUDGET_MB` / `OUTPUT_URL_TTL_SECONDS` | `14` / `20` / `86400` | base64 vs presigned `audio_url` thresholds. |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_REGION`, `S3_SECURE` | — / `us-east-1` / auto | S3-compatible storage for large takes. |
| `CONTRACT_DIR` | next to the image/repository (`contract/`) | JSON Schemas used by the handler. |

`S3_ACCESS_KEY` and `S3_SECRET_KEY` are secrets: set them only in the endpoint
environment, never in the image, the repository or logs. Weights live on the
network volume (`models-manifest.txt`, `populate-volume.sh`), never in the image.

## 6. Version

Current contract version: **`tts-serverless/v1`**. Input and output are additive
within a minor version. An incompatible change requires a new contract version
and a matching update of the schemas, examples and this document.
