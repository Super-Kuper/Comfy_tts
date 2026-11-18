# Worker contract — tts-serverless/v1.2

This document is the source of truth for the job input, output and errors of the
worker in this repository. The contract is versioned separately from the worker
release: an incompatible change is a new contract version (`v2`); additions
within a minor version keep the major `v1` and are documented here. The platform
plans are `docs/tasks/2026-11-16-audio-sfx-tts.md` §4.1/§7.2 and
`docs/tasks/2026-11-18-voice-prosody.md` (variant A: Base + DSP-only prosody) in
the `video_gen` repository.

The worker loads the **pinned Qwen3-TTS 1.7B Base** model from its own
image/volume, synthesizes one Russian phrase into 1–2 takes **cloned from an
uploaded reference voice** (Base has no preset voices), applies explicit
**DSP-only prosody** (author stress/pause markup + rate/gain/sentence pause/
whisper/pitch numbers resolved by the service), aligns the takes with
**GigaAM-v3** (words + WER) and returns audio + execution-fact telemetry. It
**never chooses a model or a style**: there is no model field in the contract,
the image is the model choice, and a different engine (e.g. Chatterbox) is a
separate image/stand — not a routing flag in this worker. Emotional/instruct
styles are not part of variant A; the service rejects them (`style_unknown`)
before a job exists. An empty/broken audio file or a different number of takes
is never a successful result. The clone voice shape is additive within
`tts-serverless/v1` (worker 0.3.0); the preset shape is kept in the schema for
compatibility but is **not executable** on the Base image — it fails visibly
with `voice_mode_unsupported`, never a silent fallback. Worker 0.4.0 adds
additive `prosody` + author markup (contract `v1.2`). JSON Schemas and examples:
[`input.schema.json`](input.schema.json),
[`output.schema.json`](output.schema.json), [`examples/`](examples/) (the
prosody job is [`request-prosody.json`](examples/request-prosody.json) /
[`response-prosody.json`](examples/response-prosody.json)).

## 1. Job input

`POST {RUNPOD_SERVERLESS_API_BASE}/v2/{endpoint_id}/run`

```json
{
  "input": {
    "request_id": "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001",
    "text": "Швы затянуты. Следов нет.",
    "language": "ru",
    "voice": {"mode": "clone", "reference": {"url": "https://files.example/voice/reference-001.wav?X-Amz-Signature=...", "sha256": "<64 lowercase hex>", "text": "Ровно тот текст, который произнесён в референсе.", "language": "ru"}},
    "speed": 1.0,
    "takes": 2,
    "seed": null,
    "window": {"start": 1.2, "end": 3.4},
    "render": {"sample_rate": 48000, "channels": 2, "format": "wav"},
    "aligner": {"enabled": true, "word_timestamps": true},
    "prosody": {
      "style_id": "confident",
      "style_catalog_version": "voice-styles@2026-11-18",
      "rate": 1.0,
      "gain_db": 1.0,
      "pitch_semitones": 0.0,
      "sentence_pause_ms": 80,
      "whisper": false
    }
  },
  "policy": {"executionTimeout": 120000, "ttl": 3600000}
}
```

| Field | Meaning |
|---|---|
| `request_id` | Application UUID of the phrase request; echoed in the output so a batched phrase run can be matched. |
| `text` | Phrase text (1–1000 chars). Normalized inside the worker (§1.2). v1.2 author markup: `+` before a stressed Cyrillic vowel and `[[pause:<seconds>]]` markers (§1.6); the `[[style:<id>]]` tag is stripped by the service and never reaches the worker. |
| `language` | `ru` in v1 (product voices are Russian). |
| `voice` | One of: clone `{mode: "clone", reference: {url, sha256, text, language}}` (executable on the Base image, §1.4) or preset `{mode: "preset", speaker_id}` (additive-compatible shape; **unsupported** on Base — visible `voice_mode_unsupported`, no fallback). |
| `speed` | Speech speed factor 0.5–2.0 (default 1.0). |
| `takes` | 1–2 takes (plan §7.2). Default 1. The worker returns exactly this many takes or fails. |
| `seed` | Optional seed or `null` (engine default sampling). The reported take seed is the execution fact. |
| `window` | `{start, end}` phrase window on the scenario timeline. **Tracing only**: the worker does not fit, stretch or trim the phrase to the window; overflow is handled by the service (`voiceover_phrase_overflow`). |
| `render` | `{sample_rate, channels, format}`: `22050/24000/32000/44100/48000`, `1/2`, `wav`. Qwen3-TTS is 24 kHz native — resampling to the requested rate is the engine boundary's job. |
| `aligner` | `{enabled, word_timestamps}`, defaults `true/true`. `enabled=false` → no GigaAM run, output `words=[]`, `wer=null`; `word_timestamps=false` → GigaAM still measures WER, output `words=[]`. |
| `prosody` | Optional v1.2 block: explicit DSP parameters of the author's style, resolved by the service from the DSP-only catalog (§1.5). Absent → old v1 path unchanged. Execution facts are reported in `telemetry.prosody`. |
| `policy` | Optional RunPod policy (`executionTimeout`, `ttl`). The worker does not read it; RunPod applies it before the handler runs. |

The preset shape is kept for additive compatibility only:

```json
"voice": {"mode": "preset", "speaker_id": 0}
```

On the pinned **Qwen3-TTS 1.7B Base** image there is no preset voice catalog, so
such a job fails visibly with `voice_mode_unsupported` (the job never falls back
to a preset — see §4).

`input` is strict (`additionalProperties: false`): fields such as `model`,
`model_slug` or `engine` are rejected with `input_invalid`, as are unknown keys
inside `voice` / `voice.reference` (the `clone` shape is exactly `url`, `sha256`,
`text`, `language`).

### 1.1. Worker and model are one image

The model is a property of the image/volume, not of a job: the image ships
pinned Qwen3-TTS 1.7B weights (or reads them from the network volume) and the
contract has no model selector. The plan allows a second engine (Chatterbox) only
as a **separate endpoint with a separate image/volume**, never as routing inside
one worker. If a future product needs engine switching, it is a new contract
field and a matching service change — not a silent fallback.

### 1.2. Text normalization

Normalization (numbers, abbreviations, brands, dictionary stress and the v1.2
author markup) happens **inside the worker** and is data-driven:
`data/normalization.ru.json` is versioned in the image (`ru-normalization@
2026-11-18`), code only applies the rules. Rules that are not covered by the
pinned dictionary stay as written (they are visible, not silently "fixed").
Since v1.2 the worker's `telemetry.normalization = {version, applied}` and
`telemetry.stress_marks` are execution facts, so the service snapshot can pin
reproduction; the data file remains the single worker-side source (the export
from `shared/seeds` is a separate service step).

### 1.3. Window is tracing only

`window` (`{start, end}`) is the phrase window on the scenario timeline. The
worker does **not** fit, stretch or trim the phrase to the window — reported
durations are facts of the synthesized file, and overflow is handled by the
service (`voiceover_phrase_overflow`). There is no `window.fit` flag.

### 1.4. Voice: clone reference (executable) and preset (unsupported)

Base voice cloning input:

```json
"voice": {
  "mode": "clone",
  "reference": {
    "url": "https://files.example/voice/reference-001.wav?X-Amz-Signature=...",
    "sha256": "<64 lowercase hex>",
    "text": "Ровно тот текст, который произнесён в референсе.",
    "language": "ru"
  }
}
```

- `url` — presigned **HTTP(S)** URL of the service's own storage object. The
  worker downloads it with httpx (timeout, redirects followed) and enforces
  **≤ 25 MiB** (`reference_too_large`). The URL and the reference content are
  never logged.
- `sha256` — lowercase hex sha256 of the file, equal to the media item
  `content_hash` in the service. The worker recomputes it and fails with
  `reference_hash` on mismatch; the verified value is echoed in
  `telemetry.reference_hash`.
- `text` — exact transcript of the recording (1–500 chars), used as `ref_text`
  for Qwen3-TTS ICL cloning.
- `language` — reference language (`ru`).
- The file is decoded as WAV/FLAC/MP3/OGG (soundfile/librosa), mixed to mono
  and resampled to **24 kHz** (Qwen3-TTS native) before it reaches the engine.
- Duration must be **1–15 s** (`reference_invalid` otherwise); the product
  guidance (3–10 s) is enforced by the app when uploading.
- Base has **no preset voices**: `{mode: "preset"}` fails visibly with
  `voice_mode_unsupported`. It is never mapped onto an arbitrary voice and
  never silently ignored. Voice design is out of scope (a separate
  model/stand).

### 1.5. Prosody block (v1.2, additive; DSP-only)

`prosody` is **optional**. The service resolves the author's style tag
`[[style:<id>]]` against the DSP-only catalog (`voice-styles@2026-11-18`) and
sends explicit numbers; the worker executes them and never resolves/chooses a
style. Emotional/instruct styles are not in variant A: the service rejects them
with `style_unknown` before a job and gates old images with
`prosody_unsupported` (capabilities), so this block is only sent to worker
`>= 0.4.0`.

```json
"prosody": {
  "style_id": "confident",
  "style_catalog_version": "voice-styles@2026-11-18",
  "rate": 1.0,
  "gain_db": 1.0,
  "pitch_semitones": 0.0,
  "sentence_pause_ms": 80,
  "whisper": false
}
```

| Field | Range | DSP effect in the worker |
|---|---|---|
| `style_id` | non-empty string | Echoed in `telemetry.prosody`; not a worker choice. |
| `style_catalog_version` | non-empty string | Echoed implicitly by the service snapshot; the worker validates the field. |
| `rate` | 0.5–2.0 | `effective_rate = input.speed × rate` must stay 0.5–2.0 (else `prosody_rate`); applied as **one** `librosa.effects.time_stretch`, no auto-atempo. |
| `gain_db` | −6…+6 | Gain with a peak limiter: the output peak never exceeds 0.999 FS (no clipping); the limiter may apply less than requested for loud takes. |
| `pitch_semitones` | −2…+2 | Bounded `librosa.effects.pitch_shift`, applied only when non-zero (variant A catalog uses 0). |
| `sentence_pause_ms` | 0–500 int | Exact silence inserted after sentence-ending punctuation (`.!?…`) during segmentation; 0 disables. |
| `whisper` | boolean | Experimental DSP: FIR lowpass (~3.5 kHz) + gain for the `whisper` style; marked experimental in the catalog/UI. |

The block is strict (`additionalProperties: false`) and all fields are required
when it is present: unknown/missing/out-of-range values are `input_invalid`
(schema) or `prosody_invalid` (worker defensive check), never silently
defaulted. Absent `prosody` → the old v1 path: no time-stretch from a style, no
gain/whisper/pitch.

### 1.6. Author markup in `text` (v1.2)

The plan text may carry two author markers; the service leaves them in `text`
and only strips the leading `[[style:<id>]]` tag.

**Stress:** `+` immediately before the stressed Cyrillic vowel
(`звон+ит`, `зам+ок`, `+это`) becomes the combining acute (U+0301) **after** the
vowel: `звони́т`. Rules:

- one mark per word; `+` must be directly before a Cyrillic vowel;
- `+ё`/`+Ё` is invalid (ё already carries stress) — `stress_invalid`;
- `+` outside a Cyrillic word (`C++`, `5+5`, `a+b`) stays a literal character;
- an explicit mark wins over the dictionary `stress` entry and is protected
  from dictionary replacement;
- facts in output: `telemetry.stress_marks` (count) and
  `telemetry.normalization.applied` (`stress:<word>`), plus the normalizer data
  version. Base follows U+0301 only partially (external ~21% measurement): the
  mark is a **priority hint**, not a guarantee — the service/UI says so.

**Pause:** `[[pause:<seconds>]]` (lowercase tag), 0.05–2.0, step 0.05 (the
canonical form is a decimal like `[[pause:0.4]]`; integer forms such as
`[[pause:2]]` are accepted as 2.0), ≤ 6 markers per phrase and ≤ 6 s total.
Markers split the phrase into segments:

- each segment is synthesized with the **same** voice/reference/takes/seed and
  the same `prosody` parameters;
- take #i of each segment is joined with **exact** silence
  (round(pause × sample_rate) frames, WAV PCM, same format) into take #i of the
  phrase; a phrase with a single segment and no pause returns the engine bytes
  unchanged;
- GigaAM word timings are measured per segment and shifted by the **actual**
  preceding segment durations + pauses; the aggregate WER is the word-weighted
  mean over segments (see §2.2);
- marker at the start/end, two markers in a row (empty segment), a malformed
  marker, more than 6 markers or more than 6 s total is `pause_invalid`; the
  marker is never silently stripped.

Style tags are not part of the worker contract: `[[style:<id>]]` is resolved by
the service and removed before the job. Unknown/invalid markup fails the job
visibly (`stress_invalid`/`pause_invalid`), so no GPU time is spent on a phrase
that the author wrote wrong.

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
    "license": "Apache-2.0",
    "voice_mode": "clone",
    "reference_hash": "<64 hex>",
    "normalization": {"version": "ru-normalization@2026-11-18", "applied": ["pause:0.4", "stress:звонит"]},
    "stress_marks": 1,
    "pauses": [0.4],
    "prosody": {
      "style_id": "confident",
      "effective_rate": 1.0,
      "gain_db": 1.0,
      "pitch_semitones": 0.0,
      "sentence_pause_ms": 80,
      "whisper": false
    }
  }
}
```

- `request_id` is an exact echo of the input.
- `takes` has exactly `input.takes` entries (1–2); a different count is
  `takes_mismatch`. Each take has exactly one of `audio_b64` / `audio_url`
  (inline vs S3 delivery, `OUTPUT_INLINE_*` / `S3_*`); no silent truncation.
- `words` are produced by **GigaAM-v3**, not by the TTS model, with absolute
  milliseconds from the take start. They are empty when the aligner is disabled
  or `word_timestamps=false`. With `[[pause]]`/sentence-pause segmentation they
  are measured per segment and shifted by the fact durations+pauses (§2.2).
- `wer` is the GigaAM-v3 word error rate against the normalized phrase; `null`
  when `aligner.enabled=false`. For segmented phrases it is the word-weighted
  mean (§2.2), never `null` just because the phrase was split.
- Audio bytes are WAV (PCM). The produced file facts (`duration_seconds`,
  `sample_rate`, `channels`) are read from the take's WAV header, not from
  `window` or `render`.
- `telemetry.voice_mode` is the executed mode (`clone` on this image);
  `telemetry.reference_hash` is the **verified** sha256 of the clone reference
  (raw lowercase hex, the same value the service put into `voice.reference.
  sha256` / media `content_hash`); it is **required for `voice_mode=clone`**
  and `null` for preset mode. See §2.1.

### 2.1. Telemetry facts

| Field | Measured from |
|---|---|
| `engine` | The TTS engine callable that executed the job (`qwen3_tts` in this image). |
| `engine_version` | The loaded engine/model version (pinned engine object; e.g. `qwen3-tts-12hz-1.7b@2026-01`). |
| `aligner_version` | The aligner that actually ran (`gigaam-v3@2026-04`); `null` when disabled. |
| `model_hash` | Measured sha256 of the executed TTS weight file, verified equal to the pinned manifest sha256 (`provision_hash_mismatch` otherwise), prefixed with `sha256:`. |
| `gpu_seconds` | Wall clock of synthesis + alignment inside the handler. |
| `license` | License identifier of the executed TTS weights from the pinned manifest row (`Apache-2.0`). |
| `voice_mode` | Voice mode the job executed: `clone` (Base) or `preset`. Preset is not executable on Base (`voice_mode_unsupported`) — the field is still additive in the contract. |
| `reference_hash` | Verified sha256 (lowercase hex) of the reference file used for cloning; **required when `voice_mode=clone`**, absent/`null` in preset mode. The value is the worker's own recomputation, not the request field. |
| `normalization` | v1.2 fact: `{version, applied[]}` from the pinned dictionary execution. `version` is the data version (`ru-normalization@2026-11-18`); `applied` lists the rules that actually fired (`abbreviation:*`, `brand:*`, `number:*`, `pause:<seconds>`, `stress:<word>`) in report order (dictionary rules, then author pause markers, then author stress marks). |
| `stress_marks` | v1.2 fact: number of author `+` marks converted to U+0301 (0 when none). |
| `pauses` | v1.2 fact: author `[[pause:<seconds>]]` values in text order. Sentence pauses are not duplicated here — they are covered by `prosody.sentence_pause_ms`. |
| `prosody` | v1.2 fact (only when input `prosody` was sent): `{style_id, effective_rate, gain_db, pitch_semitones, sentence_pause_ms, whisper}`. `effective_rate = input.speed × prosody.rate`; the service verifies it against the snapshot and tolerates float rounding. `gain_db` is the requested DSP parameter (the peak limiter may apply less to avoid clipping). |

A fact that cannot be measured fails the job with `telemetry_missing`; expected
values are never substituted. The aligner weight file is verified against the
pinned manifest too (a mismatch is `provision_hash_mismatch`).

### 2.2. Segments, word timings and WER (v1.2)

When `[[pause]]` markers or `sentence_pause_ms` split a phrase, the worker:

1. synthesizes every segment with the same voice/reference/`takes`/`seed` and
   the same `prosody`, then joins take #i across segments with exact silence
   (§1.6) — `duration_seconds` is read from the joined WAV header;
2. runs GigaAM on **each segment** (its own `words`/`wer`) and shifts the word
   timings by the actual duration of all preceding segments plus the exact
   pauses; if `aligner.enabled=false` → `words=[]`, `wer=null` (also for clone
   jobs when the service disables the aligner);
3. aggregates WER as the **word-weighted mean**:
   `Σ(wer_seg × words_seg) / Σ(words_seg)`, where `words_seg` is the number of
   aligned words of the segment (fallback: whitespace tokens of the segment
   text), rounded to 4 decimals. A `null` segment WER is ignored; if all
   segments are `null`, the phrase WER is `null`. A segment WER outside [0, 1]
   fails the job with `output_invalid` (no clamping).

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
| Prosody (v1.2) | `prosody_invalid` (malformed `prosody` block caught by the worker's defensive check), `prosody_rate` (`speed × prosody.rate` outside 0.5–2.0), `stress_invalid` (bad `+` mark), `pause_invalid` (bad `[[pause]]` marker/limits) |
| Model manifest | `model_file_missing`, `provision_hash_mismatch` |
| Voice reference | `reference_missing`, `reference_invalid`, `reference_hash`, `reference_too_large`, `voice_mode_unsupported` |
| Engine/aligner | `engine_unavailable` (runtime/model not installed or not loadable), `engine_failed` (model generation error), `aligner_unavailable` (visible when the live model code is not wired yet) |
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
| `TTS_DEVICE` | `cuda` | Torch device for the engine/aligner (`cpu` works for diagnostics but is very slow; the GPU image uses `cuda`). |
| `TTS_MAX_TEXT_CHARS` | `1000` | Hard cap on normalized input length (schema max is 1000). |
| `TTS_DICTIONARIES` | `/opt/tts-worker/data/normalization.ru.json` | Versioned normalization dictionaries (data, not code). |
| `JOB_TIMEOUT_SECONDS` | `120` | Synthesis/alignment time bound used by the engine call (endpoint timeout must be ≥). |
| `OUTPUT_INLINE_MAX_MB` / `OUTPUT_INLINE_BUDGET_MB` / `OUTPUT_URL_TTL_SECONDS` | `14` / `20` / `86400` | base64 vs presigned `audio_url` thresholds. |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_REGION`, `S3_SECURE` | — / `us-east-1` / auto | S3-compatible storage for large takes. |
| `CONTRACT_DIR` | next to the image/repository (`contract/`) | JSON Schemas used by the handler. |

Reference limits are contract constants, not endpoint settings: download ≤ **25
MiB**, duration **1–15 s**, reference text **1–500 chars**. The download timeout
is 30 s. A presigned reference URL must stay valid for the worker's execution
window (the service issues it with the queue TTL); the URL and audio content are
never logged.

`S3_ACCESS_KEY` and `S3_SECRET_KEY` are secrets: set them only in the endpoint
environment, never in the image, the repository or logs. Weights live on the
network volume (`models-manifest.txt`, `populate-volume.sh`), never in the image.

## 6. Version

Current contract version: **`tts-serverless/v1.2`** (worker release **0.4.0**,
Qwen3-TTS 1.7B Base cloning + DSP-only prosody). Input and output are additive
within a minor version:

- 0.3.0 added the clone `voice` shape, the required `telemetry.voice_mode` and
  the nullable `telemetry.reference_hash`;
- 0.4.0 / v1.2 added the optional input `prosody` block, author `+`/`[[pause]]`
  markup, and the additive telemetry `normalization`/`stress_marks`/`pauses`/
  `prosody` facts. Old v1 inputs (no `prosody`, no markup) execute exactly as
  before and old v1 outputs stay valid.

The service must not send `prosody` to an older image: it checks
`capabilities.prosody` / the worker version and returns `prosody_unsupported`
before placing the job. An incompatible change requires a new contract major
version and a matching update of the schemas, examples and this document.
