# TTS Serverless — Qwen3-TTS 1.7B + GigaAM-v3 worker

RunPod serverless worker for the video_gen **tts** stand (`tts` modality): it
synthesizes one Russian phrase into 1–2 takes with the pinned **Qwen3-TTS 1.7B
Base** model, aligns the takes with **GigaAM-v3** (word timestamps + WER) and
returns audio plus execution-fact telemetry. It does not choose a model (the
model is the image/volume), ships no weights and has no external AI APIs at
runtime. Contract: **`tts-serverless/v1`** —
[`contract/worker-contract.md`](contract/worker-contract.md).
Platform plan: `docs/tasks/2026-11-16-audio-sfx-tts.md` §4.1/§7.2 in the
`video_gen` repository (separate repository: `/tts-serverless/` is in the parent
`.gitignore`).

## Models on the stand

| Model | Role | Volume files | License |
|---|---|---|---|
| Qwen3-TTS 12Hz 1.7B Base | synthesis (RU in 10 languages, cloning-ready) | `qwen3-tts-1.7b/model.safetensors` (~3.86 GB) + speech tokenizer (~0.68 GB) + tokenizer/configs | Apache-2.0 |
| GigaAM-v3 | aligner/QC: words + WER (Qwen3-TTS has no native word timings) | `gigaam-v3/pytorch_model.bin` (~0.45 GB) + tokenizer/config/modeling | MIT |

Pinned URL/revision/sha256, telemetry version and license:
[`models-manifest.txt`](models-manifest.txt). Chatterbox, if the pilot ever asks
for it, is a **separate image/stand** with its own manifest — one worker = one
engine (contract §1.1), no routing in a single worker and no silent fallback.

## Deploy

1. **Storage first.** Create a network volume (≥ 15 GB) in a datacenter that
   offers 24 GB GPUs (L4, A5000, RTX 3090/4090, A6000). The pinned set is
   ~5.1 GB; outputs/temp and future voices need headroom.
2. **Populate the volume before traffic.** Run
   [`populate-volume.sh`](populate-volume.sh) with
   `MODELS_ROOT=/runpod-volume/models` (or set `VOLUME_AUTOPOPULATE=true` once
   the image entrypoint lands). The script refuses non-pinned URLs and verifies
   every sha256; a mismatch fails before the GPU is started.
3. **Build and push the image.** Requires a CUDA runtime base with pinned
   torch + Qwen3-TTS/GigaAM runtime packages (TODO(live) in the
   [Dockerfile](Dockerfile)); the image carries no weights.
4. **Publish on RunPod Hub (optional).** The Hub listing metadata is
   [`.runpod/hub.json`](.runpod/hub.json) (GPU pool `ADA_24,AMPERE_48`,
   CUDA 12.8–13.0, 25 GB container disk, endpoint env vars) and the build test
   is [`.runpod/tests.json`](.runpod/tests.json) — it runs the health action.
5. **Create the serverless endpoint** for the `tts` stand: endpoint type
   `QUEUE`, **`workers.max = 1`**, idle timeout 60–120 s, FlashBoot, endpoint
   timeout ≥ `policy.executionTimeout`. TTS batch = all phrase jobs of one
   voiceover request, so the 3–10 min cold start is amortized by the batch.
6. **Smoke-test** with `contract/examples/request.json` on a warm worker and
   with the health action (see [Health](#health)); the Hub build test runs the
   same health request, then the RU quality pilot (plan §11) before enabling
   authors.

## Job input / output

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

| Field | Description |
|---|---|
| `request_id` | Phrase request UUID; echoed in the output. |
| `text` | Phrase text; normalized inside the worker with the versioned dictionaries in [`data/normalization.ru.json`](data/normalization.ru.json) (numbers, abbreviations, brands, stress). |
| `voice` | `{mode: "preset", speaker_id}`. |
| `speed`, `takes`, `seed` | Synthesis parameters; 1–2 takes per phrase (plan §7.2). |
| `window` | Phrase window. **Tracing only** — the worker does not fit/stretch audio to the window; overflow is a service-side decision. |
| `render` | Requested `sample_rate`/`channels`/`wav`. |
| `aligner` | `{enabled, word_timestamps}`; disabled → `words=[]`, `wer=null`. |
| `policy` | Optional RunPod policy; the worker does not read it. |

Result: `request_id` echo, 1–2 takes (each with exactly one of `audio_b64` /
`audio_url`, WAV facts, `seed`, GigaAM `words`, `wer`) and telemetry
(`engine`, `engine_version`, `aligner_version`, `model_hash`, `gpu_seconds`,
`license`). Empty/broken audio, a different take count or a missing fact is an
error, never a success. Full details and error codes:
[contract](contract/worker-contract.md). Examples:
[`contract/examples/`](contract/examples/).

## Health

Request (`POST /run`): `{"input": {"action": "health"}}`

```json
{"status": "ok", "volume_mounted": true, "engine": "ready", "aligner": "ready"}
```

- `status`: `ok` (volume mounted, both pinned files present), `starting` (volume
  mounted but the engine/aligner files are still being populated — normal
  cold-start state) or `degraded` (volume not mounted);
- `volume_mounted`: whether `VOLUME_ROOT` exists;
- `engine` / `aligner`: `ready` when the pinned file from `models-manifest.txt`
  is on the volume, `missing` otherwise.

Health only checks the volume and file presence: it does not hash weights, load
a model or run synthesis/alignment. RunPod Hub runs it on build
([`.runpod/tests.json`](.runpod/tests.json)).

## Environment

Secrets only in the endpoint env; never in the image/repository/logs.

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_ROOT` / `MODELS_MANIFEST` | `/runpod-volume/models` / `/opt/tts-worker/models-manifest.txt` | Volume model root and pinned manifest. |
| `TTS_DEVICE` | `cuda` | Torch device. |
| `TTS_MAX_TEXT_CHARS` | `1000` | Input text cap. |
| `TTS_DICTIONARIES` | `/opt/tts-worker/data/normalization.ru.json` | Versioned normalization dictionaries. |
| `JOB_TIMEOUT_SECONDS` | `120` | Synthesis/alignment time bound (endpoint timeout ≥). |
| `OUTPUT_INLINE_MAX_MB` / `OUTPUT_INLINE_BUDGET_MB` / `OUTPUT_URL_TTL_SECONDS` | `14` / `20` / `86400` | base64 vs presigned `audio_url` thresholds. |
| `S3_*` (`S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_REGION`, `S3_SECURE`) | — | Storage for large takes. |

## Tests

```bash
python -m pytest            # 0 GPU, no network, no runpod, no torch
```

The suite covers the contract schemas and examples, handler happy path (takes,
WAV facts, GigaAM words/WER from a mock aligner), empty result and broken audio
errors, "the worker does not choose a model" (strict input schema + engine from
the image), normalization (numbers/abbreviations/brands/stress from the data
file), window non-fitting, aligner-disabled semantics and the pinned model
manifest. Health statuses (volume and pinned-file presence, no synthesis) are
covered too. Local test deps: `pytest`, `jsonschema` (see `pyproject.toml`).

`.runpod/` holds the RunPod Hub listing (`hub.json`) and the build test (`tests.json`).

## Remaining live steps (not in this repository yet)

1. **Engine/aligner code** (`TODO(live)` in `src/tts_worker/engine.py` and
   `src/tts_worker/aligner.py`): wire pinned Qwen3-TTS synthesis (resample
   24 kHz → `render.sample_rate`, WAV PCM) and GigaAM-v3 ASR/CTC forced
   alignment → `words` + `wer`. Until then a real job fails visibly with
   `engine_unavailable`/`aligner_unavailable` — no silent fallback.
2. **Warm load at startup** (both models, 3–10 min cold start) and the image
   entrypoint/GPU base with pinned torch + model packages.
3. **Single source for normalization data:** this repository pins
   `data/normalization.ru.json`; the service plan §10 puts the dictionary data in
   `shared/seeds/tts_dictionaries.json`. Export one pinned version from that
   source into the image and put its hash into the service snapshot.
4. **Measure** WER/latency/GPU seconds per phrase on the stand, cold/warm start,
   and the RU quality pilot against the voice catalog (plan §11).
5. **Service wiring:** phrase jobs (`kind='audio'`, `mode='tts'`), takes 1–2,
   selection/approval, `words`/`wer` into `voiceover_take` snapshots and A1
   assembly; `policy` timeout ≥ `JOB_TIMEOUT_SECONDS`.
