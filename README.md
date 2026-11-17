# TTS Serverless — Qwen3-TTS 1.7B Base (voice cloning) + GigaAM-v3

RunPod serverless worker for the video_gen **tts** stand (`tts` modality): it
synthesizes one Russian phrase into 1–2 takes with the pinned **Qwen3-TTS 1.7B
Base** model **cloned from an uploaded reference voice**, aligns the takes with
**GigaAM-v3** (word timestamps + WER) and returns audio plus execution-fact
telemetry. It does not choose a model (the model is the image/volume), ships no
weights and has no external AI APIs at runtime. Contract:
**`tts-serverless/v1`** (worker **0.3.0**, cloning is additive) —
[`contract/worker-contract.md`](contract/worker-contract.md).
Platform plan: `docs/tasks/2026-11-16-audio-sfx-tts.md` §4.1/§7.2 in the
`video_gen` repository (separate repository: `/tts-serverless/` is in the parent
`.gitignore`).

## Models on the stand

| Model | Role | Volume files | License |
|---|---|---|---|
| Qwen3-TTS 12Hz 1.7B **Base** | synthesis via **voice cloning** (RU in 10 languages; Base has **no preset voices**) | `qwen3-tts-1.7b/model.safetensors` (~3.86 GB) + speech tokenizer (~0.68 GB) + tokenizer/configs | Apache-2.0 |
| GigaAM-v3 | aligner/QC: words + WER (Qwen3-TTS has no native word timings) | `gigaam-v3/pytorch_model.bin` (~0.45 GB) + tokenizer/config/modeling | MIT |

Pinned URL/revision/sha256, telemetry version and license:
[`models-manifest.txt`](models-manifest.txt). Chatterbox, if the pilot ever asks
for it, is a **separate image/stand** with its own manifest — one worker = one
engine (contract §1.1), no routing in a single worker and no silent fallback.

## Deploy

1. **Storage first.** Create a network volume (≥ 15 GB) in a datacenter that
   offers 24 GB GPUs (L4, A5000, RTX 3090/4090, A6000). The pinned set is
   ~5.1 GB; outputs/temp and future voices need headroom.
2. **Volume auto-populates on first boot.** `docker/entrypoint.sh` runs
   [`populate-volume.sh`](populate-volume.sh) with
   `VOLUME_AUTOPOPULATE=true` (default): missing files from
   `models-manifest.txt` are downloaded to `/runpod-volume/models` and every
   sha256 is verified; unpinned URLs/`main` are refused before GPU work. Set
   `VOLUME_AUTOPOPULATE=false` to prepare the volume manually from a CPU pod
   (`MODELS_ROOT=/runpod-volume/models ./populate-volume.sh`).
3. **Build and push the image.** Default base is the pinned CUDA/PyTorch runtime
   `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`; the image installs pinned
   `qwen-tts==0.1.1`, `transformers==4.57.3`, `soundfile==0.13.1`,
   `librosa==0.11.0` (torch comes from the base and is never reinstalled). The
   image carries no weights. A GPU-free structural build is available:
   `docker build --build-arg BASE_IMAGE=python:3.12-slim -t tts-serverless:stub .`
   (the model runtime is skipped because the stub base has no torch; a real job
   on it fails visibly with `engine_unavailable`).
4. **Publish on RunPod Hub (optional).** The Hub listing metadata is
   [`.runpod/hub.json`](.runpod/hub.json) (40 GB container disk; GPU pools
   `ADA_24,AMPERE_24,AMPERE_48,ADA_48_PRO,ADA_80_PRO`; host CUDA `13.2`/`13.0`
   — the image is CUDA 12.8, so older host CUDA is not offered) and the build
   test is [`.runpod/tests.json`](.runpod/tests.json) — it runs the health action
   on an NVIDIA H100 80GB (CUDA 13.2).
5. **Create the serverless endpoint** for the `tts` stand: endpoint type
   `QUEUE`, **`workers.max = 1`**, idle timeout 60–120 s, FlashBoot, endpoint
   timeout ≥ `policy.executionTimeout`. TTS batch = all phrase jobs of one
   voiceover request, so the 3–10 min cold start is amortized by the batch.
6. **Smoke-test** with `contract/examples/request-clone.json` on a warm worker
   and with the health action (see [Health](#health)); the Hub build test runs
   the same health request, then the RU quality pilot (plan §11) before enabling
   authors.

## Job input / output

```json
{
  "input": {
    "request_id": "3f6c2b7a-1d0e-4c55-9a77-2b1cd4a9e001",
    "text": "Швы затянуты. Следов нет.",
    "language": "ru",
    "voice": {
      "mode": "clone",
      "reference": {
        "url": "https://files.example/voice/reference-001.wav?X-Amz-Signature=...",
        "sha256": "<64 lowercase hex>",
        "text": "Ровно тот текст, который произнесён в референсе.",
        "language": "ru"
      }
    },
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
| `voice` | `oneOf`: `clone` `{mode, reference: {url, sha256, text, language}}` (executable on the Base image) or `preset` `{mode, speaker_id}` (compatibility shape; **unsupported** on Base — visible `voice_mode_unsupported`, no fallback). |
| `speed`, `takes`, `seed` | Synthesis parameters; 1–2 takes per phrase (plan §7.2). |
| `window` | Phrase window. **Tracing only** — the worker does not fit/stretch audio to the window; overflow is a service-side decision. |
| `render` | Requested `sample_rate`/`channels`/`wav`. |
| `aligner` | `{enabled, word_timestamps}`; disabled → `words=[]`, `wer=null`. |
| `policy` | Optional RunPod policy; the worker does not read it. |

### Reference voice (cloning)

The reference is a presigned HTTP(S) URL of the service's own storage object
plus the exact transcript, the file sha256 and the language. The worker:

1. downloads it with httpx (30 s timeout, redirects followed) and enforces
   **≤ 25 MiB** — otherwise `reference_too_large`;
2. recomputes **sha256** and compares it with `voice.reference.sha256`
   (= media item `content_hash` in the service) — mismatch is `reference_hash`;
3. decodes **WAV/FLAC/MP3/OGG** with soundfile/librosa, mixes to mono and
   resamples to **24 kHz** (Qwen3-TTS native);
4. enforces the reference duration **1–15 s** (product guidance 3–10 s is
   checked by the app on upload) — otherwise `reference_invalid`;
5. calls Qwen3-TTS Base `generate_voice_clone(text, language, ref_audio=(pcm,
   24000), ref_text)` once per take.

The URL and the audio content are never logged; the verified sha256 is echoed
in telemetry. Missing/invalid pieces are visible codes (`reference_missing`,
`reference_invalid`, `reference_hash`, `reference_too_large`) — the job never
falls back to a preset voice.

Result: `request_id` echo, 1–2 takes (each with exactly one of `audio_b64` /
`audio_url`, WAV facts, `seed`, GigaAM `words`, `wer`) and telemetry
(`engine`, `engine_version`, `aligner_version`, `model_hash`, `gpu_seconds`,
`license`, plus additive `voice_mode` and `reference_hash`). Empty/broken audio,
a different take count or a missing fact is an error, never a success. On the
pinned Base image `voice.mode=preset` fails visibly with
`voice_mode_unsupported`. Full details and error codes:
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

Reference limits (≤ 25 MiB, 1–15 s, 1–500 transcript chars, 30 s download
timeout) are contract constants, not endpoint settings.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest   # 0 GPU, no network, no runpod, no torch
```

The suite covers the contract schemas and examples (preset and clone shapes),
handler happy path (takes, WAV facts, GigaAM words/WER from a mock aligner),
clone reference download/hash/size/duration failures with fake
downloader/decoder, preset-on-Base `voice_mode_unsupported`, engine takes and
target-rate control flow with injected fakes, empty result and broken audio
errors, "the worker does not choose a model" (strict input schema + engine from
the image), normalization, window non-fitting, aligner-disabled semantics and
the pinned model manifest. Health statuses (volume and pinned-file presence, no
synthesis) are covered too. Local test deps: `pytest`, `jsonschema`, `httpx`
(see `pyproject.toml`).

`.runpod/` holds the RunPod Hub listing (`hub.json`) and the build test (`tests.json`).

## Remaining live steps (not verified in this repository)

1. **GPU runtime verification** (`TODO(live)` in `src/tts_worker/engine.py` is
   now wired but not executed on a GPU): load the pinned Base from the volume,
   run `generate_voice_clone` on a real reference, verify RU quality, latency
   and the exact qwen-tts/transformers pins on the CUDA image. Until it is run
   on a live endpoint this is unverified; failures stay visible
   (`engine_unavailable`/`engine_failed`), never a silent fallback.
2. **GigaAM-v3 aligner** (`TODO(live)` in `src/tts_worker/aligner.py`): ASR/CTC
   alignment → `words` + `wer`. Until then a job with `aligner.enabled=true`
   fails visibly with `aligner_unavailable`.
3. **Warm load at startup** (both models, 3–10 min cold start) is optional:
   the engine loads lazily on the first job; the phrase batch amortizes it.
4. **Single source for normalization data:** this repository pins
   `data/normalization.ru.json`; the service plan §10 puts the dictionary data in
   `shared/seeds/tts_dictionaries.json`. Export one pinned version from that
   source into the image and put its hash into the service snapshot.
5. **Measure** WER/latency/GPU seconds per phrase on the stand, cold/warm start,
   and the RU quality pilot against the voice catalog (plan §11).
6. **Service wiring:** phrase jobs (`kind='audio'`, `mode='tts'`), cloning
   snapshot (`voice_mode`/`voice_reference` + content_hash), presigned reference
   URL with TTL ≥ queue TTL, telemetry check (`voice_mode`/`reference_hash`),
   takes 1–2, selection/approval, `words`/`wer` into `voiceover_take` snapshots
   and A1 assembly; `policy` timeout ≥ `JOB_TIMEOUT_SECONDS`.
