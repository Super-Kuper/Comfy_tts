# TTS Serverless — Qwen3-TTS 1.7B Base (voice cloning) + GigaAM-v3

RunPod serverless worker for the video_gen **tts** stand (`tts` modality): it
synthesizes one Russian phrase into 1–2 takes with the pinned **Qwen3-TTS 1.7B
Base** model **cloned from an uploaded reference voice**, applies explicit
**DSP-only prosody** (author `+`/`[[pause]]` markup and rate/gain/sentence
pause/whisper numbers resolved by the service — no instruct/emotional styles),
aligns the takes with **GigaAM-v3** (word timestamps + WER) and returns audio
plus execution-fact telemetry. It does not choose a model or a style (the model
is the image/volume; the style is resolved by the service), ships no weights and
has no external AI APIs at runtime. Contract: **`tts-serverless/v1.2`** (worker
**0.4.0**, additive `prosody`/markup + telemetry) —
[`contract/worker-contract.md`](contract/worker-contract.md).
Platform plans: `docs/tasks/2026-11-16-audio-sfx-tts.md` §4.1/§7.2 and
`docs/tasks/2026-11-18-voice-prosody.md` (variant A: Base + DSP) in the
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
3. **Use the prebuilt base image (recommended).** The heavy CUDA/PyTorch/
   qwen-tts layer is published once as
   `crappytackle/tts-serverless-base:0.3.4`; the repository `Dockerfile` only
   does `FROM` it and `COPY` the worker code. This keeps Hub builds inside the
   30-minute `docker build` limit:
   ```bash
   docker build -t crappytackle/tts-serverless:0.4.0 .
   docker push crappytackle/tts-serverless:0.4.0
   ```
   Rebuild/push the base only when dependencies change:
   ```bash
   IMAGE=crappytackle/tts-serverless-base:0.3.4 scripts/build-base.sh
   ```
   The base recipe is [`docker/Dockerfile.base`](docker/Dockerfile.base)
   (CUDA/PyTorch 2.7.1, `qwen-tts==0.1.1`, `transformers==4.57.3`,
   `soundfile`, `librosa`). No model weights are baked into the image — they
   stay on the network volume. A ready-to-use worker image is also published
   as `crappytackle/tts-serverless:0.4.0`; an endpoint can point at it directly,
   bypassing Hub builds.
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

| Field | Description |
|---|---|
| `request_id` | Phrase request UUID; echoed in the output. |
| `text` | Phrase text; normalized inside the worker with the versioned dictionaries in [`data/normalization.ru.json`](data/normalization.ru.json) (numbers, abbreviations, brands, dictionary stress). v1.2 author markup: `+` before a stressed Cyrillic vowel (`звон+ит`), `[[pause:<seconds>]]`; the `[[style:<id>]]` tag is stripped by the service. |
| `voice` | `oneOf`: `clone` `{mode, reference: {url, sha256, text, language}}` (executable on the Base image) or `preset` `{mode, speaker_id}` (compatibility shape; **unsupported** on Base — visible `voice_mode_unsupported`, no fallback). |
| `speed`, `takes`, `seed` | Synthesis parameters; 1–2 takes per phrase (plan §7.2). |
| `window` | Phrase window. **Tracing only** — the worker does not fit/stretch audio to the window; overflow is a service-side decision. |
| `render` | Requested `sample_rate`/`channels`/`wav`. |
| `aligner` | `{enabled, word_timestamps}`; disabled → `words=[]`, `wer=null`. |
| `prosody` | Optional v1.2 block of explicit DSP parameters resolved by the service from the DSP-only catalog: `style_id`, `style_catalog_version`, `rate` 0.5–2.0, `gain_db` −6…+6, `pitch_semitones` −2…+2, `sentence_pause_ms` 0–500, `whisper`. Absent → old v1 path; execution facts are echoed in `telemetry.prosody`. |
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
`license`, plus additive `voice_mode` and `reference_hash`, and the v1.2 facts
`normalization {version, applied}`, `stress_marks`, `pauses`, and `prosody` when
the input carried it). Empty/broken audio, a different take count or a missing
fact is an error, never a success. On the pinned Base image `voice.mode=preset`
fails visibly with `voice_mode_unsupported`. Full details and error codes:
[contract](contract/worker-contract.md). Examples:
[`contract/examples/`](contract/examples/) (v1.2:
[`request-prosody.json`](contract/examples/request-prosody.json) /
[`response-prosody.json`](contract/examples/response-prosody.json)).

## Prosody (v1.2, DSP-only)

Variant A keeps Qwen3-TTS Base voice cloning and adds **deterministic**
pre/post-processing instead of instruct/emotional styles. The service resolves
the author's `[[style:<id>]]` tag against the DSP-only catalog
(`voice-styles@2026-11-18`) and sends explicit numbers in `input.prosody`; the
worker executes them and never chooses a style. Emotional/instruct tags are not
in the registry — the service rejects them with `style_unknown` before a job.

**Author markup in `text`:**

- `+` immediately before the stressed Cyrillic vowel (`звон+ит`, `зам+ок`,
  `+это`) → U+0301 after the vowel (`звони́т`). One mark per word, `+ё` and
  `+` not before a vowel are `stress_invalid`; `+` outside a Cyrillic word
  (`C++`, `5+5`, `a+b`) stays literal. An explicit mark wins over the
  dictionary and is protected from replacement. Base follows U+0301 only
  partially (external ~21% measurement) — it is a priority hint, not a
  guarantee; `stress_marks`/`normalization` are the honest facts.
- `[[pause:<seconds>]]` (0.05–2.0, step 0.05, ≤ 6 per phrase, total ≤ 6 s)
  splits the phrase into segments; each is synthesized with the same
  voice/reference/takes/seed and joined with **exact** silence. GigaAM words are
  measured per segment and shifted by the actual segment durations+pauses;
  aggregate WER is the word-weighted mean. Bad position/step/limits are
  `pause_invalid`; markers are never stripped silently.

**DSP parameters:** effective rate = `speed × prosody.rate`, bounded 0.5–2.0
(else `prosody_rate`) and applied as **one** `librosa.effects.time_stretch`
(no auto-atempo); `gain_db` −6…+6 with a peak limiter (no clipping; actual gain
may be lower for loud takes); `sentence_pause_ms` 0–500 inserts exact silence
after `. ! ? …` during segmentation; `whisper` is an experimental FIR lowpass
+gain; `pitch_semitones` is applied only when non-zero (0 in variant A). No
output is ever fitted/stretched to the phrase window.

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

The suite covers the contract schemas and examples (preset, clone and v1.2
prosody shapes), handler happy path (takes, WAV facts, GigaAM words/WER from a
mock aligner), clone reference download/hash/size/duration failures with fake
downloader/decoder, preset-on-Base `voice_mode_unsupported`, engine takes and
target-rate control flow with injected fakes, empty result and broken audio
errors, "the worker does not choose a model" (strict input schema + engine from
the image), normalization, window non-fitting, aligner-disabled semantics and
the pinned model manifest. Health statuses (volume and pinned-file presence, no
synthesis) are covered too.

`tests/test_prosody.py` covers v1.2 without GPU/numpy/librosa: `+` stress
parsing (incl. `ё`/literal `+`/dictionary priority), `[[pause]]`
segments/limits/errors, sentence pauses, exact segment joining
(stdlib `wave`), word offsets and word-weighted WER, DSP ranges/effective
rate/peak-limited gain/whisper kernel (pure math), engine `_postprocess`
wiring with monkeypatched lazy functions, and handler telemetry/error-before-
synthesis. Local test deps: `pytest`, `jsonschema`, `httpx` (see
`pyproject.toml`).

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
   the RU quality pilot against the voice catalog (plan §11) and the prosody
   pilot (`docs/tasks/2026-11-18-voice-prosody.md` §11): pause accuracy ±20 ms,
   rate ±2%, gain ±0.5 dB/no clipping, the share of phrases where Base follows
   `+` (a fact to record, not a 100% promise), and listen verdicts for the DSP
   styles. Until this runs, the DSP path is covered by fakes/tests only.
6. **Service wiring:** phrase jobs (`kind='audio'`, `mode='tts'`), cloning
   snapshot (`voice_mode`/`voice_reference` + content_hash), presigned reference
   URL with TTL ≥ queue TTL, telemetry check (`voice_mode`/`reference_hash`),
   takes 1–2, selection/approval, `words`/`wer` into `voiceover_take` snapshots
   and A1 assembly; `policy` timeout ≥ `JOB_TIMEOUT_SECONDS`.
