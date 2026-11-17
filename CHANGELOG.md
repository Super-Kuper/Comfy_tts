# Changelog

Изменения воркера `tts-serverless` (стенд `tts`). Формат —
[Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версии —
[SemVer](https://semver.org/lang/ru/).

## [0.3.1] — 2026-11-17

Исправлен Hub-билд: RunPod PyTorch база вместо `pytorch/pytorch`.

### Исправлено

- v0.3.0 не собрался в RunPod Hub (успешный clone, затем падение на этапе
  сборки; точный лог в консоли Hub). Базовый образ заменён на официальный
  RunPod `runpod/pytorch:2.2.0-rc.229-cu1281-torch291-ubuntu2404`
  (CUDA 12.8, torch 2.9.1): RunPod-сборщик тянет RunPod-образы из своего
  registry/кэша, а не общий Docker Hub `pytorch/pytorch` (риск rate limit /
  долгого pull).
- `torchaudio` ставится только если его нет в базе (теперь `2.9.1 --no-deps`);
  остальные пины (`qwen-tts==0.1.1`, `transformers==4.57.3`, `soundfile`,
  `librosa`) не менялись.
- Версия `0.3.1`; контракт, handler, скачивание/проверка референса, движок и
  тесты не менялись.

## [0.3.0] — 2026-11-17

Base-клонирование голоса (Qwen3-TTS 1.7B Base): контракт `tts-serverless/v1`
аддитивно, живой движок, референс по presigned URL, CUDA-образ.

### Добавлено

- **Контракт §1.4:** `input.voice` — `oneOf`: `{mode:"clone",reference:{url,
  sha256, text, language}}` или совместимый `{mode:"preset",speaker_id}`.
  Референс — presigned HTTP(S) URL, sha256 (64 hex, равен `media_items.
  content_hash`), точный текст 1–500 символов. Схемы, примеры
  (`request-clone.json`/`response-clone.json`) и контракт обновлены без смены
  строки версии контракта.
- **Референс** (`src/tts_worker/reference.py`): скачивание httpx (таймаут 30 с,
  redirects) с лимитом ≤25 МБ, проверка sha256, декодирование WAV/FLAC/MP3/OGG
  (soundfile/librosa), mono 24 кГц, длительность 1–15 с. Видимые коды
  `reference_missing`, `reference_invalid`, `reference_hash`,
  `reference_too_large`; URL и аудио не логируются.
- **Движок** (`src/tts_worker/engine.py`): `Qwen3TtsEngine` лениво грузит
  pinned Base с тома и вызывает `generate_voice_clone(text, language,
  ref_audio=(pcm, sr), ref_text)`; `ru` → `russian`; takes 1–2, seed через
  `torch.manual_seed`; явная скорость (`librosa.effects.time_stretch`),
  ресемплинг в `render.sample_rate`, WAV PCM s16le. Preset на Base —
  видимый `voice_mode_unsupported` (без подмены); сбой генерации —
  `engine_failed`.
- **Телеметрия §2.1:** аддитивные `voice_mode` (обязательное) и
  `reference_hash` (проверенный sha256, raw hex; `null` для preset).
- **Образ:** default base — pinned `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
  + `qwen-tts==0.1.1` (--no-deps), `transformers==4.57.3`,
  `accelerate==1.12.0`, `soundfile==0.13.1`, `librosa==0.11.0`; torch из базы не
  переустанавливается. Stub-путь `--build-arg BASE_IMAGE=python:3.12-slim`
  сохранён (runtime авто-пропускается без torch), добавлен структурный smoke
  импортов.
- **Hub:** `v0.3.0`, container disk 40 ГБ, host CUDA `13.2`/`13.0` (образ CUDA
  12.8); changelog/description обновлены.
- **Тесты:** fake downloader/decoder и fake engine; clone input, hash/size/
  duration/url, preset-on-Base, takes, телеметрия, схемы/примеры контракта,
  `_postprocess` control flow (без numpy/librosa). Сьют без GPU/сети/torch.

### Не проверено живьём

- `generate_voice_clone` на реальном GPU-стенде (качество RU, латентность,
  совместимость пинов qwen-tts/transformers с CUDA-образом) — не запускалось.
- GigaAM-v3 aligner остаётся `TODO(live)` (`aligner_unavailable`).

## [0.2.3] — 2026-11-17

Исправлен Hub-билд: test pod на H100 80GB вместо недоступной RTX 4090.

### Исправлено

- **Причина:** `.runpod/tests.json` просил `NVIDIA GeForce RTX 4090`; в момент
  сборки не было ни её инстансов, ни fallback-карт (`B300 SXM6 AC MIG 1g.34gb`,
  `RTX PRO 4500 Blackwell`). Ошибка RunPod Hub:
  `There were no instances available for your specified GPU ...`.
- **`tts`:** test pod — `NVIDIA H100 80GB HBM3` (CUDA 13.2; сток подтверждён
  сборкой audio-воркера). Тестовый pod не подключает том, поэтому
  `VOLUME_AUTOPOPULATE=false` — health не ждёт скачивания.
- Пулы листинга Hub расширены `ADA_48_PRO`/`ADA_80_PRO`, чтобы тестовая карта
  входила в опубликованные пулы; практические 24–48-ГБ пулы сохранены.
- Версия `0.2.3`; контракт, обработчик, тесты и манифест не менялись.

## [0.2.2] — 2026-11-17

Живой entrypoint: авто-популяция тома перед handler.

### Добавлено

- **`docker/entrypoint.sh`:** при `VOLUME_AUTOPOPULATE=true` (по умолчанию)
  запускает `populate-volume.sh` (pinned sha256 из `models-manifest.txt`), затем
  `exec handler`; `VOLUME_AUTOPOPULATE=false` — только ручной populate. Ошибка
  скачивания не валит контейнер: health остаётся доступным, job падает видимым
  `model_file_missing`.
- Hub env: `VOLUME_AUTOPOPULATE` (default `true`).

### Изменено

- Dockerfile: ENV `TTS_WORKER_DIR`/`VOLUME_AUTOPOPULATE`, `ENTRYPOINT` —
  `docker/entrypoint.sh` (прежний `python handler.py` убран).

### Проверка

`pytest` — 130 passed; `shellcheck` — только прежние SC2034 в
`populate-volume.sh`; stub-сборка и smoke-entrypoint зелёные. Живой синтез не
проверялся: движок/aligner всё ещё `TODO(live)`.

## [0.2.1] — 2026-11-17

Исправлен Hub-билд: test pod просил CUDA-версии и GPU, которых нет в наличии.

### Исправлено

- **Причина:** `.runpod/tests.json` просил `12.8/12.9/13.0` и
  `gpuTypeId=NVIDIA L4`; у пула `AMPERE_24` нет стока L4, а реальные host-CUDA
  в каталоге RunPod — `13.2`, `13.0`, `12.6`, `12.4`. Ни одна машина не
  совпала, и Hub вернул `All returned supply constraints` (попытки
  L4 → 3090 → A5000).
- **`tts`:** test pod — NVIDIA GeForce RTX 4090 (сток есть, CUDA 13.2);
  CUDA-список — только реальные версии `13.2/13.0/12.6/12.4`; пулы листинга
  расширены до `ADA_24,AMPERE_24,AMPERE_48`.
- Версия `0.2.1`; контракт `tts-serverless/v1`, обработчик, тесты и манифест не
  менялись; тестовый образ по-прежнему без CUDA. Живые шаги — прежние
  `TODO(live)`.

## [0.2.0] — 2026-11-17

Релиз для публикации на RunPod Hub: метаданные листинга и build-тест, плюс
health-проверка без синтеза.

### Добавлено

- **RunPod Hub** (`.runpod/hub.json`, `.runpod/tests.json`): листинг (GPU-пул
  `ADA_24,AMPERE_48`, CUDA 12.8–13.0, 25 GB container disk, env-переменные) и
  build-тест `action=health`.
- **Health** (`src/tts_worker/health.py`, contract §3): вход
  `{"action": "health"}` → `{status, volume_mounted, engine, aligner}`.
  Проверяются только смонтированный том и наличие pinned-файлов engine/aligner
  на нём (без хеширования, синтеза и выравнивания); статусы
  `ok`/`starting`/`degraded`.
- **Схема входа** допускает `action: health` без текста/голоса/рендера
  (аддитивно в `tts-serverless/v1`); примеры `health.request.json`/
  `health.response.json`, health-тесты и contract-тесты на health.
- Версия `0.2.0` (pyproject, Dockerfile, `__version__`, Hub-метаданные);
  README — раздел Health и шаг публикации через Hub.

### Не менялось

- Контракт генерации `tts-serverless/v1`, телеметрия, pinned-манифест и данные
  нормализации. Живой движок/выравнивание и CUDA-база образа — прежние
  live-шаги (`TODO(live)`).

## [0.1.0] — 2026-11-17

Первый релиз: контракт `tts-serverless/v1` (одна фраза → 1–2 дубля), сборка
под Qwen3-TTS 1.7B Base и GigaAM-v3 (words + WER), pinned-манифест тома.
Живой синтез и выравнивание ещё не подключены — `TODO(live)` в
`src/tts_worker/engine.py` и `src/tts_worker/aligner.py`; реальная задача
падает видимыми `engine_unavailable`/`aligner_unavailable`, а не тишиной.

### Добавлено

- **Контракт `tts-serverless/v1`** (`contract/`): вход `{request_id, text,
  language, voice, speed, takes, seed, window, render, aligner}`, выход
  `{request_id, takes[...], telemetry}`; дубль — `{audio_b64 | audio_url,
  duration_seconds, seed, words, wer}`, телеметрия — `{engine,
  engine_version, aligner_version, model_hash, gpu_seconds, license}`.
  JSON-схемы входа/выхода, примеры запроса и ответа (base64 и presigned URL),
  документ контракта с лимитами и кодами ошибок.
- **Handler** (`src/tts_worker/handler.py`): ровно `takes` дублей (1–2), факты
  WAV из заголовка, `window` — только трассировка (воркер не подгоняет и не
  растягивает фразу), выключенный `aligner` → `words=[]`, `wer=null`; пустой
  или битый звук, неверное число дублей, отсутствующий факт — ошибка, не
  успех. Модель не выбирается: её нет в контракте (строгая схема входа), она —
  свойство образа и тома.
- **Движок и aligner**: Qwen3-TTS 1.7B Base (Apache-2.0, RU в 10 языках) и
  GigaAM-v3 (MIT) за `TODO(live)`-хуками; подключение не меняет контракт.
  Нормализация текста — данные `data/normalization.ru.json` (числа,
  аббревиатуры, бренды, ударения; непокрытые формы остаются как написаны и
  видны в QC). Версия словаря в телеметрию не входит — её фиксирует снимок
  на стороне сервиса (контракт §1.2).
- **Телеметрия из фактов исполнения**: `engine`, `engine_version`,
  `aligner_version`, `model_hash` (sha256 исполненных весов, расхождение с
  манифестом — `provision_hash_mismatch`), `gpu_seconds`, `license`.
- **Выдача результата**: base64 с порогом `OUTPUT_INLINE_MAX_MB`, presigned
  `audio_url` в S3 для крупных дублей (`OUTPUT_URL_TTL_SECONDS`).
- **Том и веса**: `models-manifest.txt` с pinned URL, HF-ревизией и sha256 для
  Qwen3-TTS 1.7B (~3.86 GB + speech tokenizer ~0.68 GB) и GigaAM-v3
  (~0.45 GB); ~5.1 GB, том ≥ 15 GB. `populate-volume.sh` отказывает URL без
  `/resolve/<40-hex-commit>/`, проверяет sha256, идемпотентен и докачивает
  `.tmp` после обрыва.
- **Образ**: слой на `python:3.12-slim` для структурной сборки без GPU
  (CUDA-база, pinned torch и пакеты моделей — `TODO(live)`), без весов и
  секретов; ComfyUI не используется, наружу только RunPod handler.
- **Тесты**: 123 теста без GPU, сети, torch и runpod (схемы и примеры
  контракта, happy path дублей, пустой/битый звук, «воркер не выбирает
  модель», нормализация по данным файла, `window` без подгонки, выключенный
  aligner, манифест, storage); `ruff check` — чисто.

### Ограничения релиза

- Живой синтез и выравнивание не подключены (`TODO(live)`): нужны pinned
  CUDA-образ с torch/Qwen3-TTS/GigaAM и warm-load обеих моделей на старте
  (cold start 3–10 мин на стенде, амортизируется фразовым батчем).
- `data/normalization.ru.json` — baseline воркера; экспорт единого источника
  словарей (`shared/seeds`) и его хеш в снимке сервиса — отдельный шаг.
- Endpoint и том не созданы; замеры WER/latency/GPU-секунд и пилот 20–30 фраз
  (WER/MOS, план §11) не выполнены — внешний блокер приёмки.
