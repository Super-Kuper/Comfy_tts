# Changelog

Изменения воркера `tts-serverless` (стенд `tts`). Формат —
[Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версии —
[SemVer](https://semver.org/lang/ru/).

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
