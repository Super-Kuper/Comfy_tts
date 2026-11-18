# Changelog

Изменения воркера `tts-serverless` (стенд `tts`). Формат —
[Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версии —
[SemVer](https://semver.org/lang/ru/).

## [0.4.0] — 2026-11-18

Просодия варианта A: авторская разметка `+`/`[[pause]]` и DSP-only стили.
Контракт `tts-serverless/v1` → аддитивно `v1.2`; старый input без `prosody` и
без разметки работает как раньше.

### Добавлено

- **Контракт v1.2** (`contract/`): необязательный `input.prosody`
  `{style_id, style_catalog_version, rate 0.5–2.0, gain_db −6…+6,
  pitch_semitones −2…+2, sentence_pause_ms 0–500, whisper}` — строгий объект,
  который сервис собирает из DSP-only каталога (`voice-styles@2026-11-18`);
  воркер стиль не выбирает и эмоциональные/instruct-стили не поддерживает
  (вне варианта A). Схемы входа/выхода, примеры
  `request-prosody.json`/`response-prosody.json`, `worker-contract.md` §1.5/
  §1.6/§2.2/§4/§6 обновлены; версия контракта `tts-serverless/v1.2`.
- **Авторское ударение:** `+` непосредственно перед ударной кириллической
  гласной → U+0301 после гласной (`звон+ит` → `звони́т`); одна метка на слово,
  `+ё`/`+` не перед гласной/вторая метка — видимый `stress_invalid`; `+` вне
  кириллического слова (`C++`, `5+5`, `a+b`) остаётся литералом; явная метка
  приоритетнее словаря и защищена от его замены. Факты:
  `telemetry.stress_marks`, `normalization.applied` (`stress:<word>`).
- **Авторские паузы:** `[[pause:<секунды>]]` (0.05–2.0, шаг 0.05, ≤6 на фразу,
  суммарно ≤6 с, не в начале/конце и не подряд) режут фразу на сегменты; каждый
  синтезируется тем же голосом/референсом/дублями/seed, дубль #i склеивается с
  **точной** тишиной (round(pause × sample_rate) кадров). Слова GigaAM —
  по сегментам со смещением на фактические длительности+паузы; WER склейки —
  взвешенное по числу слов среднее. Ошибки — видимый `pause_invalid`; метки не
  срезаются молча. `sentence_pause_ms` вставляет ту же точную тишину после
  `. ! ? …` на этапе сегментации.
- **DSP** (`src/tts_worker/dsp.py`, новые примитивы с ленивыми
  numpy/librosa/soundfile): effective rate = `speed × prosody.rate` в 0.5–2.0
  (иначе видимый `prosody_rate`) — **один** `time_stretch`, без авто-atempo;
  gain −6…+6 dB с пик-лимитером (пик ≤ 0.999 FS, клиппинга нет); bounded
  pitch shift только при ненулевом значении (в каталоге A — 0); экспериментальный
  whisper — FIR lowpass ~3.5 кГц + gain. Склейка сегментов — stdlib `wave`,
  поэтому тесты идут без numpy/librosa.
- **Телеметрия** (аддитивно): `normalization {version, applied[]}`,
  `stress_marks`, `pauses[]`, `prosody {style_id, effective_rate, gain_db,
  pitch_semitones, sentence_pause_ms, whisper}` (последнее — только когда input
  `prosody` был). `normalization.version = ru-normalization@2026-11-18`.
- **Новые коды ошибок:** `prosody_invalid`, `prosody_rate`, `stress_invalid`,
  `pause_invalid`.
- **Нормализатор:** `NormalizationResult.stress_marks`; метки пауз защищены от
  правил чисел/словаря; версия данных `ru-normalization@2026-11-18` (содержимое
  словарей не менялось — уточнена заметка, что разметка это код воркера).
- **Тесты:** `tests/test_prosody.py` (нормализатор/ударения, паузы и лимиты,
  паузы предложений, точная склейка, смещение слов/взвешенный WER, DSP
  effective rate/gain/whisper/порядок `_postprocess`, engine+handler
  интеграция, телеметрия, ошибки до синтеза), обновлены
  `tests/test_contract.py` и `tests/test_handler.py`; v1.2-примеры валидны.
  GPU-free: движок/аллайнер — инъекции, DSP-функции подменяются.
- Версия воркера **0.4.0** (`pyproject.toml`, `__init__.__version__`,
  `Dockerfile`, Hub-метаданные); README — раздел Prosody, таблица полей и
  телеметрии.

### Не менялось

- Qwen3-TTS Base + клонирование, GigaAM-v3, preset-on-Base
  `voice_mode_unsupported`, `voice_mode`/`reference_hash`, лимиты референса,
  доставка base64/URL, pinned-манифест и база образа `0.3.4`. Окно фразы
  по-прежнему только трассировка, авто-подгонки/atempo нет.

### Проверка

`pytest` — 365 passed без GPU/сети/torch/numpy/librosa; `ruff check src tests`
— чисто. Живой пилот просодии (точность пауз/rate/gain, следование U+0301,
листен стилей) остаётся внешним GPU-этапом.

## [0.3.5] — 2026-11-17

Hub-тест: GPU с фактическим стоком (A100 SXM 80GB).

### Исправлено

- Hub-тест `.runpod/tests.json` просил `NVIDIA H100 80GB HBM3` и fallback
  H100 NVL/PCIe — все вернули `supply constraints` (нет инстансов).
- Тест переведён на `NVIDIA A100-SXM4-80GB`; доступность проверена реальным
  `POST /v2/pods` (SECURE, CUDA 13.2) с последующим terminate.
- `allowedCudaVersions` — `13.2`/`13.0`; в `hub.json` добавлен пул `AMPERE_80`,
  чтобы тестовая карта входила в опубликованные пулы.
- Версия образа воркера `0.3.5`; код/контракт/база `0.3.4` не менялись.

## [0.3.4] — 2026-11-17

Предсобранный base-образ + тонкий Dockerfile.

### Добавлено

- `docker/Dockerfile.base` — тяжёлый слой (CUDA/PyTorch 2.7.1, `runpod`,
  `transformers==4.57.3`, `qwen-tts==0.1.1`, `librosa`/`soundfile`) собирается
  один раз и публикуется как `crappytackle/tts-serverless-base:0.3.4`.
- `scripts/build-base.sh` — сборка и push базового образа.
- Корневой `Dockerfile` теперь только `FROM crappytackle/tts-serverless-base:0.3.4`
  + `COPY` кода; Hub-билд не устанавливает зависимости и не упирается в
  30-минутный лимит.
- Полный готовый образ воркера также публикуется как
  `crappytackle/tts-serverless:0.3.4` — endpoint можно перевести на него
  напрямую, без Hub-сборки.
- Версия `0.3.4`; контракт, handler, движок и тесты не менялись.

## [0.3.3] — 2026-11-17

Уменьшен образ: возвращена база `pytorch/pytorch`.

### Исправлено

- `runpod/pytorch` база v0.3.1 весит ~7.9 ГБ сжатой и тянет лишние слои;
  Hub-билд тратит на её pull больше 30-минутного лимита. Возвращена
  `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` (~4.2 ГБ сжатой).
- `torchaudio` fallback — `2.7.1` под torch базы (а не 2.9.1).
- Сохранён фикс v0.3.2: `cryptography`/`runpod` ставятся с `--ignore-installed`,
  чтобы не удалять Debian-пакет без RECORD.
- Версия `0.3.3`; контракт, handler, движок и тесты не менялись.

## [0.3.2] — 2026-11-17

Исправлен Hub-билд: Debian `cryptography` без RECORD.

### Исправлено

- RunPod PyTorch база ставит `cryptography 41.0.7` из Debian (без RECORD);
  `pip install runpod==2.2.1` требует `cryptography>=50` и падал с
  `error: uninstall-no-record-file` при попытке удалить системный пакет.
- Теперь `cryptography` и `runpod` ставятся с `--ignore-installed`: новые копии
  в `/usr/local/lib/python3.12/dist-packages`, системный Debian-пакет не
  удаляется. Остальные пины и базовый образ не менялись.
- Версия `0.3.2`; контракт, handler, движок и тесты не менялись.

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
