"""tts_worker — RunPod serverless воркер озвучки (contract tts-serverless/v1).

Пакет не импортирует runpod/httpx/torch на уровне модулей: handler и тесты
работают без GPU, сети и модели; тяжёлое — ленивые импорты в ``engine``/
``aligner`` и инъекции.
"""

from __future__ import annotations

__version__ = "0.2.1"

__all__ = ["__version__"]
