"""RunPod serverless entrypoint for the TTS worker.

The container boots with the pinned Qwen3-TTS 1.7B + GigaAM-v3 weights from the
network volume (TODO(live): warm load at startup, see README) and then runs this
file. The job handler itself lives in ``src/tts_worker/handler.py``; this module
only makes the ``tts_worker`` package importable (``PYTHONPATH`` is already set
in the image, the ``sys.path`` line covers a plain repository checkout) and
starts the RunPod serverless loop.

Job contract: ``contract/worker-contract.md`` (``tts-serverless/v1``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def main() -> None:
    """Start the RunPod serverless loop with the worker handler."""

    # Импорты внутри функции: `import handler` не требует runpod, если нужен
    # только модуль (структурные проверки образа, smoke).
    import runpod

    from tts_worker.handler import handler

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
