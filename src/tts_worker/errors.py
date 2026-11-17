"""Стабильные коды ошибок воркера (contract/worker-contract.md §3).

Ошибка handler'а проваливает RunPod job (FAILED); код виден в сообщении и в
логах. Никаких «тихих» успехов: пустой/битый результат и не то число дублей —
исключение.
"""

from __future__ import annotations


class WorkerError(Exception):
    """Видимая ошибка воркера: стабильный код + человекочитаемое сообщение."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


__all__ = ["WorkerError"]
