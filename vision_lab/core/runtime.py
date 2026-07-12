"""Настройка окружения процесса обучения (ТЗ §6.4).

Вызывается один раз в точке входа обучения, ДО создания DataLoader'ов:
OpenCV и BLAS не должны конкурировать за ядра с DataLoader-воркерами.
"""

from __future__ import annotations

import os

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_threads() -> None:
    """cv2 в 0 потоков, BLAS/OpenMP в 1 (setdefault — явные настройки уважаем)."""
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass
