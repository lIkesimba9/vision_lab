"""Пер-source выравнивание цвета (ТЗ §7.4).

Для каждого источника (устройство/клиника) заранее считаются канальные
mean/std; при загрузке изображение линейно переносится в глобальный референс —
междевайсный сдвиг убирается, внутридевайсный относительный цвет сохраняется.

Фиксы техдолга прототипа (требования ТЗ):

* источник берётся ТОЛЬКО из колонки ``source`` манифеста, никогда из пути;
* отсутствие статистик для source — ОШИБКА, а не тихий фолбэк на референс;
* загрузка/расчёт статистик — одна реализация здесь; eval-скрипты
  переиспользуют её, не копируют.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

#: Ключ глобального референса в файле статистик.
GLOBAL_KEY = "_global"


def load_source_stats(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Читает JSON ``{source: {"mean": [3], "std": [3]}, "_global": {...}}``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if GLOBAL_KEY not in raw:
        raise KeyError(f"В файле статистик {path} нет ключа {GLOBAL_KEY!r} (референс)")
    return {
        name: (np.asarray(v["mean"], np.float32), np.asarray(v["std"], np.float32))
        for name, v in raw.items()
    }


def compute_source_stats(dataset, batch_pixels: int = 0) -> dict:
    """Считает канальные mean/std по source для датасета БЕЗ transform/preprocessing.

    Возвращает dict, готовый к ``json.dump`` (включая ``_global`` по всем
    изображениям). Каждый источник обязан присутствовать в выборке.
    """
    sums: dict[str, list] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        img = item["image"]
        img = img.numpy().transpose(1, 2, 0) if hasattr(img, "numpy") else img
        px = img.reshape(-1, 3).astype(np.float64)
        for key in (item["source"], GLOBAL_KEY):
            acc = sums.setdefault(key, [0.0, 0.0, 0])
            acc[0] += px.sum(axis=0)
            acc[1] += (px**2).sum(axis=0)
            acc[2] += px.shape[0]
    out = {}
    for key, (s, s2, n) in sums.items():
        mean = s / n
        var = np.maximum(s2 / n - mean**2, 1e-12)
        out[key] = {"mean": mean.tolist(), "std": np.sqrt(var).tolist()}
    return out


class SourceAlignment:
    """Preprocessing-шаг: линейный перенос из статистик источника в референс."""

    def __init__(self, stats: str | Path | Mapping):
        self.stats = load_source_stats(stats) if isinstance(stats, (str, Path)) else {
            k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32))
            for k, v in stats.items()
        }
        if GLOBAL_KEY not in self.stats:
            raise KeyError(f"В статистиках нет ключа {GLOBAL_KEY!r} (референс)")

    def __call__(self, image: np.ndarray, sample: Mapping) -> np.ndarray:
        source = sample["source"]
        if source not in self.stats:
            raise KeyError(
                f"Нет статистик для source={source!r}. Пересчитайте файл статистик "
                f"(compute_source_stats) — тихий фолбэк на глобальный референс запрещён (ТЗ §7.4)."
            )
        m_src, s_src = self.stats[source]
        m_ref, s_ref = self.stats[GLOBAL_KEY]
        aligned = (image - m_src) / s_src * s_ref + m_ref
        return np.clip(aligned, 0.0, 1.0).astype(np.float32)
