"""Обрезка виньетки — снятие геометрической подписи устройства (ТЗ §7.4).

Зачем. В дерматоскопии кадр снимают камерой через тубус прибора: поле зрения
камеры шире выходного зрачка тубуса, и по краям в кадр попадает его чёрная
внутренняя стенка. Форма и толщина этого кольца зависят от модели прибора и от
того, как он закреплён, — то есть виньетка это **подпись устройства прямо в
кадре**. Три следствия:

1. Модель может опознавать источник по углам кадра вместо самого образования
   (shortcut learning), и метрики на смеси источников оказываются завышенными.
2. Чёрное поле съедает разрешение: при ресайзе к входу сети на полезную часть
   кадра приходится тем меньше пикселей, чем толще кольцо.
3. Любая статистика по кадру (среднее, std, оценка освещения в
   :mod:`~vision_lab.data.preprocessing.color_constancy`) считается вместе с
   чернотой и оказывается смещённой.

Шаг детерминированный и применяется ОДИНАКОВО на train и inference, как и
остальная предобработка: кадры без виньетки проходят через него без изменений.

Пример::

    from vision_lab.data.preprocessing import VignetteCrop

    dataset = ManifestDataset(..., preprocessing=[VignetteCrop(), ColorConstancy()])
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

#: Порог «тёмного» пикселя в шкале [0, 1].
DARK_THRESHOLD = 0.15
#: Какая доля строки/столбца должна быть светлой, чтобы считать её частью кадра.
COVERAGE = 0.5
#: Минимальная доля тёмных пикселей, при которой вообще имеет смысл обрезать.
MIN_DARK_FRACTION = 0.05


def vignette_bbox(
    image: np.ndarray,
    dark_threshold: float = DARK_THRESHOLD,
    coverage: float = COVERAGE,
    min_dark_fraction: float = MIN_DARK_FRACTION,
    inscribe: bool = True,
) -> tuple[int, int, int, int]:
    """Границы полезной части кадра: ``(x0, y0, x1, y1)``, полуинтервал по x1/y1.

    Яркость берётся как максимум по каналам — так цветной блик от геля не
    принимается за темноту, а чёрная стенка тубуса остаётся тёмной во всех
    каналах. Строка (столбец) считается частью кадра, если доля светлых пикселей
    в ней не меньше ``coverage``; это устойчивее к несимметричной и смещённой
    виньетке, чем поиск окружности.

    ``inscribe=True`` дополнительно вписывает прямоугольник в эллипс, заданный
    найденной рамкой: у круглого поля зрения углы рамки всё ещё чёрные, и
    вписанный прямоугольник (в 1/sqrt(2) по каждой полуоси) их отрезает.

    Если тёмных пикселей меньше ``min_dark_fraction`` — виньетки нет, и
    возвращается весь кадр.
    """
    if image.ndim == 3:
        gray = image.max(axis=2)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"ожидается HW или HWC, получено {image.shape}")

    if gray.dtype != np.float32 and gray.dtype != np.float64:
        gray = gray.astype(np.float32) / 255.0

    h, w = gray.shape
    full = (0, 0, w, h)
    bright = gray > dark_threshold
    if (1.0 - bright.mean()) < min_dark_fraction:
        return full

    rows = np.flatnonzero(bright.mean(axis=1) >= coverage)
    cols = np.flatnonzero(bright.mean(axis=0) >= coverage)
    if rows.size == 0 or cols.size == 0:
        return full

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    if inscribe:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0 / math.sqrt(2.0), (y1 - y0) / 2.0 / math.sqrt(2.0)
        x0, x1 = int(round(cx - hw)), int(round(cx + hw))
        y0, y1 = int(round(cy - hh)), int(round(cy + hh))

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:  # вырожденный результат — безопаснее не трогать
        return full
    return x0, y0, x1, y1


class VignetteCrop:
    """Preprocessing-шаг для :class:`~vision_lab.data.manifest.ManifestDataset`.

    Кадры без тёмной рамки возвращаются без изменений, поэтому шаг безопасно
    включать для смеси источников, где виньетка есть лишь у части снимков.
    """

    def __init__(
        self,
        dark_threshold: float = DARK_THRESHOLD,
        coverage: float = COVERAGE,
        min_dark_fraction: float = MIN_DARK_FRACTION,
        inscribe: bool = True,
    ):
        self.dark_threshold = dark_threshold
        self.coverage = coverage
        self.min_dark_fraction = min_dark_fraction
        self.inscribe = inscribe

    def __call__(self, image: np.ndarray, sample: Mapping) -> np.ndarray:
        x0, y0, x1, y1 = vignette_bbox(
            image,
            dark_threshold=self.dark_threshold,
            coverage=self.coverage,
            min_dark_fraction=self.min_dark_fraction,
            inscribe=self.inscribe,
        )
        return image[y0:y1, x0:x1]
