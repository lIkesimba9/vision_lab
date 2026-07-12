"""Реестр декодеров изображений (ТЗ §7.3): один датасет на все форматы.

Формат берётся из колонки ``source_format_<modality>`` манифеста (фолбэк —
расширение файла). Все декодеры возвращают HWC RGB float32 в [0, 1]
(16-битные PNG нормируются на 65535 — важно для конвертированных DICOM).

DICOM в реестре НЕТ намеренно: он не декодируется на лету, только офлайн-этап
конвертации (см. :mod:`vision_lab.data.preprocessing.dicom`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

Decoder = Callable[[str | Path], np.ndarray]


def decode_with_cv2(path: str | Path) -> np.ndarray:
    """PNG/JPEG/TIFF через OpenCV; 8- и 16-битные, серые разворачиваются в 3 канала."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось декодировать изображение: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    max_value = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / max_value


def decode_npy(path: str | Path) -> np.ndarray:
    """Заранее сконвертированные массивы (.npy): HWC float32 [0,1] как есть."""
    arr = np.load(str(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


DECODERS: dict[str, Decoder] = {
    "png": decode_with_cv2,
    "jpeg": decode_with_cv2,
    "jpg": decode_with_cv2,
    "tiff": decode_with_cv2,
    "tif": decode_with_cv2,
    "webp": decode_with_cv2,
    "npy": decode_npy,
}


def register_decoder(fmt: str, decoder: Decoder) -> None:
    """Точка расширения: nifti, шардированные форматы и т.п."""
    DECODERS[fmt.lower()] = decoder


def decode_image(
    path: str | Path,
    fmt: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Декодирует по формату (или расширению) + опциональный resize (H, W).

    Resize здесь — для SSL-пути «decode → resize → tensor», где вся стохастика
    живёт на GPU (ТЗ §6.1); INTER_AREA корректен для даунскейла.
    """
    fmt = (fmt or Path(path).suffix.lstrip(".")).lower()
    decoder = DECODERS.get(fmt)
    if decoder is None:
        raise KeyError(
            f"Нет декодера для формата {fmt!r} (файл {path}). "
            f"Известные: {sorted(DECODERS)}. Добавьте через register_decoder()."
        )
    img = decoder(path)
    if image_size is not None:
        h, w = image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img
