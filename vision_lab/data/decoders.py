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


#: Форматы, которые libjpeg/libpng умеют декодировать сразу в уменьшенном
#: масштабе. Для остальных reduce игнорируется — молча, потому что это
#: оптимизация, а не изменение семантики.
REDUCIBLE = {"jpeg", "jpg"}
#: Флаги OpenCV для декода в 1/2, 1/4 и 1/8 разрешения.
_REDUCE_FLAGS = {2: cv2.IMREAD_REDUCED_COLOR_2,
                 4: cv2.IMREAD_REDUCED_COLOR_4,
                 8: cv2.IMREAD_REDUCED_COLOR_8}


def _reduce_factor(path: str | Path, max_side: int) -> int:
    """Наибольший делитель из {1,2,4,8}, при котором длинная сторона >= max_side.

    Размер читается из заголовка файла, без декодирования пикселей.
    """
    try:
        from PIL import Image  # локальный импорт: нужен только этому пути
        with Image.open(str(path)) as im:
            long_side = max(im.size)
    except Exception:  # noqa: BLE001 — не смогли прочитать заголовок, декодируем как есть
        return 1
    factor = 1
    for f in (2, 4, 8):
        if long_side / f >= max_side:
            factor = f
        else:
            break
    return factor


def decode_with_cv2(path: str | Path, reduce: int = 1) -> np.ndarray:
    """PNG/JPEG/TIFF через OpenCV; 8- и 16-битные, серые разворачиваются в 3 канала.

    ``reduce`` из {1, 2, 4, 8} — декодировать сразу в 1/reduce разрешения
    (только JPEG; для прочих форматов OpenCV вернёт полный кадр).
    """
    flag = _REDUCE_FLAGS.get(reduce, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(str(path), flag)
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
    max_side: int | None = None,
) -> np.ndarray:
    """Декодирует по формату (или расширению) + опциональный resize (H, W).

    Resize здесь — для SSL-пути «decode → resize → tensor», где вся стохастика
    живёт на GPU (ТЗ §6.1); INTER_AREA корректен для даунскейла.

    ``max_side`` — декодировать сразу в уменьшенном масштабе, пока длинная
    сторона остаётся не меньше ``max_side``. Это НЕ то же, что ``image_size``:
    ``image_size`` ужимает уже декодированный кадр, а ``max_side`` не даёт
    развернуть его целиком. Для 24-мегапиксельного JPEG разница на замере —
    354 мс против 66 мс на снимок, и при обучении на таких данных именно декод,
    а не аугментации, оказывается узким местом.

    Соотношение сторон сохраняется: делитель целочисленный (1/2, 1/4, 1/8).
    Формат, который так декодировать нельзя, обрабатывается как раньше.
    """
    fmt = (fmt or Path(path).suffix.lstrip(".")).lower()
    decoder = DECODERS.get(fmt)
    if decoder is None:
        raise KeyError(
            f"Нет декодера для формата {fmt!r} (файл {path}). "
            f"Известные: {sorted(DECODERS)}. Добавьте через register_decoder()."
        )
    if max_side is not None and fmt in REDUCIBLE and decoder is decode_with_cv2:
        img = decoder(path, reduce=_reduce_factor(path, max_side))
    else:
        img = decoder(path)
    if image_size is not None:
        h, w = image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img
