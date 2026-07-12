"""Офлайн-конвертация DICOM → 16-bit PNG (ТЗ §7.3).

DICOM не декодируется на лету. Этот модуль — офлайн-этап «raw → processed»:
параметры конвертации возвращаются словарём и фиксируются в колонке
``conv_params`` манифеста. Оригиналы immutable.

Требует extras: ``pip install vision-lab[dicom]``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def convert_dicom_to_png16(dcm_path: str | Path, out_path: str | Path) -> dict:
    """Конвертирует один DICOM в 16-bit PNG; возвращает conv_params.

    Применяет rescale slope/intercept, оконное преобразование (window
    center/width из тегов; при отсутствии — min/max), инверсию MONOCHROME1.
    """
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImportError("Нужен pydicom: установите vision-lab[dicom]") from e

    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
        ww = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
        lo, hi = wc - ww / 2, wc + ww / 2
    else:
        lo, hi = float(arr.min()), float(arr.max())

    arr = np.clip((arr - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

    inverted = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"
    if inverted:
        arr = 1.0 - arr

    png16 = (arr * 65535.0).round().astype(np.uint16)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), png16):
        raise OSError(f"Не удалось записать {out_path}")

    return {
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "window_lo": lo,
        "window_hi": hi,
        "monochrome1_inverted": inverted,
        "bit_depth": 16,
    }
