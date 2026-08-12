"""Декодеры: уменьшенный декод JPEG (max_side)."""

import cv2
import numpy as np
import pytest
from PIL import Image

from vision_lab.data.decoders import decode_image


def _write_random(path, h, w):
    cv2.imwrite(str(path), np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def test_max_side_reduces_jpeg_keeping_aspect(tmp_path):
    """max_side декодирует JPEG в уменьшенном масштабе, не трогая пропорции."""
    path = tmp_path / "big.jpg"
    h, w = 1600, 900
    _write_random(path, h, w)

    full = decode_image(path)
    small = decode_image(path, max_side=400)

    assert full.shape[:2] == (h, w)
    # 1600/4 = 400 >= 400, значит делитель 4
    assert small.shape[:2] == (h // 4, w // 4)
    assert max(small.shape[:2]) >= 400
    assert abs(full.shape[0] / full.shape[1] - small.shape[0] / small.shape[1]) < 0.01


def test_max_side_ignored_for_png(tmp_path):
    """PNG уменьшенного декода не поддерживает — кадр должен остаться полным."""
    path = tmp_path / "img.png"
    _write_random(path, 800, 600)
    assert decode_image(path, max_side=100).shape[:2] == (800, 600)


def test_max_side_never_below_requested(tmp_path):
    """Мелкий кадр не должен уменьшаться: делителя, оставляющего >= max_side, нет."""
    path = tmp_path / "small.jpg"
    _write_random(path, 300, 200)
    assert decode_image(path, max_side=512).shape[:2] == (300, 200)


def test_reduced_decode_ignores_exif_orientation(tmp_path):
    """Редуцированный декод не применяет EXIF-ориентацию — как и полный путь
    (IMREAD_UNCHANGED), иначе ориентация кадра зависела бы от сработавшей редукции."""
    path = tmp_path / "exif.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Orientation: поворот на 90°
    Image.fromarray(np.random.randint(0, 255, (1600, 900, 3), dtype=np.uint8)).save(
        path, format="JPEG", exif=exif.tobytes()
    )

    full = decode_image(path)
    small = decode_image(path, max_side=400)

    assert full.shape[:2] == (1600, 900)
    assert small.shape[:2] == (400, 225)


def test_max_side_skips_mislabeled_non_jpeg(tmp_path):
    """16-битный PNG с меткой jpg не должен терять битность и разрешение:
    формат проверяется по заголовку файла, а не по колонке манифеста."""
    png = tmp_path / "scan.png"
    cv2.imwrite(str(png), np.random.randint(0, 65535, (800, 600), dtype=np.uint16))
    fake = tmp_path / "scan.jpg"
    png.rename(fake)

    out = decode_image(fake, fmt="jpg", max_side=100)

    assert out.shape[:2] == (800, 600)  # редукция не применилась
    assert out.max() > 255.0 / 65535.0  # нормировка 16-битная, /65535


def test_max_side_must_be_positive(tmp_path):
    """max_side <= 0 — ошибка конфигурации, а не молчаливый декод в 1/8."""
    path = tmp_path / "img.jpg"
    _write_random(path, 300, 200)
    with pytest.raises(AssertionError):
        decode_image(path, max_side=0)
