"""Декодеры: уменьшенный декод JPEG (max_side)."""

def test_max_side_reduces_jpeg_keeping_aspect(tmp_path):
    """max_side декодирует JPEG в уменьшенном масштабе, не трогая пропорции."""
    import cv2
    import numpy as np

    from vision_lab.data.decoders import decode_image

    path = tmp_path / "big.jpg"
    h, w = 1600, 900
    cv2.imwrite(str(path), np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))

    full = decode_image(path)
    small = decode_image(path, max_side=400)

    assert full.shape[:2] == (h, w)
    # 1600/4 = 400 >= 400, значит делитель 4
    assert small.shape[:2] == (h // 4, w // 4)
    assert max(small.shape[:2]) >= 400
    assert abs(full.shape[0] / full.shape[1] - small.shape[0] / small.shape[1]) < 0.01


def test_max_side_ignored_for_png(tmp_path):
    """PNG уменьшенного декода не поддерживает — кадр должен остаться полным."""
    import cv2
    import numpy as np

    from vision_lab.data.decoders import decode_image

    path = tmp_path / "img.png"
    cv2.imwrite(str(path), np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8))
    assert decode_image(path, max_side=100).shape[:2] == (800, 600)


def test_max_side_never_below_requested(tmp_path):
    """Мелкий кадр не должен уменьшаться: делителя, оставляющего >= max_side, нет."""
    import cv2
    import numpy as np

    from vision_lab.data.decoders import decode_image

    path = tmp_path / "small.jpg"
    cv2.imwrite(str(path), np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8))
    assert decode_image(path, max_side=512).shape[:2] == (300, 200)
