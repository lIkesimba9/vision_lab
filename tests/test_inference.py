import cv2
import numpy as np
import torch
from torch import nn

from vision_lab.heads import LinearHead
from vision_lab.inference import Predictor, tta_logits
from vision_lab.inference.tta import TTA_VIEWS


class TinyBackbone(nn.Module):
    out_dim = 16

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def test_tta_views_are_invertible():
    x = torch.randn(2, 3, 8, 8)
    for name, fn in TTA_VIEWS.items():
        # применяя дважды hflip/vflip/rot180 -> исходное; identity тоже
        twice = fn(fn(x))
        assert torch.equal(twice, x), name


def test_tta_averages_logits():
    calls = []

    def fn(x):
        calls.append(x)
        return torch.ones(x.size(0), 4)

    out = tta_logits(fn, torch.randn(3, 3, 8, 8), views=("identity", "hflip", "vflip"))
    assert len(calls) == 3
    assert torch.allclose(out, torch.ones(3, 4))  # среднее одинаковых = то же


def test_predictor_tensor_and_disabled_tta():
    pred = Predictor(TinyBackbone(), LinearHead(4, 16, mode="ce"),
                     tta_views=("identity", "hflip"))
    logits = pred.predict_tensor(torch.rand(5, 3, 32, 32))
    assert logits.shape == (5, 4)

    pred_notta = Predictor(TinyBackbone(), LinearHead(4, 16), tta_views=())
    assert pred_notta.predict_tensor(torch.rand(2, 3, 32, 32)).shape == (2, 4)


def test_predictor_from_paths(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8))
        paths.append(p)
    pred = Predictor(TinyBackbone(), LinearHead(4, 16), image_size=16)
    proba = pred.predict_proba(paths, batch_size=2)
    assert proba.shape == (3, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
