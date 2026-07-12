import numpy as np
import pytest
import torch

import kornia.augmentation as K

from vision_lab.data.transforms import build_classification_transform
from vision_lab.ssl.gpu_augs import (
    MultiViewAugment,
    build_ssl_views,
    build_view_pipeline,
)


# --- CPU classification recipes -------------------------------------------
@pytest.mark.parametrize("recipe", ["light", "medium", "heavy_v1"])
def test_classification_recipe_produces_normalized_tensor(recipe):
    img = np.random.RandomState(0).uniform(0, 1, (64, 64, 3)).astype(np.float32)
    t = build_classification_transform(recipe, image_size=32, train=True)
    out = t(image=img)["image"]
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32


def test_eval_transform_deterministic():
    img = np.random.RandomState(1).uniform(0, 1, (40, 50, 3)).astype(np.float32)
    t = build_classification_transform("heavy_v1", image_size=24, train=False)  # train=False -> eval
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    assert torch.equal(a, b)


def test_unknown_recipe_raises():
    with pytest.raises(KeyError):
        build_classification_transform("nonexistent", train=True)


# --- GPU SSL views ---------------------------------------------------------
def test_view_pipeline_normalize_is_last():
    p = build_view_pipeline(16)
    ops = list(p)
    assert isinstance(ops[-1], K.Normalize)


def test_order_invariant_enforced_in_code():
    from vision_lab.ssl.gpu_augs import _assert_order

    bad = K.AugmentationSequential(
        K.Normalize(mean=torch.zeros(3), std=torch.ones(3)),
        K.RandomGaussianNoise(p=1.0),  # шум ПОСЛЕ Normalize — нарушение §6.3
    )
    with pytest.raises(ValueError, match="§6.3"):
        _assert_order(bad)


def test_byol_v1_makes_two_independent_views():
    views = build_ssl_views("byol_v1", image_size=16)
    x = torch.rand(4, 3, 24, 24)
    vs = views(x)
    assert len(vs.globals) == 2 and not vs.locals
    v1, v2 = vs.globals
    assert v1.shape == (4, 3, 16, 16)
    # независимые вьюхи: не совпадают
    assert not torch.allclose(v1, v2)


def test_dino_v1_multicrop_shapes():
    views = build_ssl_views("dino_v1", image_size=16, local_size=8, n_local=3)
    x = torch.rand(2, 3, 24, 24)
    vs = views(x)
    assert len(vs.globals) == 2 and len(vs.locals) == 3
    assert vs.globals[0].shape == (2, 3, 16, 16)
    assert vs.locals[0].shape == (2, 3, 8, 8)


def test_hydra_listconfig_size_coerced():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"size": [16, 16]})
    # ListConfig не должен ронять сборку (коэрция в _coerce_size)
    p = build_view_pipeline(cfg.size)
    out = p(torch.rand(2, 3, 20, 20))
    assert out.shape == (2, 3, 16, 16)


def test_multiview_requires_global():
    with pytest.raises(ValueError):
        MultiViewAugment([])
