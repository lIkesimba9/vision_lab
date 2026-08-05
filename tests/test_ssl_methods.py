"""Тесты новых SSL-методов: SimCLR, MoCo v3, SimSiam, MAE, SimMIM.

BYOL/DINOv2 покрыты в test_ssl.py; здесь — контрактные проверки (forward/
backward/extract), сверка NT-Xent с независимым референсом и MIM-примитивы.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.models.backbones import TokenBackbone, TokenOutput
from vision_lab.ssl.components import block_mask, patchify
from vision_lab.ssl.gpu_augs import MultiViewAugment, build_ssl_views, build_view_pipeline
from vision_lab.ssl.mae import MAE, random_masking
from vision_lab.ssl.moco import MoCoV3
from vision_lab.ssl.simclr import SimCLR, nt_xent_loss
from vision_lab.ssl.simmim import SimMIM
from vision_lab.ssl.simsiam import SimSiam


class TinyEmbBackbone(nn.Module):
    out_dim = 16

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class TinyTokenBackbone(nn.Module):
    """ViT-подобный токен-бэкбон без timm: patchify 8x8 -> токены."""

    out_dim = 12

    def __init__(self, patch=8):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, 12, patch, stride=patch)

    def grid_size(self, hw):
        return hw[0] // self.patch, hw[1] // self.patch

    def forward(self, x):
        f = self.proj(x)
        b, c, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2)
        return TokenOutput(pooled=tokens.mean(1), tokens=tokens, grid=(h, w))

    def embed(self, x):
        return self.forward(x).pooled


def two_views(size=16):
    v = build_view_pipeline(size, scale=(0.5, 1.0))
    return MultiViewAugment([v, build_view_pipeline(size, scale=(0.5, 1.0))])


def batch(n=6):
    return {"image": torch.rand(n, 3, 24, 24)}


# --- NT-Xent / SimCLR ---------------------------------------------------------
def ref_nt_xent(z1, z2, t):
    """Независимый референс NT-Xent (Chen 2020, eq. 1) — прямые циклы."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z.shape[0]
    losses = []
    for i in range(n):
        j = (i + n // 2) % n
        num = torch.exp(z[i] @ z[j] / t)
        den = sum(torch.exp(z[i] @ z[k] / t) for k in range(n) if k != i)
        losses.append(-torch.log(num / den))
    return torch.stack(losses).mean()


def test_nt_xent_matches_reference():
    torch.manual_seed(0)
    z1, z2 = torch.randn(4, 8), torch.randn(4, 8)
    got = nt_xent_loss(z1, z2, temperature=0.3)
    assert got.item() == pytest.approx(ref_nt_xent(z1, z2, 0.3).item(), abs=1e-5)


def test_simclr_forward_backward_and_extract():
    model = SimCLR(TinyEmbBackbone(), two_views(), hidden_dim=32, projection_dim=16)
    out = model(batch())
    assert out["total_loss"].ndim == 0 and torch.isfinite(out["total_loss"])
    out["total_loss"].backward()
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert model.extract_embeddings(torch.rand(3, 3, 24, 24)).shape == (3, 16)
    model.momentum_update()  # no-op: учителя нет


# --- MoCo v3 -------------------------------------------------------------------
def make_moco():
    return MoCoV3(TinyEmbBackbone(), two_views(), hidden_dim=32, projection_dim=16)


def test_moco_forward_backward_keys_no_grad():
    moco = make_moco()
    out = moco(batch())
    assert torch.isfinite(out["total_loss"])
    out["total_loss"].backward()
    assert all(p.grad is None for p in moco.momentum_encoder.parameters())
    assert any(p.grad is not None for p in moco.predictor.parameters())
    assert moco.extract_embeddings(torch.rand(2, 3, 24, 24)).shape == (2, 16)


def test_moco_momentum_update_moves_encoder():
    moco = make_moco()
    moco.current_tau = 0.9
    before = [p.clone() for p in moco.momentum_encoder.parameters()]
    with torch.no_grad():
        for p in moco.online_encoder.parameters():
            p.add_(0.1)
    moco.momentum_update()
    assert any(not torch.allclose(b, a) for b, a in
               zip(before, moco.momentum_encoder.parameters(), strict=True))


def test_moco_aligned_keys_beat_shuffled():
    """InfoNCE: выровненные q=k дают меньший лосс, чем перепутанные позитивы."""
    moco = make_moco()
    q = torch.eye(4, 16)  # ортогональные строки
    aligned = moco._contrastive(q, q)
    shuffled = moco._contrastive(q, q.flip(0))
    assert aligned.item() < shuffled.item()


# --- SimSiam --------------------------------------------------------------------
def test_simsiam_forward_backward_stopgrad():
    model = SimSiam(TinyEmbBackbone(), two_views(), hidden_dim=32, projection_dim=16)
    out = model(batch())
    assert torch.isfinite(out["total_loss"])
    out["total_loss"].backward()
    # градиент течёт и в predictor, и в projector (через p-ветку)
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert any(p.grad is not None for p in model.projector.parameters())
    model.momentum_update()  # no-op


# --- MIM-примитивы ---------------------------------------------------------------
def test_random_masking_guarantees_and_partition():
    tok = torch.arange(10).float().reshape(1, 10, 1).expand(3, 10, 1)
    for ratio in (0.01, 0.5, 0.99):
        visible, mask, ids_restore = random_masking(tok, ratio)
        assert 1 <= visible.shape[1] <= 9  # «не всё и не ничего»
        assert (mask.sum(dim=1) == 10 - visible.shape[1]).all()
        # видимые токены = ровно немаскированные позиции
        vis_ids = set(visible[0, :, 0].long().tolist())
        unmasked = set((~mask[0]).nonzero(as_tuple=True)[0].tolist())
        assert vis_ids == unmasked
        assert ids_restore.shape == (3, 10)


def test_patchify_content():
    x = torch.arange(2 * 3 * 4 * 4).float().reshape(2, 3, 4, 4)
    p = patchify(x, grid=(2, 2))
    assert p.shape == (2, 4, 2 * 2 * 3)
    # патч [0]: верхний-левый блок 2x2, порядок (ph, pw, C)
    expected = torch.stack([x[0, :, i, j] for i in (0, 1) for j in (0, 1)]).reshape(-1)
    assert torch.equal(p[0, 0], expected)


def test_block_mask_zeroes_input_and_never_all():
    x = torch.rand(5, 3, 16, 16) + 0.1  # строго > 0
    x_masked, mask = block_mask(x, grid=(2, 2), mask_ratio=0.9)
    assert not mask.all(dim=1).any()  # хотя бы один блок виден
    m0 = mask[0].reshape(2, 2)
    for i in range(2):
        for j in range(2):
            block = x_masked[0, :, i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            assert (block == 0).all() if m0[i, j] else (block > 0).all()


# --- MAE --------------------------------------------------------------------------
def make_mae(image_size=32, **kwargs):
    backbone = TokenBackbone("test_vit", pretrained=False)
    views = build_ssl_views("mim_v1", image_size=image_size)
    return MAE(backbone, views, image_size=32, mask_ratio=0.5,
               decoder_dim=32, decoder_depth=1, decoder_heads=2, **kwargs)


def test_mae_forward_backward_and_extract():
    mae = make_mae()
    out = mae({"image": torch.rand(2, 3, 32, 32)})
    assert torch.isfinite(out["total_loss"]) and "mae_loss" in out
    out["total_loss"].backward()
    assert mae.decoder_pred.weight.grad is not None
    assert any(p.grad is not None for p in mae.backbone.net.patch_embed.parameters())
    emb = mae.extract_embeddings(torch.rand(2, 3, 32, 32))
    assert emb.shape == (2, mae.backbone.out_dim)
    mae.momentum_update()  # no-op: учителя нет


def test_mae_rejects_view_grid_mismatch():
    mae = make_mae(image_size=16)  # вьюхи 16px -> сетка (1,1) != (2,2)
    with pytest.raises(ValueError, match="Сетка"):
        mae({"image": torch.rand(2, 3, 32, 32)})


def test_mae_requires_vit_backbone():
    backbone = TokenBackbone("test_convnext", pretrained=False)  # нет patch_embed/ViT-блоков
    with pytest.raises(ValueError, match="SimMIM"):
        MAE(backbone, build_ssl_views("mim_v1", image_size=32), image_size=32)


# --- SimMIM -----------------------------------------------------------------------
def make_simmim(patch_px=8, **kwargs):
    views = build_ssl_views("mim_v1", image_size=16)
    return SimMIM(TinyTokenBackbone(), views, patch_px=patch_px, **kwargs)


def test_simmim_forward_backward_and_extract():
    model = make_simmim(mask_ratio=0.5)
    out = model({"image": torch.rand(4, 3, 24, 24)})
    assert torch.isfinite(out["total_loss"]) and "simmim_loss" in out
    out["total_loss"].backward()
    assert model.head.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert model.extract_embeddings(torch.rand(2, 3, 16, 16)).shape == (2, 12)


def test_simmim_patch_px_mismatch_raises():
    model = make_simmim(patch_px=4)  # фактический патч 16//2=8
    with pytest.raises(ValueError, match="patch_px"):
        model({"image": torch.rand(2, 3, 24, 24)})
