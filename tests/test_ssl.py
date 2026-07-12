import copy
from functools import partial

import lightning.pytorch as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vision_lab.core.batch import MISSING_LABEL
from vision_lab.core.callbacks import KNNProbeCallback
from vision_lab.core.module import SSLTrainer
from vision_lab.core.schedules import CosineSchedule, ScheduleDriver
from vision_lab.ssl.base import MomentumTeacher
from vision_lab.ssl.byol import BYOL, shuffle_within_groups
from vision_lab.ssl.components import DINOHead, koleo_loss, sinkhorn_knopp
from vision_lab.ssl.dinov2 import DINOv2
from vision_lab.ssl.gpu_augs import MultiViewAugment, build_view_pipeline


# --- MomentumTeacher -------------------------------------------------------
def test_momentum_teacher_ema_and_no_grad():
    student = nn.Linear(4, 4)
    teacher = MomentumTeacher(student)
    for p in teacher.module.parameters():
        assert not p.requires_grad
    # веса учителя в state_dict (resume бесплатно)
    assert any("module" in k for k in teacher.state_dict())

    before = teacher.module.weight.clone()
    with torch.no_grad():
        student.weight.add_(1.0)
    teacher.update(student, tau=0.9)
    expected = 0.9 * before + 0.1 * student.weight
    assert torch.allclose(teacher.module.weight, expected)


def test_momentum_teacher_factory_for_weightnorm_head():
    def make():
        return DINOHead(8, out_dim=16, hidden_dim=16, bottleneck_dim=8)

    student = make()
    teacher = MomentumTeacher(student, factory=make)  # deepcopy weight_norm упал бы
    x = torch.randn(2, 8)
    assert torch.allclose(teacher(x), student(x), atol=1e-5)


# --- SSL components --------------------------------------------------------
def test_koleo_and_sinkhorn_shapes():
    assert koleo_loss(torch.randn(8, 16)).ndim == 0
    q = sinkhorn_knopp(torch.randn(6, 10))
    assert q.shape == (6, 10)
    assert torch.isfinite(q).all()


def test_shuffle_within_groups_keeps_class_and_skips_missing():
    views = torch.arange(6 * 3).float().reshape(6, 3)
    groups = torch.tensor([0, 0, 1, 1, MISSING_LABEL, MISSING_LABEL])
    torch.manual_seed(0)
    out = shuffle_within_groups(views, groups)
    # -1 группа не тронута
    assert torch.equal(out[4:], views[4:])
    # строки группы 0 — перестановка исходных строк группы 0
    assert {tuple(r.tolist()) for r in out[:2]} == {tuple(r.tolist()) for r in views[:2]}


# --- BYOL ------------------------------------------------------------------
class TinyEmbBackbone(nn.Module):
    out_dim = 16

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def make_byol(**kwargs):
    views = build_view_pipeline(16, scale=(0.5, 1.0))
    mv = MultiViewAugment([views, copy.deepcopy(views)])
    return BYOL(TinyEmbBackbone(), mv, hidden_dim=32, projection_dim=16, **kwargs)


def byol_batch(n=8, labels=True):
    b = {"image": torch.rand(n, 3, 24, 24)}
    if labels:
        b["label"] = torch.randint(0, 3, (n,))
    return b


def test_byol_forward_and_backward():
    byol = make_byol()
    out = byol(byol_batch())
    assert "total_loss" in out and out["total_loss"].ndim == 0
    out["total_loss"].backward()
    # у target-энкодера нет градиента
    assert all(p.grad is None for p in byol.target_encoder.parameters())


def test_byol_positive_shuffle_runs():
    byol = make_byol(positive_shuffle=True, shuffle_source="label")
    out = byol(byol_batch())
    assert torch.isfinite(out["total_loss"])


def test_byol_momentum_update_moves_teacher():
    byol = make_byol()
    byol.current_tau = 0.9
    before = [p.clone() for p in byol.target_encoder.parameters()]
    # шаг обучения меняет online-веса
    out = byol(byol_batch())
    out["total_loss"].backward()
    with torch.no_grad():
        for p in byol.online_encoder.parameters():
            p.add_(0.1)
    byol.momentum_update()
    moved = any(not torch.allclose(b, a)
                for b, a in zip(before, byol.target_encoder.parameters(), strict=True))
    assert moved


def test_byol_extract_embeddings_uses_online_backbone():
    byol = make_byol()
    emb = byol.extract_embeddings(torch.rand(4, 3, 24, 24))
    assert emb.shape == (4, 16)


# --- DINOv2 ----------------------------------------------------------------
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
        from vision_lab.models.backbones import TokenOutput

        f = self.proj(x)  # (B, C, h, w)
        b, c, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2)
        return TokenOutput(pooled=tokens.mean(1), tokens=tokens, grid=(h, w))

    def embed(self, x):
        return self.forward(x).pooled


def make_dino(ibot_weight=1.0, center_mode="sinkhorn"):
    g = build_view_pipeline(16, scale=(0.4, 1.0))
    local = build_view_pipeline(8, scale=(0.1, 0.4))
    mv = MultiViewAugment([g, copy.deepcopy(g)], local_view=local, n_local=2)
    return DINOv2(TinyTokenBackbone(), mv, out_dim=32, head_hidden_dim=32,
                  head_bottleneck_dim=16, ibot_weight=ibot_weight, center_mode=center_mode)


@pytest.mark.parametrize("ibot_weight", [1.0, 0.0])
def test_dino_forward_backward(ibot_weight):
    dino = make_dino(ibot_weight=ibot_weight)
    out = dino({"image": torch.rand(4, 3, 24, 24)})
    assert "dino_loss" in out
    assert ("ibot_loss" in out) == (ibot_weight > 0)
    out["total_loss"].backward()
    assert all(p.grad is None for p in dino.teacher.parameters())


def test_dino_extract_embeddings_uses_teacher():
    dino = make_dino()
    emb = dino.extract_embeddings(torch.rand(2, 3, 24, 24))
    assert emb.shape == (2, 12)


def test_dino_block_mask_never_all():
    dino = make_dino()
    x = torch.rand(5, 3, 16, 16)
    _, mask = dino._block_mask(x, grid=(2, 2))
    assert not mask.all(dim=1).any()  # хотя бы один токен виден в каждом сэмпле


# --- EMA correctness under gradient accumulation ---------------------------
class OneItemDS(Dataset):
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"image": torch.rand(3, 24, 24, generator=g),
                "label": int(torch.randint(0, 3, (1,), generator=g)),
                "sample_id": f"s{i}", "source": "syn"}


def test_ema_updates_once_per_optimizer_step_under_accumulation(monkeypatch):
    """EMA-обновление ровно раз на optimizer step, не на каждый micro-batch."""
    byol = make_byol()
    calls = {"n": 0}
    orig = byol.momentum_update

    def counting():
        calls["n"] += 1
        return orig()

    byol.momentum_update = counting
    module = SSLTrainer(byol, optimizer=partial(torch.optim.AdamW, lr=1e-3))
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                         accumulate_grad_batches=4, limit_val_batches=0,
                         enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, DataLoader(OneItemDS(32), batch_size=4))
    # 8 micro-batches / accumulate 4 = 2 optimizer steps => 2 EMA-обновления
    assert calls["n"] == 2


def test_ssl_trainer_with_probe_and_schedule_smoke():
    byol = make_byol()
    module = SSLTrainer(byol, optimizer=partial(torch.optim.AdamW, lr=1e-3), warmup_steps=1)
    driver = ScheduleDriver({"method.current_tau": CosineSchedule(0.99, 1.0)}, log_values=False)
    probe = KNNProbeCallback(k=3, run_linear_probe=True)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, callbacks=[driver, probe],
                         limit_train_batches=2, limit_val_batches=2,
                         enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, DataLoader(OneItemDS(16), batch_size=8),
                DataLoader(OneItemDS(16), batch_size=8))
    assert "val/knn_f1" in trainer.callback_metrics
