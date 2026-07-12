"""Смоук: 1 батч через ClassificationTrainer на синтетике (CPU, §11.3)."""

from functools import partial

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from vision_lab.core.module import ClassificationTrainer
from vision_lab.heads import AAMHead, LinearHead, MultiTaskHead


class SyntheticImages(Dataset):
    def __init__(self, n=16, num_classes=4, size=32, aux=False):
        self.n, self.c, self.size, self.aux = n, num_classes, size, aux

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        item = {
            "image": torch.rand(3, self.size, self.size, generator=g),
            "label": int(torch.randint(0, self.c, (1,), generator=g)),
            "sample_id": f"s{i}",
            "source": "synthetic",
        }
        if self.aux:
            item["label_aux"] = int(torch.randint(0, 3, (1,), generator=g))
        return item


class TinyBackbone(torch.nn.Module):
    """Заглушка бэкбона (без timm-скачивания): conv -> global pool -> (B, D)."""

    out_dim = 16

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


def run_one_epoch(head, aux=False):
    ds = SyntheticImages(aux=aux)
    trainer_module = ClassificationTrainer(
        backbone=TinyBackbone(),
        head=head,
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        num_classes=4,
        warmup_steps=2,
    )
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                         enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(trainer_module, DataLoader(ds, batch_size=8),
                DataLoader(SyntheticImages(n=8), batch_size=8))
    return trainer_module


def test_ce_pipeline_smoke():
    run_one_epoch(LinearHead(4, 16, mode="ce"))


def test_aam_pipeline_smoke():
    run_one_epoch(AAMHead(4, 16))


def test_multitask_pipeline_smoke():
    head = MultiTaskHead(
        heads={"main": LinearHead(4, 16, target_key="label"),
               "aux": LinearHead(3, 16, target_key="label_aux")},
        primary="main",
    )
    run_one_epoch(head, aux=True)


def test_frozen_backbone_lr_zero():
    module = ClassificationTrainer(
        backbone=TinyBackbone(),
        head=LinearHead(4, 16, mode="ce"),
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        num_classes=4,
        backbone_lr=0.0,
    )
    module.trainer = type("T", (), {"estimated_stepping_batches": 10})()
    cfg = module.configure_optimizers()
    groups = {g["name"]: g for g in cfg["optimizer"].param_groups}
    assert groups["backbone.decay"]["lr"] == 0.0
    assert groups["head.decay"]["lr"] == 1e-3
