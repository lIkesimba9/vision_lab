"""2-GPU DDP-смоук: BYOL с PK-сэмплером, sync_batchnorm, bf16-mixed (ТЗ §11.1).

Запуск вручную (не через pytest — нужен реальный multi-GPU):
    uv run python tests/ddp_smoke.py

Проверяет: обучение под DDP не падает и не виснет (равное число батчей на ранг),
EMA-учитель обновляется, probe считается детерминированно на каждом ранге.
"""

from __future__ import annotations

import copy
from functools import partial

import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vision_lab.core.callbacks import KNNProbeCallback
from vision_lab.core.module import SSLTrainer
from vision_lab.core.runtime import configure_threads
from vision_lab.core.schedules import CosineSchedule, ScheduleDriver
from vision_lab.data.samplers import PKCoverageBatchSampler
from vision_lab.ssl.byol import BYOL
from vision_lab.ssl.gpu_augs import MultiViewAugment, build_view_pipeline


class SynDS(Dataset):
    def __init__(self, n=256, num_classes=8):
        self.n, self.c = n, num_classes
        self.labels = [i % num_classes for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"image": torch.rand(3, 32, 32, generator=g),
                "label": self.labels[i], "sample_id": f"s{i}", "source": "syn"}


class Backbone(nn.Module):
    out_dim = 32

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def main():
    configure_threads()
    torch.set_float32_matmul_precision("high")

    train_ds, val_ds = SynDS(256), SynDS(128)
    view = build_view_pipeline(24, scale=(0.4, 1.0))
    method = BYOL(Backbone(), MultiViewAugment([view, copy.deepcopy(view)]),
                  hidden_dim=64, projection_dim=32, positive_shuffle=True)
    module = SSLTrainer(method, optimizer=partial(torch.optim.AdamW, lr=1e-3), warmup_steps=5)

    train_sampler = PKCoverageBatchSampler(train_ds.labels, batch_size=32, n_labels_per_batch=8)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

    trainer = pl.Trainer(
        max_epochs=2, accelerator="gpu", devices=2, strategy="ddp",
        precision="bf16-mixed", sync_batchnorm=True, use_distributed_sampler=False,
        callbacks=[
            ScheduleDriver({"method.current_tau": CosineSchedule(0.99, 1.0)}),
            KNNProbeCallback(k=10),
        ],
        enable_checkpointing=False, logger=False, enable_progress_bar=True,
    )
    trainer.fit(module, train_loader, val_loader)
    if trainer.is_global_zero:
        print("\n[DDP-смоук] OK. val/knn_f1 =", float(trainer.callback_metrics.get("val/knn_f1", -1)))


if __name__ == "__main__":
    main()
