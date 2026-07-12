"""LightningModule'ы: ровно два трейнера на всю библиотеку (ТЗ §2, §4).

* :class:`ClassificationTrainer` — любая голова (CE/BCE/AAM/.../multi-task);
* :class:`SSLTrainer` — любой :class:`~vision_lab.ssl.base.SSLMethod`.

Оба принимают уже ИНСТАНЦИРОВАННЫЕ компоненты (Hydra recursive `_target_`),
оптимизатор — как ``_partial_`` фабрику. Никакого ``instantiate`` внутри
``__init__`` (это связало бы библиотеку с Hydra и сломало прямое создание из REPL).
"""

from __future__ import annotations

from collections.abc import Callable

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from vision_lab.core.batch import target_view
from vision_lab.core.optim import param_groups
from vision_lab.core.schedules import build_warmup_cosine
from vision_lab.heads.base import ClassifierHead


class ClassificationTrainer(pl.LightningModule):
    """Единый трейнер классификации для ЛЮБОЙ головы.

    Контракт компонентов:
        ``backbone(image) -> (B, D)`` (EmbeddingBackbone);
        ``head(embeddings, targets) -> {"total_loss", ...}`` (ClassifierHead);
        ``head.predict_logits(embeddings) -> (B, C)``.

    ``backbone_lr``: None — как у головы (полный finetune); 0.0 — бэкбон
    заморожен (BN-статистики адаптируются); малое — low-LR доменная адаптация.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: ClassifierHead,
        optimizer: Callable[..., torch.optim.Optimizer],
        num_classes: int,
        warmup_steps: int = 0,
        backbone_lr: float | None = None,
        weight_decay: float = 0.0,
        monitor_metric: str = "val/f1_macro",
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.optimizer_factory = optimizer
        self.num_classes = num_classes
        self.warmup_steps = warmup_steps
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.monitor_metric = monitor_metric

        self._train_losses: dict[str, torchmetrics.MeanMetric] = {}
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    # -- шаги ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def _log_loss_dict(self, loss_dict: dict, stage: str) -> None:
        for name, value in loss_dict.items():
            key = f"{stage}/{name}"
            if key not in self._train_losses:
                self._train_losses[key] = torchmetrics.MeanMetric().to(value.device)
            self._train_losses[key].update(value.detach())

    def training_step(self, batch, batch_idx):
        emb = self.backbone(batch["image"])
        loss_dict = self.head(emb, target_view(batch))
        self._log_loss_dict(loss_dict, "train")
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss_dict["total_loss"]

    def on_train_epoch_end(self):
        for key, metric in self._train_losses.items():
            self.log(f"{key}_epoch", metric.compute(), sync_dist=True)
            metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        emb = self.backbone(batch["image"])
        logits = self.head.predict_logits(emb)
        labels = batch["label"].long()
        mask = labels >= 0
        if mask.any():
            preds = logits[mask].argmax(dim=1)
            self.val_accuracy.update(preds, labels[mask])
            self.val_f1_macro.update(preds, labels[mask])

    def on_validation_epoch_end(self):
        self.log("val/accuracy", self.val_accuracy.compute(), prog_bar=True, sync_dist=True)
        self.log("val/f1_macro", self.val_f1_macro.compute(), prog_bar=True, sync_dist=True)
        self.val_accuracy.reset()
        self.val_f1_macro.reset()

    # -- оптимизация -----------------------------------------------------------
    def configure_optimizers(self):
        groups = param_groups(
            {"backbone": self.backbone, "head": self.head},
            base_lr=self.optimizer_lr(),
            weight_decay=self.weight_decay,
            lr_overrides={"backbone": self.backbone_lr},
        )
        optimizer = self.optimizer_factory(groups)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = build_warmup_cosine(optimizer, self.warmup_steps, total_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def optimizer_lr(self) -> float:
        """LR из partial-фабрики оптимизатора (для param_groups base_lr)."""
        kw = getattr(self.optimizer_factory, "keywords", {})
        if "lr" in kw:
            return float(kw["lr"])
        raise ValueError("optimizer partial должен задавать lr= (нужен для base_lr param-групп)")
