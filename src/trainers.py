import math

import numpy as np
import torch
import torchmetrics
import pytorch_lightning as pl

from src.eval.knn_probe import knn_macro_f1, linear_probe_macro_f1, stratified_gallery_split

from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

import hydra
from hydra.utils import instantiate


def build_warmup_cosine(optimizer, warmup_steps, total_steps):
    """Linear warmup (0.1 → 1.0) за warmup_steps, затем cosine annealing до total_steps."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class ClassificationTrainer(pl.LightningModule):
    """Единый трейнер классификации для ЛЮБОЙ головы (CE/BCE/AAM/SubCenter/...).

    Контракт компонентов:
        model(images) -> {"embeddings": (B, D)}            (см. src.models.models.ClassificationModel)
        criterion(embeddings, labels) -> {"total_loss": ...}   (см. src.losses.heads.ClassifierHead)
        criterion.predict_logits(embeddings) -> (B, C)     (логиты для метрик/инференса, без маржина)

    backbone_lr: LR для backbone модели.
        None    → как у головы (полный finetune);
        0       → веса backbone заморожены (BN-статистики адаптируются в train-режиме);
        малое   → low-LR доменная адаптация (decoupling, этап 3).
    Голова (criterion) всегда учится с optimizer.lr.

    lr_total_steps_mult: множитель длины cosine-расписания относительно реальных шагов
        (исторический 4 у эмбеддинг-трейнера — LR не доходит до ~0; по умолчанию 1).
    """

    def __init__(
        self,
        model,
        loss,
        optimizer_cfg,
        num_classes: int,
        warmup_steps: int,
        monitor_metric: str,
        val_loader_names=None,
        valid_criterion=None,
        backbone_lr: float = None,
        lr_total_steps_mult: int = 1,
    ):
        super().__init__()

        self.model = hydra.utils.instantiate(model, _recursive_=False)
        self.criterion = hydra.utils.instantiate(loss)

        self.optimizer_config = optimizer_cfg
        self.warmup_steps = warmup_steps
        self.backbone_lr = backbone_lr
        self.lr_total_steps_mult = lr_total_steps_mult
        self.monitor_metric = monitor_metric

        self.train_losses = {}
        self.valid_losses = {}

        self.valid_loss = torchmetrics.MeanMetric()
        self.valid_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.valid_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, images):
        return self.model(images)

    def _embeddings(self, images):
        out = self.model(images)
        return out["embeddings"] if isinstance(out, dict) else out

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        embeddings = self._embeddings(images)
        # multi-task: если в батче есть вторая метка (label11) — передаём её в criterion
        loss_dict = (self.criterion(embeddings, labels.long(), batch["label11"].long())
                     if "label11" in batch else self.criterion(embeddings, labels.long()))

        for name, value in loss_dict.items():
            if name not in self.train_losses:
                self.train_losses[name] = torchmetrics.MeanMetric().to(value.device)
            self.train_losses[name].update(value.detach())

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=False, on_epoch=True, sync_dist=True)
        return loss_dict["total_loss"]

    def on_train_epoch_end(self):
        for name, metric in self.train_losses.items():
            self.log(f"train_epoch/{name}", metric.compute(), prog_bar=False, sync_dist=True)
            metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images, labels = batch["image"], batch["label"]
        embeddings = self._embeddings(images)
        loss_dict = (self.criterion(embeddings, labels.long(), batch["label11"].long())
                     if "label11" in batch else self.criterion(embeddings, labels.long()))
        logits = self.criterion.predict_logits(embeddings)

        for name, value in loss_dict.items():
            if name not in self.valid_losses:
                self.valid_losses[name] = torchmetrics.MeanMetric().to(value.device)
            self.valid_losses[name].update(value.detach())
        self.valid_loss.update(loss_dict["total_loss"])

        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        self.valid_accuracy.update(preds, labels)
        self.valid_f1_macro.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log("val/loss", self.valid_loss.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/accuracy", self.valid_accuracy.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/f1_macro", self.valid_f1_macro.compute(), prog_bar=True, on_epoch=True, sync_dist=True)

        self.valid_loss.reset()
        self.valid_accuracy.reset()
        self.valid_f1_macro.reset()

        for name, metric in self.valid_losses.items():
            self.log(f"valid/{name}", metric.compute(), prog_bar=False, sync_dist=True)
            metric.reset()

    def configure_optimizers(self):
        # backbone (self.model) и голова (self.criterion) — обе обучаемы.
        base_lr = self.optimizer_config.lr
        if self.backbone_lr is None:
            groups = list(self.model.parameters()) + list(self.criterion.parameters())
        else:
            groups = [
                {"params": list(self.model.parameters()), "lr": self.backbone_lr},
                {"params": list(self.criterion.parameters()), "lr": base_lr},
            ]
            groups = [g for g in groups if g["params"]]
        # _partial_: param-группы (list[dict]) уходят в оптимизатор как чистый Python
        optimizer = instantiate(self.optimizer_config, _partial_=True)(groups)
        total_steps = self.lr_total_steps_mult * self.trainer.estimated_stepping_batches
        scheduler = build_warmup_cosine(optimizer, self.warmup_steps, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class BYOLTrainer(pl.LightningModule):

    def __init__(
        self,
        model,
        loss,
        optimizer_cfg,
        valid_criterion,
        val_loader_names,
        num_classes: int,
        warmup_steps: int,
        monitor_metric: str,
        lr_total_steps_mult: float = 4.0,
        gallery_loader_name: str = "",
        knn_k: int = 20,
        gallery_frac: float = 0.5,
        knn_seed: int = 42,
        run_probe: bool = True,
    ):
        super().__init__()
        self.model = hydra.utils.instantiate(model, _recursive_=False) #, _recursive_=False
        self.criterion = hydra.utils.instantiate(loss)

        self.optimizer_config = optimizer_cfg
        self.warmup_steps = warmup_steps
        self.lr_total_steps_mult = lr_total_steps_mult

        self.train_losses = {}
        self.valid_losses_by_loader = {}

        self.valid_criterion = hydra.utils.instantiate(valid_criterion)

        self.val_loader_names = val_loader_names
        self.monitor_metric = monitor_metric

        # отбор чекпоинтов по kNN / linear-probe на milk_train (галерея) + остальных квери
        self.gallery_loader_name = gallery_loader_name
        self.knn_k = knn_k
        self.gallery_frac = gallery_frac
        self.knn_seed = knn_seed
        self.run_probe = run_probe
        self._val_emb = {}
        self._val_lab = {}

        


    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx: int):
        # batch - dict
        output = self.model.forward(batch)
        losses = self.criterion(output, batch)
        for name, value in losses.items():
            if name not in self.train_losses:
                self.train_losses[name] = torchmetrics.MeanMetric().to(value.device)

            self.train_losses[name].update(value.detach())

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train/lr",
            lr,
            prog_bar=False,
            on_epoch=True,
        )

        return losses['total_loss']

    def on_train_epoch_end(self) -> None:
        for name, metric in self.train_losses.items():
            loss_value = metric.compute()

            self.log(
                f"train_epoch/{name}",
                loss_value,
                prog_bar=False,  # обычно только общий лосс в бар
                sync_dist=True,
            )

            metric.reset()

    def on_validation_epoch_start(self):
        self._val_emb = {}
        self._val_lab = {}

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        embeddings = self.model.extract_embeddings(batch['image'])
        labels = batch["label"]
        loader_name = self.val_loader_names[dataloader_idx]

        # вторичный лог: triplet-loss на эмбеддингах (как раньше, грубый прокси)
        loss = self.valid_criterion(embeddings, labels)
        if loader_name not in self.valid_losses_by_loader:
            self.valid_losses_by_loader[loader_name] = torchmetrics.MeanMetric().to(loss.device)
        self.valid_losses_by_loader[loader_name].update(loss.detach())

        # копим эмбеддинги/метки для kNN и linear-probe (на CPU)
        self._val_emb.setdefault(loader_name, []).append(embeddings.float().cpu())
        self._val_lab.setdefault(loader_name, []).append(labels.detach().cpu())

    def _compute_knn_probe(self):
        """kNN/linear-probe macro-F1: галерея = стратиф. половина milk_train,
        квери = вторая половина milk_train + остальные labeled-лоадеры (melanoscope)."""
        g_name = self.gallery_loader_name
        if g_name not in self._val_emb:
            return {}
        gx = torch.cat(self._val_emb[g_name]).numpy()
        gy = torch.cat(self._val_lab[g_name]).numpy()
        g_idx, q_idx = stratified_gallery_split(gy, self.gallery_frac, self.knn_seed)
        gallery_x, gallery_y = gx[g_idx], gy[g_idx]

        def clean(n):
            return n[len("valid_"):] if n.startswith("valid_") else n

        # квери milk = held-out половина milk_train; остальные labeled-лоадеры — как есть
        query_sets = {"milk": (gx[q_idx], gy[q_idx])}
        for name in self._val_emb:
            if name == g_name:
                continue
            qx = torch.cat(self._val_emb[name]).numpy()
            qy = torch.cat(self._val_lab[name]).numpy()
            query_sets[clean(name)] = (qx, qy)

        knn_scores, probe_scores, sel_scores, logs = [], [], [], {}
        for qname, (qx, qy) in query_sets.items():
            if len(qy) == 0:
                continue
            knn = knn_macro_f1(gallery_x, gallery_y, qx, qy, k=self.knn_k)
            logs[f"val/knn_f1/{qname}"] = knn
            knn_scores.append(knn)
            if self.run_probe:
                probe = linear_probe_macro_f1(gallery_x, gallery_y, qx, qy)
                logs[f"val/linprobe_f1/{qname}"] = probe
                probe_scores.append(probe)
                # combined selection: knn и linprobe часто пикуют на разных эпохах -> отбираем
                # чекпоинт, хороший по ОБОИМ (среднее)
                sel = 0.5 * (knn + probe)
                logs[f"val/sel_f1/{qname}"] = sel
                sel_scores.append(sel)
        if knn_scores:
            logs["val/knn_f1"] = float(np.mean(knn_scores))
        if probe_scores:
            logs["val/linprobe_f1"] = float(np.mean(probe_scores))
        if sel_scores:
            logs["val/sel_f1"] = float(np.mean(sel_scores))
        return logs

    def on_validation_epoch_end(self):
        losses = []
        for name, metric in self.valid_losses_by_loader.items():
            loss_value = metric.compute()
            losses.append(loss_value.item())
            self.log(f"valid/loss/{name}", loss_value, prog_bar=False, sync_dist=True)
            metric.reset()
        if losses:
            self.log("valid/loss", sum(losses) / len(losses), prog_bar=False, sync_dist=True)

        # каждый ранг видит полный val-набор (use_distributed_sampler=False) → метрики
        # детерминированы и одинаковы на всех рангах, sync_dist не нужен.
        for k, v in self._compute_knn_probe().items():
            # значения детерминированы и идентичны на рангах; sync_dist=True (mean) гарантирует
            # консистентный монитор чекпоинта в DDP без накладных расходов (пара скаляров/эпоху)
            in_bar = k.endswith("/milk") or k in ("val/knn_f1", "val/linprobe_f1", "val/sel_f1")
            self.log(k, v, prog_bar=in_bar, sync_dist=True)
        self._val_emb = {}
        self._val_lab = {}

    def configure_optimizers(self):
        # модель + параметры лосса: у BYOL/triplet их нет, а у иерархического AAM —
        # обучаемые центроиды (AAM-головы), которые ОБЯЗАНЫ попасть в оптимизатор.
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        optimizer = instantiate(self.optimizer_config, params=params)
        # linear warmup -> cosine annealing. lr_total_steps_mult=1.0 -> честный отжиг до ~0 за
        # реальный горизонт (канон SSL: BYOL/DINO/SimCLR). Дефолт 4.0 — легаси (LR не доходит до 0).
        total_steps = self.lr_total_steps_mult * self.trainer.estimated_stepping_batches
        scheduler = build_warmup_cosine(optimizer, self.warmup_steps, int(total_steps))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class DINOv2Trainer(pl.LightningModule):
    """SSL trainer for src.dino.DINOv2 (DINO + iBOT + KoLeo).

    Drives the two DINOv2 schedules the model itself doesn't own:
      * EMA momentum  tau: cosine base_tau → final_tau over training (teacher update);
      * teacher temperature: linear warmup over `warmup_teacher_temp_epochs` then constant.
    Also applies the DINO "freeze last layer" trick for the first epochs (stabilises start).
    Validation reuses an embedding criterion (triplet) on a labelled milk loader, like BYOL.
    """

    def __init__(
        self,
        model,
        optimizer_cfg,
        valid_criterion,
        val_loader_names,
        num_classes: int,
        warmup_steps: int,
        monitor_metric: str,
        max_epochs: int,
        base_tau: float = 0.992,
        final_tau: float = 1.0,
        teacher_temp: float = 0.07,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        freeze_last_layer_epochs: int = 1,
    ):
        super().__init__()
        self.model = hydra.utils.instantiate(model, _recursive_=False)
        self.optimizer_config = optimizer_cfg
        self.warmup_steps = warmup_steps
        self.monitor_metric = monitor_metric
        self.max_epochs = max_epochs
        self.base_tau, self.final_tau = base_tau, final_tau
        self.teacher_temp, self.warmup_teacher_temp = teacher_temp, warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.freeze_last_layer_epochs = freeze_last_layer_epochs

        self.valid_criterion = hydra.utils.instantiate(valid_criterion)
        self.val_loader_names = val_loader_names
        self.train_losses = {}
        self.valid_losses_by_loader = {}

    def _set_schedules(self):
        # EMA momentum: cosine base → final over the whole run.
        total = max(self.trainer.estimated_stepping_batches, 1)
        step = self.trainer.global_step
        tau = self.final_tau - (self.final_tau - self.base_tau) * (math.cos(math.pi * step / total) + 1) / 2
        self.model.current_tau.fill_(tau)
        # teacher temperature: linear warmup over epochs.
        e, we = self.current_epoch, max(self.warmup_teacher_temp_epochs, 1)
        if e < we:
            tt = self.warmup_teacher_temp + (self.teacher_temp - self.warmup_teacher_temp) * (e / we)
        else:
            tt = self.teacher_temp
        self.model.teacher_temp.fill_(tt)

    def training_step(self, batch, batch_idx: int):
        self._set_schedules()
        losses = self.model.forward(batch)
        for name, value in losses.items():
            if name not in self.train_losses:
                self.train_losses[name] = torchmetrics.MeanMetric().to(value.device)
            self.train_losses[name].update(value.detach())
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=True)
        self.log("train/tau", float(self.model.current_tau), on_epoch=True)
        return losses["total_loss"]

    def on_after_backward(self):
        # DINO trick: keep the prototype (last) layer frozen for the first epoch(s).
        if self.current_epoch < self.freeze_last_layer_epochs:
            for n, p in self.model.named_parameters():
                if "last_layer" in n and p.grad is not None:
                    p.grad = None

    def on_train_batch_end(self, *args):
        self.model.momentum_update()           # EMA teacher update AFTER the optimizer step

    def on_train_epoch_end(self):
        for name, metric in self.train_losses.items():
            self.log(f"train_epoch/{name}", metric.compute(), sync_dist=True)
            metric.reset()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        embeddings = self.model.extract_embeddings(batch["image"])
        loss = self.valid_criterion(embeddings, batch["label"])
        name = self.val_loader_names[dataloader_idx]
        if name not in self.valid_losses_by_loader:
            self.valid_losses_by_loader[name] = torchmetrics.MeanMetric().to(loss.device)
        self.valid_losses_by_loader[name].update(loss.detach())

    def on_validation_epoch_end(self):
        losses = []
        for name, metric in self.valid_losses_by_loader.items():
            v = metric.compute()
            losses.append(v.item())
            self.log(f"valid/loss/{name}", v, sync_dist=True)
            metric.reset()
        if losses:
            self.log("valid/loss", sum(losses) / len(losses), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = instantiate(self.optimizer_config, params=params)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = build_warmup_cosine(optimizer, self.warmup_steps, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
