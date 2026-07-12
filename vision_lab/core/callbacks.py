"""Переиспользуемые коллбеки (ТЗ §5.6, §11.1).

Ключевое решение: probe — коллбек, а не логика внутри трейнера. В прототипе
сбор эмбеддингов был зашит в ``BYOLTrainer.validation_step``, из-за чего
``DINOv2Trainer`` не мог его переиспользовать. :class:`KNNProbeCallback`
работает с ЛЮБЫМ :class:`~vision_lab.ssl.base.SSLMethod` через
``extract_embeddings``.
"""

from __future__ import annotations

from collections.abc import Mapping

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from vision_lab.eval.knn_probe import (
    knn_macro_f1,
    linear_probe_macro_f1,
    stratified_gallery_split,
)


class KNNProbeCallback(pl.Callback):
    """Онлайн kNN/linear-probe отбор чекпоинтов SSL по macro-F1.

    Галерея = стратифицированная доля ``gallery_loader`` (по умолчанию dataloader
    idx 0), квери = остаток галереи + остальные val-лоадеры. Полный val-набор на
    КАЖДОМ ранге (детерминизм в DDP, ``use_distributed_sampler=False``).

    Логирует ``val/knn_f1[/<query>]``, ``val/linprobe_f1``, ``val/sel_f1``
    (среднее — пики kNN и probe часто на разных эпохах).
    """

    def __init__(self, gallery_loader_idx: int = 0, loader_names: list[str] | None = None,
                 k: int = 20, gallery_frac: float = 0.5, seed: int = 42,
                 run_linear_probe: bool = True):
        super().__init__()
        self.gallery_idx = gallery_loader_idx
        self.loader_names = loader_names
        self.k = k
        self.gallery_frac = gallery_frac
        self.seed = seed
        self.run_linear_probe = run_linear_probe
        self._emb: dict[int, list] = {}
        self._lab: dict[int, list] = {}

    def _name(self, idx: int) -> str:
        if self.loader_names and idx < len(self.loader_names):
            return self.loader_names[idx]
        return f"loader{idx}"

    def on_validation_epoch_start(self, trainer, pl_module):
        self._emb, self._lab = {}, {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx: int = 0):
        if "label" not in batch:
            return
        emb = pl_module.method.extract_embeddings(batch["image"])
        self._emb.setdefault(dataloader_idx, []).append(emb.float().cpu())
        self._lab.setdefault(dataloader_idx, []).append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.gallery_idx not in self._emb:
            return
        gx = torch.cat(self._emb[self.gallery_idx]).numpy()
        gy = torch.cat(self._lab[self.gallery_idx]).numpy()
        g_idx, q_idx = stratified_gallery_split(gy, self.gallery_frac, self.seed)
        gallery_x, gallery_y = gx[g_idx], gy[g_idx]

        query_sets = {self._name(self.gallery_idx): (gx[q_idx], gy[q_idx])}
        for idx in self._emb:
            if idx == self.gallery_idx:
                continue
            query_sets[self._name(idx)] = (
                torch.cat(self._emb[idx]).numpy(), torch.cat(self._lab[idx]).numpy())

        knn_scores, probe_scores, sel_scores = [], [], []
        for qname, (qx, qy) in query_sets.items():
            if len(qy) == 0 or len(np.unique(gallery_y)) < 2:
                continue
            knn = knn_macro_f1(gallery_x, gallery_y, qx, qy, k=self.k)
            pl_module.log(f"val/knn_f1/{qname}", knn, sync_dist=True)
            knn_scores.append(knn)
            if self.run_linear_probe:
                probe = linear_probe_macro_f1(gallery_x, gallery_y, qx, qy)
                pl_module.log(f"val/linprobe_f1/{qname}", probe, sync_dist=True)
                probe_scores.append(probe)
                sel_scores.append(0.5 * (knn + probe))

        if knn_scores:
            pl_module.log("val/knn_f1", float(np.mean(knn_scores)), prog_bar=True, sync_dist=True)
        if probe_scores:
            pl_module.log("val/linprobe_f1", float(np.mean(probe_scores)), sync_dist=True)
        if sel_scores:
            pl_module.log("val/sel_f1", float(np.mean(sel_scores)), prog_bar=True, sync_dist=True)


def topk_per_metric_checkpoints(metrics: Mapping[str, str], dirpath: str | None = None,
                                k: int = 3) -> list[ModelCheckpoint]:
    """Фабрика: по одному штатному ModelCheckpoint на метрику (ТЗ §5.6).

    ``metrics`` — {имя_метрики -> "max"|"min"}. Пики разных метрик (kNN,
    linprobe, их среднее) приходятся на разные эпохи, поэтому храним top-K по
    каждой отдельно. Переиспользуем проверенную resume-логику Lightning.
    """
    callbacks = []
    for metric, mode in metrics.items():
        safe = metric.replace("/", "_")
        callbacks.append(ModelCheckpoint(
            dirpath=dirpath, monitor=metric, mode=mode, save_top_k=k,
            filename=f"{{epoch}}-{{{metric}:.4f}}".replace("/", "_"),
            auto_insert_metric_name=False,
            save_last=(metric == next(iter(metrics))),
        ))
        callbacks[-1].CHECKPOINT_NAME_LAST = f"last-{safe}"
    return callbacks


class FreezeParams(pl.Callback):
    """Замораживает параметры по шаблонам имён до эпохи ``until_epoch``.

    DINO freeze-last-layer: без этого prototype-слой расходится на старте.
    Обнуляет градиенты подходящих параметров в ``on_after_backward``.
    """

    def __init__(self, patterns: list[str], until_epoch: int = 1):
        super().__init__()
        self.patterns = patterns
        self.until_epoch = until_epoch

    def on_after_backward(self, trainer, pl_module):
        if trainer.current_epoch >= self.until_epoch:
            return
        for name, p in pl_module.named_parameters():
            if p.grad is not None and any(pat in name for pat in self.patterns):
                p.grad = None
