"""Иерархические labeled-лоссы поверх BYOL для SSL-предобучения.

Оба плеча работают на эмбеддингах BYOL (по умолчанию online-проекции z*_o, опц. backbone-фичи
h*_o — то, что меряет kNN) и используют per-image иерархические метки: diag-уровни из
batch['levels'] + целевые 11 классов из batch['label']. Метка -1 игнорируется.

  ByolHierSupConLoss — BYOL + multi-level Supervised Contrastive (моя идея):
      по каждому уровню тянем позитивы одного класса с убывающим весом → «центроиды внутри
      центроидов». Опции: feat_source (proj|backbone), memory-bank (больше позитивов/негативов
      для хвоста: DF/INF/VASC/MAL_OTH).
  ByolHierAAMLoss    — BYOL + multi-level AAM-Softmax (идея пользователя):
      отдельная AAM-голова (обучаемые центроиды) на каждый уровень, взвешенная сумма.
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.byol_loss import byol_multicrop_loss
from src.losses.heads import AAMHead
from src.losses.speaker import AngleProtoLoss, GE2ELoss


def group_by_label(emb: torch.Tensor, labels: torch.Tensor, m: int) -> Optional[torch.Tensor]:
    """Перегруппировать (K, D) эмбеддинги в (N, m, D): N классов с >=m примерами, по m случайных
    на класс (для speaker-лоссов GE2E/AngleProto, которым нужен вход (N, M, D)). label==-1 игнор.
    Возвращает None, если классов с >=m примеров меньше 2."""
    groups = []
    for c in labels.unique():
        if c.item() == -1:
            continue
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() >= m:
            sel = idx[torch.randperm(idx.numel(), device=emb.device)[:m]]
            groups.append(emb[sel])
    if len(groups) < 2:
        return None
    return torch.stack(groups, dim=0)


def supcon_loss(f: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1,
                bank_f: Optional[torch.Tensor] = None,
                bank_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla 2020, L_out) на L2-нормированных эмбеддингах.
    Якоря = f (с градиентом); кандидаты-ключи = f (+ опц. memory-bank, detached). Строки с
    label==-1 игнорируются. Bank даёт больше позитивов/негативов (важно для редких классов)."""
    valid = labels != -1
    f, labels = f[valid], labels[valid]
    A = f.size(0)
    if A < 2:
        return f.new_tensor(0.0)

    if bank_f is not None and bank_f.numel() > 0:
        keys = torch.cat([f, bank_f], dim=0)
        klabels = torch.cat([labels, bank_labels], dim=0)
    else:
        keys, klabels = f, labels

    sim = (f @ keys.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    # self-маска: якорь i соответствует ключу i (первые A ключей — сами якоря)
    self_mask = torch.zeros(A, keys.size(0), dtype=torch.bool, device=f.device)
    diag = torch.arange(A, device=f.device)
    self_mask[diag, diag] = True
    cand = ~self_mask
    pos = (labels[:, None] == klabels[None, :]) & cand

    log_prob = sim - torch.log((sim.exp() * cand).sum(1, keepdim=True) + 1e-12)
    pos_count = pos.sum(1)
    has_pos = pos_count > 0
    if not has_pos.any():
        return f.new_tensor(0.0)
    mean_log_prob_pos = (pos * log_prob).sum(1)[has_pos] / pos_count[has_pos]
    return -mean_log_prob_pos.mean()


def _stack_level_labels(batch, include_target_label: bool) -> List[torch.Tensor]:
    """Метки по уровням (каждая (2N,), дублирована под две вьюхи): diag-уровни coarse->fine,
    затем (опц.) целевые 11 классов."""
    levels = []
    lv = batch.get("levels")
    if lv is not None:
        lv = lv.long()
        levels += [torch.cat([lv[:, j], lv[:, j]]) for j in range(lv.size(1))]
    if include_target_label:
        lab = batch["label"].view(-1).long()
        levels.append(torch.cat([lab, lab]))
    return levels


class _ByolHierLoss(nn.Module):
    """База: BYOL + взвешенная сумма per-level labeled-членов.

    feat_source: 'proj' — online-проекции z*_o (256d); 'backbone' — фичи h*_o (feature_size).
    Подкласс задаёт ``key`` (имя суммарного члена), ``_level_term(j, z, labels)`` и опц. ``_embeddings``."""

    key: str = "labeled_loss"

    def __init__(self, level_weights: List[float], byol_weight: float, label_weight: float,
                 include_target_label: bool, feat_source: str = "proj"):
        super().__init__()
        assert feat_source in ("proj", "backbone"), feat_source
        self.level_weights = list(level_weights)
        self.byol_weight = byol_weight
        self.label_weight = label_weight
        self.include_target_label = include_target_label
        self.feat_source = feat_source

    def _raw(self, output) -> torch.Tensor:
        if self.feat_source == "backbone":
            return torch.cat([output["h1_o"], output["h2_o"]], dim=0)
        return torch.cat([output["z1_o"], output["z2_o"]], dim=0)

    def _embeddings(self, output) -> torch.Tensor:
        return self._raw(output)

    def _level_term(self, j: int, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _after_levels(self, z: torch.Tensor, level_labels: List[torch.Tensor]) -> None:
        pass

    def forward(self, output, batch):
        byol = byol_multicrop_loss(output)
        z = self._embeddings(output)
        level_labels = _stack_level_labels(batch, self.include_target_label)
        assert len(level_labels) == len(self.level_weights), \
            f"level_weights={len(self.level_weights)} != уровней={len(level_labels)}"

        labeled = z.new_tensor(0.0)
        logs = {}
        for j, (labels, w) in enumerate(zip(level_labels, self.level_weights)):
            lj = self._level_term(j, z, labels)
            logs[f"{self.key}_l{j}"] = lj.detach()
            labeled = labeled + w * lj
        self._after_levels(z, level_labels)

        return {
            "byol_loss": self.byol_weight * byol,
            self.key: self.label_weight * labeled,
            "total_loss": self.byol_weight * byol + self.label_weight * labeled,
            **logs,
        }


class ByolHierSupConLoss(_ByolHierLoss):
    key = "supcon_loss"

    def __init__(self, level_weights: List[float], temperature: float = 0.1,
                 byol_weight: float = 1.0, supcon_weight: float = 1.0,
                 include_target_label: bool = True, feat_source: str = "proj",
                 bank_size: int = 0, embedding_dim: Optional[int] = None):
        super().__init__(level_weights, byol_weight, supcon_weight, include_target_label, feat_source)
        self.temperature = temperature
        self.bank_size = bank_size
        n_levels = len(self.level_weights)
        if bank_size > 0:
            assert embedding_dim is not None, "для memory-bank нужен embedding_dim"
            # единая очередь эмбеддингов + мультиуровневые метки (FIFO), на ранг свой банк
            self.register_buffer("bank_emb", torch.zeros(bank_size, embedding_dim))
            self.register_buffer("bank_lab", torch.full((bank_size, n_levels), -1, dtype=torch.long))
            self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))

    def _embeddings(self, output):
        return F.normalize(self._raw(output), dim=1)   # supcon на нормированных, считаем один раз

    def _level_term(self, j, z, labels):
        bank_f = bank_lab = None
        if self.bank_size > 0:
            mask = self.bank_lab[:, j] != -1
            if mask.any():
                bank_f, bank_lab = self.bank_emb[mask], self.bank_lab[mask, j]
        return supcon_loss(z, labels, self.temperature, bank_f, bank_lab)

    @torch.no_grad()
    def _after_levels(self, z, level_labels):
        if self.bank_size == 0:
            return
        emb = z.detach()
        labs = torch.stack(level_labels, dim=1)          # (2N, n_levels)
        n, ptr, B = emb.size(0), int(self.bank_ptr.item()), self.bank_size
        idx = (ptr + torch.arange(n, device=emb.device)) % B   # FIFO с заворотом
        self.bank_emb[idx] = emb
        self.bank_lab[idx] = labs
        self.bank_ptr[0] = (ptr + n) % B


class ByolHierSupConAngleProto(_ByolHierLoss):
    """Гибрид: BYOL + SupCon на всех уровнях, КРОМЕ одного (angle_level, по умолчанию diag_3),
    где вместо SupCon применяется speaker-verification лосс (AngleProto/GE2E) на PK-структуре
    (N диагнозов × M примеров). Прочие уровни (diag1/diag2/11-класс) — SupCon (лучшее из эксп.).
    unlabeled (label==-1) -> только BYOL."""

    key = "labeled_loss"

    def __init__(self, level_weights: List[float], temperature: float = 0.1,
                 byol_weight: float = 1.0, label_weight: float = 1.0,
                 include_target_label: bool = True, feat_source: str = "proj",
                 angle_level: int = 2, samples_per_class: int = 4,
                 speaker_method: str = "angleproto", init_w: float = 10.0, init_b: float = -5.0):
        super().__init__(level_weights, byol_weight, label_weight, include_target_label, feat_source)
        self.temperature = temperature
        self.angle_level = angle_level          # индекс уровня в [diag1,diag2,diag3,(11-class)]; diag_3=2
        self.M = samples_per_class
        if speaker_method == "angleproto":
            self.speaker = AngleProtoLoss(init_w, init_b)
        elif speaker_method == "ge2e":
            self.speaker = GE2ELoss(init_w, init_b)
        else:
            raise ValueError(speaker_method)

    def _embeddings(self, output):
        return F.normalize(self._raw(output), dim=1)

    def _level_term(self, j, z, labels):
        if j == self.angle_level:
            grouped = group_by_label(z, labels, self.M)   # (N, M, D)
            return self.speaker(grouped) if grouped is not None else z.new_tensor(0.0)
        return supcon_loss(z, labels, self.temperature)


class ByolHierAAMLoss(_ByolHierLoss):
    key = "aam_loss"

    def __init__(self, embedding_dim: int, level_sizes: List[int], num_target_classes: int,
                 level_weights: List[float], m: float = 0.2, s: float = 30.0,
                 byol_weight: float = 1.0, aam_weight: float = 1.0,
                 include_target_label: bool = True, feat_source: str = "proj"):
        super().__init__(level_weights, byol_weight, aam_weight, include_target_label, feat_source)
        sizes = list(level_sizes) + ([num_target_classes] if include_target_label else [])
        assert len(sizes) == len(self.level_weights), \
            f"level_weights={len(self.level_weights)} != голов={len(sizes)}"
        self.heads = nn.ModuleList([AAMHead(n, embedding_dim, m=m, s=s) for n in sizes])

    def _level_term(self, j, z, labels):
        mask = labels != -1
        if mask.sum() < 2:
            return z.new_tensor(0.0)
        return self.heads[j](z[mask], labels[mask])["total_loss"]
