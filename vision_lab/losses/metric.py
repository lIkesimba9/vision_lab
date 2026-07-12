"""Метрик-лоссы: triplet semi-hard, SupCon (ТЗ §5.1).

v1 без абстракции miner: структуру батча (гарантию позитивов) обеспечивают
PK-сэмплеры (§5.4) — это и есть «майнер» библиотеки. Оба лосса маскируют
``-1`` (unlabeled не участвует) и возвращают 0 при отсутствии позитивных пар
(сохраняет граф, не роняет DDP).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.heads.primitives import has_positive_pairs, valid_rows


def pairwise_distance(embeddings: torch.Tensor) -> torch.Tensor:
    """Матрица попарных L2-расстояний с численной стабилизацией; диагональ = 0."""
    x = embeddings.float()
    sq = (x * x).sum(dim=1, keepdim=True)
    dist_sq = (sq + sq.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
    dist = dist_sq.sqrt()
    return dist * (1.0 - torch.eye(x.size(0), device=x.device))


class TripletSemiHardLoss(nn.Module):
    """Triplet loss с semi-hard negative mining (Schroff 2015, порт TF-логики).

    Позитивные расстояния тянутся меньше минимального негативного, которое
    при этом больше позитивного на ``margin`` (semi-hard). Нет такого негатива
    — берётся наибольшее негативное расстояние.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask_valid = valid_rows(labels)
        if mask_valid.sum() < 2 or not has_positive_pairs(labels):
            return embeddings.sum() * 0.0
        emb = embeddings[mask_valid]
        lab = labels[mask_valid].view(-1, 1)
        n = emb.size(0)

        pdist = pairwise_distance(emb)
        adjacency = torch.eq(lab, lab.t())
        adjacency_not = ~adjacency

        pdist_tile = pdist.repeat(n, 1)
        adj_not_tile = adjacency_not.repeat(n, 1)
        greater = pdist_tile > pdist.t().reshape(-1, 1)
        mask = adj_not_tile & greater

        mask_final = (mask.float().sum(dim=1) > 0.0).reshape(n, n).t()
        mask_f = mask.float()
        adj_not_f = adjacency_not.float()

        # negatives_outside: наименьшее D_an, где D_an > D_ap
        row_max = pdist_tile.max(dim=1, keepdim=True)[0]
        neg_outside = (torch.min((pdist_tile - row_max) * mask_f, dim=1, keepdim=True)[0]
                       + row_max).reshape(n, n).t()
        # negatives_inside: наибольшее D_an
        row_min = pdist.min(dim=1, keepdim=True)[0]
        neg_inside = (torch.max((pdist - row_min) * adj_not_f, dim=1, keepdim=True)[0]
                      + row_min).repeat(1, n)

        semi_hard = torch.where(mask_final, neg_outside, neg_inside)
        loss_mat = self.margin + pdist - semi_hard

        mask_pos = adjacency.float() - torch.eye(n, device=emb.device)
        num_pos = mask_pos.sum().clamp_min(1.0)
        loss = (loss_mat * mask_pos).clamp_min(0.0).sum() / num_pos
        return loss.to(embeddings.dtype)


class SupConLoss(nn.Module):
    """Supervised Contrastive (Khosla 2020), одна вьюха на семпл.

    Позитивы = семплы того же класса в батче; знаменатель — все, кроме себя.
    Работает на PK-батчах; unlabeled (-1) исключается.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask_valid = valid_rows(labels)
        if mask_valid.sum() < 2 or not has_positive_pairs(labels):
            return embeddings.sum() * 0.0
        emb = F.normalize(embeddings[mask_valid], dim=1)
        lab = labels[mask_valid].view(-1, 1)
        n = emb.size(0)

        logits = (emb @ emb.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        self_mask = torch.eye(n, dtype=torch.bool, device=emb.device)

        pos_mask = torch.eq(lab, lab.t()) & ~self_mask
        exp_logits = torch.exp(logits) * (~self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_count = pos_mask.sum(dim=1)
        has_pos = pos_count > 0
        if not has_pos.any():
            return embeddings.sum() * 0.0
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[has_pos] / pos_count[has_pos]
        return (-mean_log_prob_pos).mean().to(embeddings.dtype)
