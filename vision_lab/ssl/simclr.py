"""SimCLR (Chen 2020, https://arxiv.org/abs/2002.05709) — contrastive SSL.

NT-Xent: два аугментированных вида каждого изображения — позитивная пара,
остальные ``2B-2`` примеров батча — негативы. Качество растёт с числом
негативов, поэтому в DDP проекции СОБИРАЮТСЯ со всех рангов ЯВНО
(:func:`~vision_lab.core.dist.all_gather_grad`, §11.1) — при world_size=1 это
identity, код одинаков на одном GPU и в кластере.

Без учителя (``momentum_update`` — no-op). ``extract_embeddings`` — бэкбон
(проекционная голова после претрейна выбрасывается, как в статье).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.core.dist import all_gather_grad
from vision_lab.ssl.base import SSLMethod
from vision_lab.ssl.components import MLP


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent по двум наборам проекций ``(B, D)``: позитив ``i ↔ i+B``.

    Вход — УЖЕ собранные с рангов проекции (gather — забота вызывающего).
    """
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z.shape[0]
    sim = (z @ z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))  # исключаем self-пару из знаменателя
    b = n // 2
    targets = torch.cat([torch.arange(b, n), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


class SimCLR(SSLMethod):
    """SimCLR: симметричные вьюхи (рецепт ``simclr_v1``) + NT-Xent.

    Компоненты передаются инстанцированными:
        ``backbone`` — :class:`~vision_lab.models.backbones.EmbeddingBackbone`;
        ``views``    — :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment`
                       (ровно 2 глобальных вьюхи).

    Эффективное число негативов = ``2·B·world_size - 2`` — большой батч
    критичен (урок статьи); при малом батче предпочтителен BYOL/MoCo.
    """

    def __init__(self, backbone: nn.Module, views: nn.Module,
                 hidden_dim: int = 2048, projection_dim: int = 128,
                 temperature: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.views = views
        self.projector = MLP(backbone.out_dim, hidden_dim, projection_dim)
        self.temperature = temperature

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Вектор бэкбона (проекционная голова для probe не используется)."""
        return self.backbone(images)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        view_set = self.views(batch["image"])
        if len(view_set.globals) < 2:
            raise ValueError("SimCLR требует ровно 2 глобальных вьюхи")
        v1, v2 = view_set.globals[0], view_set.globals[1]
        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        loss = nt_xent_loss(all_gather_grad(z1), all_gather_grad(z2), self.temperature)
        return {"nt_xent_loss": loss, "total_loss": loss}
