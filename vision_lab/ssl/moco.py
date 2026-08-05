"""MoCo v3 (Chen 2021, https://arxiv.org/abs/2104.02057) — momentum contrastive.

Онлайн-ветка (backbone → projector → predictor) выдаёт запросы q; EMA-ветка
(:class:`~vision_lab.ssl.base.MomentumTeacher` над backbone+projector) — ключи k
без градиента. InfoNCE: позитив — ключ того же изображения, негативы — ключи
остальных изображений ГЛОБАЛЬНОГО батча (gather ключей по рангам; градиент
через ключи не течёт, поэтому gather безопасен). Симметрично: q1↔k2 и q2↔k1;
лосс масштабируется ``2·τ`` (как в официальной репе).

Очереди memory-bank (MoCo v1/v2) нет — v3 работает от негативов батча.
EMA-момент — ``current_tau`` (пишет ScheduleDriver); ``extract_embeddings`` —
online-бэкбон.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.core.dist import all_gather_grad, rank
from vision_lab.ssl.base import MomentumTeacher, SSLMethod
from vision_lab.ssl.components import MLP


class MoCoV3(SSLMethod):
    """MoCo v3: BYOL-инфраструктура + InfoNCE на негативах батча.

    Компоненты передаются инстанцированными:
        ``backbone`` — :class:`~vision_lab.models.backbones.EmbeddingBackbone`;
        ``views``    — :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment`
                       (2 глобальных вьюхи; асимметричный ``byol_v1`` — как в статье).
    """

    def __init__(self, backbone: nn.Module, views: nn.Module,
                 hidden_dim: int = 4096, projection_dim: int = 256,
                 temperature: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.views = views
        self.temperature = temperature
        feat = backbone.out_dim
        self.projector = MLP(feat, hidden_dim, projection_dim)
        self.online_encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = MLP(projection_dim, hidden_dim, projection_dim)
        self.momentum_encoder = MomentumTeacher(self.online_encoder)

    def momentum_update(self) -> None:
        self.momentum_encoder.update(self.online_encoder, float(self.current_tau))

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Вектор online-бэкбона (как BYOL; momentum-ветка — только для ключей)."""
        return self.backbone(images)

    def _contrastive(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """InfoNCE(q, k): позитив — свой индекс в собранном по рангам наборе ключей."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1).detach()
        k_all = all_gather_grad(k)  # градиента в k нет — gather просто собирает негативы
        logits = (q @ k_all.t()) / self.temperature
        labels = torch.arange(q.shape[0], device=q.device) + rank() * q.shape[0]
        return F.cross_entropy(logits, labels) * (2.0 * self.temperature)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        view_set = self.views(batch["image"])
        if len(view_set.globals) < 2:
            raise ValueError("MoCo v3 требует >= 2 глобальных вьюхи")
        v1, v2 = view_set.globals[0], view_set.globals[1]

        q1 = self.predictor(self.online_encoder(v1))
        q2 = self.predictor(self.online_encoder(v2))
        with torch.no_grad():
            k1 = self.momentum_encoder(v1)
            k2 = self.momentum_encoder(v2)

        loss = self._contrastive(q1, k2) + self._contrastive(q2, k1)
        return {"moco_loss": loss, "total_loss": loss}
