"""SimSiam (Chen & He 2021, https://arxiv.org/abs/2011.10566) — «BYOL без EMA».

Одна сиамская ветка: backbone → projector → predictor; таргет — проекция
ВТОРОЙ вьюхи той же (не EMA) сети под stop-gradient. Ключевой вывод статьи:
от коллапса спасает сама асимметрия predictor + stop-grad, учитель не нужен.
Лосс — та же косинусная форма, что у BYOL (``2 - 2·cos`` эквивалентен
``-cos`` статьи с точностью до аффинного сдвига).

Без учителя и без негативов: ``momentum_update`` — no-op, gather по рангам не
нужен. Устойчив к малым батчам (в отличие от SimCLR). ``extract_embeddings`` —
бэкбон.
"""

from __future__ import annotations

import torch
from torch import nn

from vision_lab.ssl.base import SSLMethod
from vision_lab.ssl.byol import byol_cosine_loss
from vision_lab.ssl.components import MLP


class SimSiam(SSLMethod):
    """SimSiam: stop-grad вместо EMA-учителя.

    Компоненты передаются инстанцированными:
        ``backbone`` — :class:`~vision_lab.models.backbones.EmbeddingBackbone`;
        ``views``    — :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment`
                       (2 симметричные вьюхи, рецепт ``simclr_v1``).
    """

    def __init__(self, backbone: nn.Module, views: nn.Module,
                 hidden_dim: int = 2048, projection_dim: int = 2048):
        super().__init__()
        self.backbone = backbone
        self.views = views
        self.projector = MLP(backbone.out_dim, hidden_dim, projection_dim)
        self.predictor = MLP(projection_dim, hidden_dim // 4, projection_dim)

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Вектор бэкбона (единственного — веток-близнецов с своими весами нет)."""
        return self.backbone(images)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        view_set = self.views(batch["image"])
        if len(view_set.globals) < 2:
            raise ValueError("SimSiam требует ровно 2 глобальных вьюхи")
        v1, v2 = view_set.globals[0], view_set.globals[1]

        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        p1, p2 = self.predictor(z1), self.predictor(z2)

        # byol_cosine_loss сам делает stop-grad (detach) на таргет-проекции
        loss = 0.5 * (byol_cosine_loss(p1, z2) + byol_cosine_loss(p2, z1))
        return {"simsiam_loss": loss, "total_loss": loss}
