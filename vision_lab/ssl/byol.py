"""BYOL (порт боевой реализации прототипа, ТЗ §5.2).

BYOL: Bootstrap Your Own Latent (https://arxiv.org/abs/2006.07733). Первым по
ТЗ — рабочий на реальных данных SSL-метод. Кастомные приёмы прототипа как
конфигурируемые опции:

* **positive-shuffle** — внутриклассовое перемешивание вьюхи-2
  (supervised-contrastive без негативов), с исключениями (``no_shuffle_labels``)
  и выбором источника группировки (``shuffle_source``: 'label' / 'levels:<уровень>');
* **multi-crop** — n локальных вьюх только через online-энкодер;
* иерархия — группировка позитивов по уровню таксономии.

EMA-учитель — :class:`~vision_lab.ssl.base.MomentumTeacher` (обновляется
трейнером через ``momentum_update``, НЕ внутри forward — это фикс бага
прототипа под grad accumulation). ``extract_embeddings`` использует online-бэкбон
(проверено прототипом).
"""

from __future__ import annotations

import torch
from torch import nn

from vision_lab.core.batch import MISSING_LABEL
from vision_lab.heads.primitives import valid_rows
from vision_lab.ssl.base import MomentumTeacher, SSLMethod
from vision_lab.ssl.components import MLP


def shuffle_within_groups(views: torch.Tensor, groups: torch.Tensor,
                          no_shuffle: tuple[int, ...] = (MISSING_LABEL,)) -> torch.Tensor:
    """Внутри каждой группы перемешивает строки (делает снимки группы позитивами).

    Группы из ``no_shuffle`` пропускаются (обычный BYOL для них). Возвращает
    новый тензор (вход не мутируется).
    """
    out = views.clone()
    skip = set(no_shuffle)
    for g in groups.unique().tolist():
        if g in skip:
            continue
        idx = (groups == g).nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            perm = idx[torch.randperm(idx.numel(), device=views.device)]
            out[idx] = views[perm]
    return out


def byol_cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """2 - 2·cos(p, sg(z)) — BYOL-регрессия предсказания к target-проекции."""
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()


class BYOL(SSLMethod):
    """BYOL с positive-shuffle, multi-crop и иерархическими опциями.

    Компоненты передаются инстанцированными (никаких config-объектов):
        ``backbone`` — :class:`~vision_lab.models.backbones.EmbeddingBackbone`;
        ``views``    — :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment`
                       (>=2 глобальных вьюхи; локальные — multi-crop).
    """

    def __init__(self, backbone: nn.Module, views: nn.Module,
                 hidden_dim: int = 4096, projection_dim: int = 256,
                 positive_shuffle: bool = False, shuffle_source: str = "label",
                 no_shuffle_labels: tuple[int, ...] = (MISSING_LABEL,)):
        super().__init__()
        self.backbone = backbone
        self.views = views
        feat = backbone.out_dim
        self.projector = MLP(feat, hidden_dim, projection_dim)
        self.online_encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = MLP(projection_dim, hidden_dim, projection_dim)
        self.target_encoder = MomentumTeacher(self.online_encoder)

        self.positive_shuffle = positive_shuffle
        self.shuffle_source = shuffle_source
        self.no_shuffle_labels = tuple(no_shuffle_labels)

    def momentum_update(self) -> None:
        self.target_encoder.update(self.online_encoder, float(self.current_tau))

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Вектор online-бэкбона (то, что меряет kNN-probe)."""
        return self.backbone(images)

    def _groups(self, batch: dict) -> torch.Tensor | None:
        """Группы для positive-shuffle: по метке или уровню таксономии."""
        if self.shuffle_source == "label":
            return batch.get("label")
        if self.shuffle_source.startswith("levels:"):
            level_idx = int(self.shuffle_source.split(":", 1)[1])
            levels = batch.get("levels")
            return levels[:, level_idx] if levels is not None else None
        raise ValueError(f"shuffle_source={self.shuffle_source!r} (label|levels:<idx>)")

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x = batch["image"]
        view_set = self.views(x)
        globals_ = view_set.globals
        if len(globals_) < 2:
            raise ValueError("BYOL требует >= 2 глобальных вьюхи")
        v1, v2 = globals_[0], globals_[1]

        # positive-shuffle вьюхи-2: снимки одной группы становятся позитивами
        if self.positive_shuffle:
            groups = self._groups(batch)
            if groups is not None:
                v2 = shuffle_within_groups(v2, groups.long(), self.no_shuffle_labels)

        # online: проекция + предсказание обеих глобальных вьюх
        z1_o = self.projector(self.backbone(v1))
        z2_o = self.projector(self.backbone(v2))
        p1_o, p2_o = self.predictor(z1_o), self.predictor(z2_o)

        # target (EMA, no grad): проекции глобальных вьюх
        with torch.no_grad():
            z1_t = self.target_encoder(v1)
            z2_t = self.target_encoder(v2)

        # симметричный BYOL: p1↔z2, p2↔z1
        terms = [byol_cosine_loss(p1_o, z2_t), byol_cosine_loss(p2_o, z1_t)]

        # multi-crop: локальные вьюхи (только online) тянутся к обеим глобальным target
        for local in view_set.locals:
            pl = self.predictor(self.projector(self.backbone(local)))
            terms.append(byol_cosine_loss(pl, z1_t))
            terms.append(byol_cosine_loss(pl, z2_t))

        byol = sum(terms) / len(terms)
        return {"byol_loss": byol, "total_loss": byol}


class BYOLTriplet(BYOL):
    """BYOL + supervised triplet-член на online-проекциях (иерархический SSL, §5.2).

    triplet на PK-батче структурирует пространство размеченными парами; unlabeled
    (-1) в triplet не участвует, но полноценно работает в BYOL-члене.
    """

    def __init__(self, *args, triplet_weight: float = 1.0, triplet_margin: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        from vision_lab.losses.metric import TripletSemiHardLoss

        self.triplet = TripletSemiHardLoss(margin=triplet_margin)
        self.triplet_weight = triplet_weight

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x = batch["image"]
        view_set = self.views(x)
        v1, v2 = view_set.globals[0], view_set.globals[1]
        z1_o = self.projector(self.backbone(v1))
        z2_o = self.projector(self.backbone(v2))
        p1_o, p2_o = self.predictor(z1_o), self.predictor(z2_o)
        with torch.no_grad():
            z1_t = self.target_encoder(v1)
            z2_t = self.target_encoder(v2)
        byol = 0.5 * (byol_cosine_loss(p1_o, z2_t) + byol_cosine_loss(p2_o, z1_t))

        labels = batch.get("label")
        trip = x.new_tensor(0.0)
        if labels is not None:
            emb = torch.cat([z1_o, z2_o])
            lab = torch.cat([labels, labels]).long()
            trip = self.triplet(emb[valid_rows(lab)], lab[valid_rows(lab)])

        total = byol + self.triplet_weight * trip
        return {"byol_loss": byol, "triplet_loss": trip, "total_loss": total}
