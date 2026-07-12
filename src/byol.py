# Copyright (C) 2023. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
import copy
from typing import List, Tuple, Any
from hydra.utils import instantiate
import numpy as np
    

# Метки, для которых по умолчанию НЕ делаем внутриклассовое перемешивание вьюхи-2:
#   -1  — no_label (класса нет),
#   10  —
# Для них работает обычный BYOL: вьюха-2 = аугментация того же изображения.
# Переопределяется из конфига: config.no_shuffle_labels.
NO_POSITIVE_SHUFFLE_LABELS = (-1, 10)


def shuffle_inplace(feats, labels, no_shuffle_labels=NO_POSITIVE_SHUFFLE_LABELS):
    """Внутри каждого класса перемешивает строки feats (вьюху-2), делая снимки одного
    класса позитивными парами (SupCon-style). Классы из no_shuffle_labels пропускаются."""
    no_shuffle = set(no_shuffle_labels)
    for cls in labels.unique():
        if cls.item() in no_shuffle:
            continue
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) > 1:
            perm = idx[torch.randperm(len(idx), device=feats.device)]
            feats[idx] = feats[perm]

    return feats


def finest_group_labels(label, levels, skip_target_labels=(-1, 10), big=100000):
    """Группа на КАЖДЫЙ снимок = самый ТОНКИЙ доступный уровень (для BYOL-перемешивания
    позитивов на всех уровнях иерархии, а не только на 11-классе).

    Порядок (грубый -> тонкий, тонкий перетирает): diag-уровни levels[:, 0..L-1], затем 11-класс.
    11-класс пропускаем для skip_target_labels (-1 и BEN_OTH=10): BEN_OTH тогда останется на своём
    diag3 (корректнее, чем грубый «прочий доброкач.»). Уровни неймспейсятся (j+1)*big+class_id,
    чтобы один и тот же class_id с разных уровней не сливался. Снимок без меток -> -1 (не мешается).
    """
    N = label.size(0)
    g = torch.full((N,), -1, dtype=torch.long, device=label.device)
    if levels is not None:
        levels = levels.long()
        for j in range(levels.size(1)):
            col = levels[:, j]
            valid = col != -1
            g[valid] = (j + 1) * big + col[valid]
    skip = set(int(s) for s in skip_target_labels)
    tgt_valid = torch.ones(N, dtype=torch.bool, device=label.device)
    for s in skip:
        tgt_valid &= label != s
    g[tgt_valid] = 0 * big + label[tgt_valid]   # неймспейс уровня "0" = 11-класс (самый тонкий target)
    return g


class BYOL(nn.Module):
    """ 
    BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
    Link: https://arxiv.org/abs/2006.07733
    Implementation: https://github.com/deepmind/deepmind-research/tree/master/byol
    """
    def __init__(self, config):
        super().__init__()
        self.projection_dim = config.projection_dim
        self.tau = config.tau # EMA update
        self.current_tau = config.tau
        
        self.backbone = instantiate(config.backbone)

        # Внутриклассовое перемешивание вьюхи-2 (SupCon-style позитивные пары):
        #   positive_pair_shuffle — вкл/выкл аугментацию (False → обычный BYOL);
        #   no_shuffle_labels     — метки, которые НЕ перемешиваем (напр. [-1, 10]).
        self.positive_pair_shuffle = config.get("positive_pair_shuffle", True)
        self.no_shuffle_labels = tuple(
            config.get("no_shuffle_labels", NO_POSITIVE_SHUFFLE_LABELS)
        )
        # 'label' — перемешивать позитивы по 11-классу (как было); 'finest' — по самому тонкому
        # доступному уровню иерархии на каждый снимок (включает unlabeled-с-diag и BEN_OTH по diag3).
        self.shuffle_source = config.get("shuffle_source", "label")

        feature_size = config.feature_size
        
        self.projector = MLP(feature_size, hidden_dim=config.hidden_dim, out_dim=config.projection_dim)

        
        self.online_encoder = self.encoder = nn.Sequential(
                self.backbone,
                self.projector
        )
        self.online_predictor = MLP(in_dim=config.projection_dim, hidden_dim=config.hidden_dim, out_dim=config.projection_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder) # target must be a deepcopy of online, since we will use the backbone trained by online
        self._init_target_encoder()

        
        # _convert_="all": OmegaConf ListConfig (напр. RandomResizedCrop size=[H,W]) -> нативные
        # python-типы, иначе kornia/torch.interpolate падает на ListConfig в output_size.
        self.augment1 = instantiate(config.augment1, _convert_="all")
        self.augment2 = instantiate(config.augment2, _convert_="all")

        # multi-crop (DINO-style): n_local_crops локальных вьюх меньшего размера идут только через
        # online (student) и предсказывают глобальные target-вьюхи. 0 -> обычный 2-view BYOL.
        self.n_local_crops = config.get("n_local_crops", 0)
        self.augment_local = (
            instantiate(config.augment_local, _convert_="all")
            if self.n_local_crops > 0 and "augment_local" in config else None
        )

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone(images)
        return embeddings
    def forward(self, batch):
        x = batch['image']
        labels = batch['label']
        x1 = self.augment1(x)
        x2 = self.augment2(x)
        if self.positive_pair_shuffle:
            if self.shuffle_source == "label":
                x2 = shuffle_inplace(x2, labels, self.no_shuffle_labels)
            elif self.shuffle_source == "finest":
                groups = finest_group_labels(labels, batch.get("levels"), self.no_shuffle_labels)
                x2 = shuffle_inplace(x2, groups, no_shuffle_labels=(-1,))
            else:
                # имя diag-уровня (напр. diagnosis_3): группируем позитивы по нему,
                # -1 (нет метки на этом уровне) не перемешиваем (обычный BYOL для них)
                from src.data.hier import LEVEL_COLUMNS
                idx = LEVEL_COLUMNS.index(self.shuffle_source)
                groups = batch["levels"][:, idx].long()
                x2 = shuffle_inplace(x2, groups, no_shuffle_labels=(-1,))
        # backbone-фичи (h*_o) и projection (z*_o) раздельно: supervised-член может работать на
        # backbone-пространстве (то, что меряет kNN), а не только на projection.
        h1_o, h2_o = self.backbone(x1), self.backbone(x2)
        z1_o, z2_o = self.projector(h1_o), self.projector(h2_o)
        p1_o, p2_o = self.online_predictor(z1_o), self.online_predictor(z2_o)
        # todo byol
        with torch.no_grad():
            self._momentum_update_target_encoder(self.current_tau)
            z1_t, z2_t = self.target_encoder(x1), self.target_encoder(x2)

        out = {
            "p1_o": p1_o,
            "z2_t": z2_t,
            "p2_o": p2_o,
            "z1_t": z1_t,
            "z1_o": z1_o,
            "z2_o": z2_o,
            "h1_o": h1_o,
            "h2_o": h2_o,
        }
        # локальные вьюхи (только online): предсказывают глобальные target-вьюхи в multi-crop лоссе
        if self.augment_local is not None:
            preds = []
            for _ in range(self.n_local_crops):
                xl = self.augment_local(x)
                preds.append(self.online_predictor(self.projector(self.backbone(xl))))
            out["pl_o"] = torch.stack(preds, dim=0)   # (n_local, B, proj_dim)
        return out
    
    def _init_target_encoder(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
            
    @torch.no_grad()
    def _momentum_update_target_encoder(self, tau):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = tau * param_t.data + (1. - tau) * param_o.data



class MLP(nn.Module):
    """ Projection Head and Prediction Head for BYOL """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
    
    
