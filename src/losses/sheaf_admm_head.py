"""Sheaf-ADMM голова-классификатор (совместима с ``ClassifierHead``).

Заменяет ``GlobalAvgPool + Linear`` на координацию агентов над сеткой признаков
backbone'а: каждая пространственная ячейка карты признаков — агент, они согласуются
через разворачиваемый ADMM с обучаемым клеточным пучком, затем каждый декодируется в
локальный логит класса; глобальное предсказание — усреднение по агентам (см. разбор в
../../sheaf_admm/ANALYSIS.md).

Контракт как у прочих голов:
    forward(feature_map, labels) -> {"total_loss", "ce", "consensus_rms"}
    predict_logits(feature_map)  -> [B, C]

ВАЖНО: на вход подаётся КАРТА ПРИЗНАКОВ ``[B, C, H', W']`` (а не пулинг-эмбеддинг).
Backbone должен возвращать её: используйте модель ``SpatialFeatureModel`` (global_pool='').
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.base import ClassifierHead
from src.sheaf_torch import SheafADMMModule


class SheafADMMHead(ClassifierHead):
    def __init__(self, n_class, in_channels, label_smoothing=0.0, **module_kwargs):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = module_kwargs.get("d_v", 32)  # для интерфейса base
        self.in_channels = in_channels
        self.label_smoothing = label_smoothing
        self.module = SheafADMMModule(
            in_channels=in_channels, num_classes=n_class, **module_kwargs
        )

    # --- контракт ClassifierHead ---------------------------------------------
    @property
    def classifier_weight(self):
        # веса финального декодера (если линейный) — для совместимости с переносом FC
        net = self.module.decoder.net
        return net.weight if hasattr(net, "weight") else None

    def _check(self, feature_map):
        if feature_map.dim() != 4:
            raise ValueError(
                "SheafADMMHead ожидает карту признаков [B,C,H,W]; получено "
                f"{tuple(feature_map.shape)}. Используйте SpatialFeatureModel (global_pool='')."
            )

    def forward(self, feature_map, labels):
        self._check(feature_map)
        logits_window, _ = self.module(feature_map)  # [W,B,N,C]
        Wn, B, N, C = logits_window.shape
        # каждый агент на каждой из последних W итераций предсказывает глобальную метку
        tgt = labels.long().view(1, B, 1).expand(Wn, B, N).reshape(-1)
        ce = F.cross_entropy(
            logits_window.reshape(-1, C), tgt, label_smoothing=self.label_smoothing
        )
        return {"total_loss": ce, "ce": ce.detach()}

    @torch.no_grad()
    def _agg_logits(self, feature_map):
        _, agg = self.module(feature_map)
        return agg

    def predict_logits(self, feature_map):
        self._check(feature_map)
        # без no_grad: вызывается и в train-валидации, и в инференсе; агрегированные логиты
        _, agg = self.module(feature_map)
        return agg
