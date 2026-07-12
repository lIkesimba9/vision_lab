"""Multi-task голова: композиция под-голов (ТЗ §5.1).

Каждая под-голова читает свою метку по ``target_key`` из ОБЩЕГО словаря
таргетов и сама маскирует ``-1`` (невалидные/неразмеченные строки задачи).
Так multi-task не требует особого трейнера и особой формы батча — только
дополнительные ключи ``label_<task>`` (§7.2). Это заменяет прототипный
``forward(emb, labels, labels2=None)`` и ветку ``if "label11" in batch``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from vision_lab.heads.base import ClassifierHead


class MultiTaskHead(ClassifierHead):
    """Взвешенная сумма лоссов под-голов; predict_logits — по основной задаче.

    ``heads`` — маппинг «имя задачи -> ClassifierHead» (у каждой свой
    ``target_key``). ``weights`` — веса лоссов (по умолчанию 1.0). ``primary`` —
    имя задачи, чьи логиты идут в метрики/инференс.
    """

    def __init__(self, heads: Mapping[str, ClassifierHead],
                 primary: str, weights: Mapping[str, float] | None = None):
        super().__init__()
        if primary not in heads:
            raise ValueError(f"primary={primary!r} не среди голов {list(heads)}")
        self.heads = nn.ModuleDict(dict(heads))
        self.primary = primary
        self.weights = {name: float((weights or {}).get(name, 1.0)) for name in self.heads}
        primary_head = self.heads[primary]
        self.n_class = primary_head.n_class
        self.embedding_dim = primary_head.embedding_dim
        self.target_key = primary_head.target_key

    @property
    def classifier_weight(self):
        return self.heads[self.primary].classifier_weight

    def forward(self, embeddings, targets: Mapping[str, torch.Tensor]):
        out: dict[str, torch.Tensor] = {}
        total = embeddings.sum() * 0.0
        for name, head in self.heads.items():
            loss = head(embeddings, targets)["total_loss"]
            out[f"{name}_loss"] = loss
            total = total + self.weights[name] * loss
        out["total_loss"] = total
        return out

    def predict_logits(self, embeddings):
        return self.heads[self.primary].predict_logits(embeddings)
