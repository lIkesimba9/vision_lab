"""Multi-task голова: композиция под-голов (ТЗ §5.1).

Каждая под-голова читает свою метку по ``target_key`` из ОБЩЕГО словаря
таргетов и сама маскирует ``-1`` (невалидные/неразмеченные строки задачи).
Так multi-task не требует особого трейнера и особой формы батча — только
дополнительные ключи ``label_<task>`` (§7.2). Это заменяет прототипный
``forward(emb, labels, labels2=None)`` и ветку ``if "label11" in batch``.

Иерархическая классификация — частный случай multi-task: под-голова на
уровень таксономии (:func:`hierarchical_head`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from vision_lab.heads.base import ClassifierHead
from vision_lab.heads.classification import LinearHead


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


def hierarchical_head(
    taxonomy,
    embedding_dim: int,
    head_factory: Callable[..., ClassifierHead] | None = None,
    weights: Mapping[str, float] | None = None,
    primary_level: str | None = None,
) -> MultiTaskHead:
    """Иерархическая классификация (§5.1): под-голова на уровень таксономии.

    Под-голова уровня ``L`` читает ключ батча ``label_<L>`` — эти ключи
    добавляет :class:`~vision_lab.data.manifest.ManifestDataset` при заданной
    ``taxonomy`` (предки выводятся из самой специфичной метки, ``-1`` на
    уровнях тоньше метки маскируется автоматически — частичная глубина
    разметки работает из коробки).

    ``taxonomy`` — :class:`~vision_lab.data.taxonomy.Taxonomy` (нужны
    ``levels`` и ``num_classes``). ``head_factory(n_class, embedding_dim,
    target_key) -> ClassifierHead`` — фабрика под-голов (Hydra: ``_partial_``);
    по умолчанию ``LinearHead(mode="ce")`` — baseline-first (§5.1).
    ``primary_level`` — чьи логиты идут в метрики/инференс; по умолчанию
    самый тонкий (последний) уровень.
    """
    if head_factory is None:
        def head_factory(n_class: int, embedding_dim: int, target_key: str) -> ClassifierHead:
            return LinearHead(n_class, embedding_dim, mode="ce", target_key=target_key)

    heads = {
        level: head_factory(n_class=taxonomy.num_classes(level),
                            embedding_dim=embedding_dim,
                            target_key=f"label_{level}")
        for level in taxonomy.levels
    }
    primary = primary_level if primary_level is not None else taxonomy.levels[-1]
    return MultiTaskHead(heads=heads, primary=primary, weights=weights)
