"""Прототипная голова (ProtoPNet-style, «this looks like that») — эксперимент (§3.1).

k обучаемых прототипов на класс в пространстве эмбеддингов; косинусная
similarity (устойчива к масштабу фичей замороженного бэкбона). Структурный
регуляризатор: каждый класс (в т.ч. хвостовой) получает выделенную ёмкость
из прототипов, а не долю в глобальной линейной границе.

Лоссы (ProtoPNet): CE + clustering (к своему прототипу) + separation (от чужих)
+ L1 (разреженность off-class связей).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.heads.base import ClassifierHead
from vision_lab.heads.primitives import valid_rows


class ProtoHead(ClassifierHead):
    def __init__(self, n_class: int, embedding_dim: int, k: int = 10, l_clst: float = 0.5,
                 l_sep: float = 0.5, l_l1: float = 1e-4, scale: float = 10.0,
                 target_key: str = "label"):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.k = k
        self.m = k * n_class
        self.l_clst, self.l_sep, self.l_l1, self.scale = l_clst, l_sep, l_l1, scale
        self.target_key = target_key

        self.prototypes = nn.Parameter(torch.empty(self.m, embedding_dim))
        nn.init.normal_(self.prototypes, std=0.02)

        proto_class = torch.arange(n_class).repeat_interleave(k)
        self.register_buffer("proto_class", proto_class)
        onehot = torch.zeros(n_class, self.m)
        onehot[proto_class, torch.arange(self.m)] = 1.0
        self.register_buffer("proto_onehot", onehot)

        self.fc = nn.Linear(self.m, n_class, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(onehot)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def _sim(self, embeddings):
        e = F.normalize(embeddings, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        return e @ p.t()

    def predict_logits(self, embeddings):
        return self.scale * self.fc(self._sim(embeddings))

    def forward(self, embeddings, targets: Mapping[str, torch.Tensor]):
        labels = self.take_target(targets)
        mask = valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        emb, y = embeddings[mask], labels[mask]
        sim = self._sim(emb)
        logits = self.scale * self.fc(sim)
        ce = F.cross_entropy(logits, y)

        own = self.proto_class.unsqueeze(0) == y.unsqueeze(1)
        max_own = sim.masked_fill(~own, -1e4).max(1).values
        max_oth = sim.masked_fill(own, -1e4).max(1).values
        l_clst = (1.0 - max_own).mean()
        l_sep = max_oth.clamp_min(0.0).mean()
        l1 = (self.fc.weight * (1.0 - self.proto_onehot)).abs().sum()

        total = ce + self.l_clst * l_clst + self.l_sep * l_sep + self.l_l1 * l1
        return {"total_loss": total, "ce": ce.detach(),
                "clst": l_clst.detach(), "sep": l_sep.detach()}
