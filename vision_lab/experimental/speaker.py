"""Speaker-verification метрик-лоссы (GE2E / Angular Prototypical) + голова.

Вход лоссов (N, M, D): N классов, M примеров на класс. PK-сэмплер даёт ровно
такую структуру. Экспериментально: перенос из прототипа как есть (§3.1).

Источники: GE2E — arXiv:1710.10467; AngleProto — arXiv:2003.11982.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.heads.base import ClassifierHead
from vision_lab.heads.classification import AAMHead


class AngleProtoLoss(nn.Module):
    """Angular Prototypical (векторизованный). x: (N, M, D), M>=2."""

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, _label=None) -> torch.Tensor:
        assert x.size(1) >= 2, "AngleProto требует M>=2"
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        n = out_anchor.size(0)
        cos = F.cosine_similarity(
            out_positive.unsqueeze(-1).expand(-1, -1, n),
            out_anchor.unsqueeze(-1).expand(-1, -1, n).transpose(0, 2),
        )
        self.w.data.clamp_(1e-6)
        cos = cos * self.w + self.b
        return self.criterion(cos, torch.arange(n, device=cos.device))


class SpeakerHead(ClassifierHead):
    """AAM-классификатор (логиты/инференс) + AngleProto SV-член на PK-батче."""

    def __init__(self, n_class: int, embedding_dim: int, sv_weight: float = 0.5,
                 m: float = 0.2, s: float = 30.0, target_key: str = "label",
                 fc_weight_path: str | None = None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.target_key = target_key
        self.aam = AAMHead(n_class, embedding_dim, m, s, target_key=target_key,
                           fc_weight_path=fc_weight_path)
        self.sv = AngleProtoLoss()
        self.sv_weight = sv_weight

    @property
    def classifier_weight(self):
        return self.aam.weight

    def forward(self, embeddings, targets: Mapping[str, torch.Tensor]):
        labels = self.take_target(targets)
        aam_loss = self.aam(embeddings, targets)["total_loss"]
        sv_loss = embeddings.new_tensor(0.0)
        uniq, counts = labels.unique(return_counts=True)
        valid = uniq[(counts >= 2) & (uniq >= 0)]
        if valid.numel() >= 2:
            m = int(counts[(counts >= 2) & (uniq >= 0)].min().item())
            groups = [embeddings[(labels == c).nonzero(as_tuple=True)[0][:m]] for c in valid.tolist()]
            sv = self.sv(torch.stack(groups))
            if torch.isfinite(sv):
                sv_loss = sv
        total = aam_loss + self.sv_weight * sv_loss
        return {"total_loss": total, "aam_loss": aam_loss, "sv_loss": sv_loss}

    def predict_logits(self, embeddings):
        return self.aam.predict_logits(embeddings)
