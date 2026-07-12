"""Строительные блоки SSL: проекционные/предсказательные головы, DINO-голова,
KoLeo и Sinkhorn-Knopp (порт из прототипа)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Проекционная/предсказательная голова BYOL: Linear-BN-ReLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOHead(nn.Module):
    """3-слойный MLP (GELU) → L2-norm → weight-norm линейный слой на out_dim прототипов."""

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, nlayers: int = 3, norm_last_layer: bool = True):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def koleo_loss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """−mean log(расстояние до ближайшего соседа) на L2-норм. фичах (spread-регуляризатор)."""
    x = F.normalize(x, dim=-1, p=2)
    sim = x @ x.t()
    sim.fill_diagonal_(-2.0)
    nn_sim = sim.max(dim=1).values
    dist = torch.clamp(2.0 - 2.0 * nn_sim, min=eps)
    return -torch.log(dist).mean()


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn–Knopp нормализация логитов учителя → дважды-нормализованные назначения."""
    Q = torch.exp(logits.float()).t()  # (K, B)
    Q /= Q.sum() + eps
    K_, B = Q.shape
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True) + eps
        Q /= K_
        Q /= Q.sum(dim=0, keepdim=True) + eps
        Q /= B
    Q *= B
    return Q.t()
