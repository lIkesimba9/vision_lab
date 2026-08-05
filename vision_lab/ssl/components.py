"""Строительные блоки SSL: проекционные/предсказательные головы, DINO-голова,
KoLeo, Sinkhorn-Knopp (порт из прототипа) и MIM-примитивы (block_mask, patchify)."""

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


def block_mask(x: torch.Tensor, grid: tuple[int, int],
               mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Маска входа блоками, выровненными по сетке токенов (iBOT/SimMIM, §5.2).

    Возвращает ``(x_masked, mask_flat)``: вход с занулёнными блоками и булеву
    маску ``(B, gh*gw)`` (True = замаскирован). Гарантия «не всё»: хотя бы один
    токен каждого сэмпла остаётся видимым.
    """
    b, _, h, w = x.shape
    gh, gw = grid
    m = torch.rand(b, gh, gw, device=x.device) < mask_ratio
    m_flat = m.reshape(b, gh * gw)
    m_flat[m_flat.all(dim=1), 0] = False  # не маскируем всё
    m = m_flat.reshape(b, gh, gw)
    up = m.float().repeat_interleave(max(h // gh, 1), 1).repeat_interleave(max(w // gw, 1), 2)
    x_masked = x * (1.0 - up.unsqueeze(1))
    return x_masked, m_flat


def patchify(x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """``(B, C, H, W)`` → ``(B, gh*gw, ph*pw*C)`` — пиксели патчей по сетке токенов.

    Таргет реконструкции MAE/SimMIM; ``ph = H // gh`` (для Swin/CNN патч —
    рецептивная клетка выходного токена).
    """
    b, c, h, w = x.shape
    gh, gw = grid
    ph, pw = h // gh, w // gw
    if gh * ph != h or gw * pw != w:
        raise ValueError(f"Размер входа {(h, w)} не делится на сетку токенов {grid}")
    x = x.reshape(b, c, gh, ph, gw, pw)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, ph * pw * c)


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
