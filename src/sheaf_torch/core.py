"""Sheaf-ADMM — самодостаточный PyTorch-порт координационного ядра.

Порт метода из статьи «Learning Multi-Agent Coordination via Sheaf-ADMM»
(Seely, Cupiał, Jones, ICML 2026, arXiv:2605.31005). Оригинал — JAX/Flax
(см. ../../sheaf_admm/upstream). Здесь адаптация к нашему PyTorch-пайплайну:
агенты живут на сетке признаков backbone'а (H'xW'), координируются через
разворачиваемый ADMM с обучаемым клеточным пучком и дают глобальную классификацию
изображения (прототип — MNIST-классификация из статьи).

Раскладка тензоров: состояния агентов — ``[B, N, d_v]`` (batch-first; в оригинале
``[N, B, d_v]``). Граф рёбер общий для батча. Всё батчировано и дифференцируемо.

Компоненты (см. соответствие модулям оригинала):
  * граф-сетка + направления рёбер           (data/views.py, geometry/restriction_maps.py)
  * restriction maps (directional) + LoRA     (geometry/{restriction_maps,lora}.py)
  * sheaf-лапласиан L_F = F^T F (matvec)       (geometry/lora.py::laplacian_apply)
  * x-update: диагональный prox (lasso)        (solvers/x_solvers/diagonal_prox.py)
  * z-update: unrolled CG (project|prox)       (solvers/z_solvers/unrolled_cg.py)
  * ADMM-цикл с loss/grad window               (admm.py::run_admm)
  * encoder параметров цели + decoder          (models/{encoder,decoder}.py)
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CG_EPS = 1e-8


# =============================================================================
# Граф-сетка и направления рёбер
# =============================================================================
def _direction_index(dy: int, dx: int, num_directions: int) -> int:
    """(dy, dx) -> индекс направления. 4-way: 0=N,1=E,2=S,3=W; 8-way: N,NE,E,SE,S,SW,W,NW."""
    is_n, is_s = dy < 0, dy > 0
    is_e, is_w = dx > 0, dx < 0
    if num_directions == 4:
        if is_n:
            return 0
        if is_s:
            return 2
        return 1 if is_e else 3
    if is_n and is_e:
        return 1
    if is_n and is_w:
        return 7
    if is_n:
        return 0
    if is_s and is_e:
        return 3
    if is_s and is_w:
        return 5
    if is_s:
        return 4
    return 2 if is_e else 6


@lru_cache(maxsize=32)
def _build_grid_graph_np(H: int, W: int, connectivity: int):
    """Рёбра сетки HxW и их направления (кэшируется по (H,W,connectivity)).

    Возвращает (edge_u [E], edge_v [E], dir_uv [E], dir_vu [E]) как int64 np-массивы.
    Каждое неориентированное ребро добавляется один раз (u < v).
    """
    assert connectivity in (4, 8)
    num_dir = connectivity
    if connectivity == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def nid(r, c):
        return r * W + c

    us, vs, d_uv, d_vu = [], [], [], []
    seen = set()
    for r in range(H):
        for c in range(W):
            u = nid(r, c)
            for dy, dx in neigh:
                rr, cc = r + dy, c + dx
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                v = nid(rr, cc)
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
                us.append(u)
                vs.append(v)
                # direction u->v uses (dy,dx); v->u uses (-dy,-dx)
                d_uv.append(_direction_index(dy, dx, num_dir))
                d_vu.append(_direction_index(-dy, -dx, num_dir))
    return (
        np.asarray(us, dtype=np.int64),
        np.asarray(vs, dtype=np.int64),
        np.asarray(d_uv, dtype=np.int64),
        np.asarray(d_vu, dtype=np.int64),
    )


class GridGraph:
    """Держит тензоры рёбер сетки на нужном устройстве (u, v, dir_uv, dir_vu)."""

    def __init__(self, H, W, connectivity, device):
        u, v, d_uv, d_vu = _build_grid_graph_np(H, W, connectivity)
        self.H, self.W, self.N = H, W, H * W
        self.num_directions = connectivity
        self.u = torch.as_tensor(u, device=device)
        self.v = torch.as_tensor(v, device=device)
        self.dir_uv = torch.as_tensor(d_uv, device=device)
        self.dir_vu = torch.as_tensor(d_vu, device=device)
        self.E = self.u.shape[0]


# =============================================================================
# Геометрия пучка: эффективные restriction maps по рёбрам + L_F matvec
# =============================================================================
class EdgeGeometry:
    """Собранная per-edge геометрия для одного forward'а.

    R_uv/R_vu: ``[E, d_e, d_v]`` базовые карты, разложенные по рёбрам (по направлению).
    LoRA-факторы (если context): ``A_*`` ``[B,E,d_e,r]``, ``B_*`` ``[B,E,d_v,r]``,
    ``gate_*`` ``[B,E]`` — уже собраны для конкретного ребра/направления.
    """

    def __init__(self, graph: GridGraph, R_uv, R_vu, scale=0.0,
                 A_u=None, A_v=None, B_u=None, B_v=None, gate_u=None, gate_v=None):
        self.g = graph
        self.R_uv, self.R_vu = R_uv, R_vu
        self.scale = scale
        self.A_u, self.A_v = A_u, A_v
        self.B_u, self.B_v = B_u, B_v
        self.gate_u, self.gate_v = gate_u, gate_v

    def _apply(self, z_end, R, A_e, B_e, gate_e):
        # z_end: [B,E,d_v]; R: [E,d_e,d_v]
        Rz = torch.einsum("eij,bej->bei", R, z_end)
        if A_e is None:
            return Rz
        Btz = torch.einsum("bejr,bej->ber", B_e, z_end)
        ABtz = torch.einsum("beir,ber->bei", A_e, Btz)
        if gate_e is not None:
            ABtz = ABtz * gate_e[..., None]
        return Rz + self.scale * ABtz

    def _adjoint(self, r, R, A_e, B_e, gate_e):
        # r: [B,E,d_e]
        Rtr = torch.einsum("eij,bei->bej", R, r)
        if A_e is None:
            return Rtr
        Atr = torch.einsum("beir,bei->ber", A_e, r)
        if gate_e is not None:
            Atr = Atr * gate_e[..., None]
        lora = torch.einsum("bejr,ber->bej", B_e, Atr)
        return Rtr + self.scale * lora

    def edge_residuals(self, z):
        """Кограница F z: рассогласование на ребре ``F_u z_u - F_v z_v`` -> [B,E,d_e]."""
        zu = z[:, self.g.u]
        zv = z[:, self.g.v]
        Fz_u = self._apply(zu, self.R_uv, self.A_u, self.B_u, self.gate_u)
        Fz_v = self._apply(zv, self.R_vu, self.A_v, self.B_v, self.gate_v)
        return Fz_u - Fz_v

    def laplacian_apply(self, z):
        """Sheaf-лапласиан L_F z = F^T F z, [B,N,d_v] -> [B,N,d_v] (matrix-free)."""
        r = self.edge_residuals(z)  # [B,E,d_e]
        cu = self._adjoint(r, self.R_uv, self.A_u, self.B_u, self.gate_u)  # [B,E,d_v]
        cv = self._adjoint(r, self.R_vu, self.A_v, self.B_v, self.gate_v)
        out = torch.zeros_like(z)
        out = out.index_add(1, self.g.u, cu)
        out = out.index_add(1, self.g.v, -cv)
        return out

    def consistency_rms(self, z, eps=1e-6):
        r = self.edge_residuals(z)
        return torch.sqrt((r ** 2).mean(dim=(1, 2)) + eps)  # [B]


# =============================================================================
# Солверы ADMM
# =============================================================================
def soft_threshold(x, thr):
    return torch.sign(x) * torch.clamp(x.abs() - thr, min=0.0)


def x_update_diag_prox(z, y, rho, q_diag, q, l1_weight):
    """Замкнутый диагональный prox (lasso): x = soft_threshold((rho v - q)/a, l1/a).

    v = z - y (центр prox); a = q_diag + rho. Совпадает с diagonal_prox.py (l2=0, без box).
    """
    v = z - y
    a = q_diag + rho
    t = (rho * v - q) / a
    if isinstance(l1_weight, float) and l1_weight == 0.0:
        return t
    return soft_threshold(t, l1_weight / a)


def _batched_cg(matvec, b, x0, iters):
    """Батчированный CG для A x = b по состояниям [B,N,d_v]; скаляры на элемент батча."""

    def dot(a, c):
        return (a * c).sum(dim=(1, 2))  # [B]

    x = x0
    r = b - matvec(x)
    p = r
    rs = dot(r, r)
    for _ in range(iters):
        Ap = matvec(p)
        pAp = dot(p, Ap)
        alpha = rs / (pAp + _CG_EPS)
        x = x + alpha[:, None, None] * p
        r = r - alpha[:, None, None] * Ap
        rs_new = dot(r, r)
        beta = rs_new / (rs + _CG_EPS)
        p = r + beta[:, None, None] * p
        rs = rs_new
    return x


def z_update_cg(z_target, geometry: EdgeGeometry, rho, mode, gamma, cg_iters, tikhonov_eps):
    """Консенсус z-update через unrolled CG.

    project (жёсткий, Fz=0): (L+eps I) w = L z_target, z = z_target - w.
    prox    (мягкий):        (gamma L + rho I) z = rho z_target.
    """
    if mode == "project":
        def matvec(x):
            return geometry.laplacian_apply(x) + tikhonov_eps * x
        b = geometry.laplacian_apply(z_target)
        w = _batched_cg(matvec, b, torch.zeros_like(z_target), cg_iters)
        return z_target - w
    if mode == "prox":
        def matvec(x):
            return gamma * geometry.laplacian_apply(x) + rho * x
        b = rho * z_target
        return _batched_cg(matvec, b, z_target, cg_iters)
    raise ValueError(f"unknown z mode {mode!r} (project|prox)")


def run_admm(enc, geometry, rho, z_init, num_iters, *, z_mode, gamma,
             cg_iters, tikhonov_eps, loss_window, grad_window):
    """K итераций ADMM. Возвращает последние ``loss_window`` x-итераций [W,B,N,d_v]."""
    q_diag, q, l1 = enc["q_diag"], enc["q"], enc["l1_weight"]
    z = z_init
    y = torch.zeros_like(z_init)
    n_detached = 0 if grad_window is None else max(0, num_iters - grad_window)
    window = min(loss_window, num_iters)
    xs = []

    for k in range(num_iters):
        detach = k < n_detached
        ctx = torch.no_grad() if detach else _nullcontext()
        with ctx:
            x = x_update_diag_prox(z, y, rho, q_diag, q, l1)
            z_target = x + y
            z = z_update_cg(z_target, geometry, rho, z_mode, gamma, cg_iters, tikhonov_eps)
            y = y + (x - z)
        if detach:
            x, z, y = x.detach(), z.detach(), y.detach()
        if k >= num_iters - window:
            xs.append(x)
    return torch.stack(xs, dim=0)  # [W,B,N,d_v]


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# =============================================================================
# Encoder / Decoder агента (общие для всех агентов)
# =============================================================================
def _mlp(in_dim, hidden, norm=True):
    layers = [nn.Linear(in_dim, hidden)]
    if norm:
        layers.append(nn.LayerNorm(hidden))
    layers += [nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()]
    return nn.Sequential(*layers)


class AgentEncoder(nn.Module):
    """Локальный вид агента -> seed h + параметры выпуклой цели (+ LoRA-факторы).

    objective_mode='lasso': выдаёт h, q_diag (>0), q; l1_weight — скаляр из конфига.
    rm_mode='context': дополнительно LoRA A [K,d_e,r], B [K,d_v,r], gate [K].
    """

    def __init__(self, in_dim, hidden, d_v, d_e, num_directions, rank,
                 rm_mode="context", use_gate=False, q_epsilon=1e-4):
        super().__init__()
        self.d_v, self.d_e, self.K, self.r = d_v, d_e, num_directions, rank
        self.rm_mode = rm_mode
        self.use_gate = use_gate
        self.q_epsilon = q_epsilon
        self.trunk = _mlp(in_dim, hidden)
        self.h_head = nn.Linear(hidden, d_v)
        self.qdiag_head = nn.Linear(hidden, d_v)
        self.q_head = nn.Linear(hidden, d_v)
        if rm_mode == "context":
            self.A_head = nn.Linear(hidden, num_directions * d_e * rank)
            self.B_head = nn.Linear(hidden, num_directions * d_v * rank)
            nn.init.zeros_(self.B_head.weight)  # legacy LoRA init: dF = 0 в начале
            nn.init.zeros_(self.B_head.bias)
            if use_gate:
                self.gate_head = nn.Linear(hidden, num_directions)
                nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(self, feat):
        # feat: [M, in_dim] (M = B*N)
        h = self.trunk(feat)
        out = {
            "h": self.h_head(h),
            "q_diag": F.softplus(self.qdiag_head(h)) + self.q_epsilon,
            "q": self.q_head(h),
        }
        if self.rm_mode == "context":
            M = feat.shape[0]
            out["A"] = self.A_head(h).view(M, self.K, self.d_e, self.r)
            out["B"] = self.B_head(h).view(M, self.K, self.d_v, self.r)
            if self.use_gate:
                out["gate"] = torch.sigmoid(self.gate_head(h))
        return out


class AgentDecoder(nn.Module):
    """Финальное состояние агента x -> локальные логиты класса (readout x_only)."""

    def __init__(self, d_v, num_classes, hidden=None, linear=True):
        super().__init__()
        if linear or not hidden:
            self.net = nn.Linear(d_v, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_v, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, x):  # x: [M, d_v] -> [M, C]
        return self.net(x)
