"""Сборка Sheaf-ADMM: feature-map backbone'а -> координация агентов -> логиты.

``SheafADMMModule`` владеет базовыми restriction maps, энкодером параметров цели,
декодером и обучаемым penalty rho; строит геометрию пучка (fixed или LoRA-context)
и прогоняет разворачиваемый ADMM-цикл. Вход — карта признаков ``[B, C, H', W']``;
выход — оконные per-agent логиты ``[W, B, N, C]`` и агрегированные логиты ``[B, C]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    AgentDecoder,
    AgentEncoder,
    EdgeGeometry,
    GridGraph,
    run_admm,
)


class SheafADMMModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_v: int = 32,
        d_e: int = 16,
        enc_hidden: int = 256,
        dec_hidden: int = 128,
        dec_linear: bool = True,
        connectivity: int = 8,           # 4 | 8
        rm_mode: str = "context",        # fixed | context (LoRA)
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_use_gate: bool = False,
        objective_l1: float = 0.01,      # скалярный вес ℓ1 (lasso)
        q_epsilon: float = 1e-4,
        # z-update
        z_mode: str = "prox",            # prox (мягкий, хорошо обусловлен) | project (жёсткий)
        gamma: float = 1.0,
        cg_iters: int = 5,
        tikhonov_eps: float = 1e-5,
        rho_init: float = 0.5,
        # ADMM horizon
        num_iters: int = 8,
        eval_iters: int | None = None,
        loss_window: int = 2,
        grad_window: int | None = None,
        z_init: str = "h",              # h | zeros
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_v, self.d_e = d_v, d_e
        self.connectivity = connectivity
        self.num_directions = connectivity
        self.rm_mode = rm_mode
        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank
        self.objective_l1 = float(objective_l1)
        self.z_mode, self.gamma = z_mode, gamma
        self.cg_iters, self.tikhonov_eps = cg_iters, tikhonov_eps
        self.num_iters = num_iters
        self.eval_iters = eval_iters or num_iters
        self.loss_window = loss_window
        self.grad_window = grad_window
        self.z_init_mode = z_init

        # базовые restriction maps по направлению: [K, d_e, d_v], ортонормальные строки
        base = torch.empty(self.num_directions, d_e, d_v)
        for k in range(self.num_directions):
            nn.init.orthogonal_(base[k])
        self.base_maps = nn.Parameter(base)

        self.encoder = AgentEncoder(
            in_dim=in_channels, hidden=enc_hidden, d_v=d_v, d_e=d_e,
            num_directions=self.num_directions, rank=lora_rank,
            rm_mode=rm_mode, use_gate=lora_use_gate, q_epsilon=q_epsilon,
        )
        self.decoder = AgentDecoder(d_v, num_classes, hidden=dec_hidden, linear=dec_linear)

        # learned penalty rho > 0 через softplus-сдвиг (инициализация точная)
        inv = rho_init if rho_init > 20 else math.log(math.expm1(max(rho_init, 1e-6)))
        self.rho_raw = nn.Parameter(torch.tensor(float(inv)))

        self._graph_cache: dict = {}

    @property
    def rho(self):
        return F.softplus(self.rho_raw)

    def _graph(self, H, W, device) -> GridGraph:
        key = (H, W, device)
        g = self._graph_cache.get(key)
        if g is None:
            g = GridGraph(H, W, self.connectivity, device)
            self._graph_cache[key] = g
        return g

    def _build_geometry(self, graph: GridGraph, enc, B) -> EdgeGeometry:
        R_uv = self.base_maps[graph.dir_uv]  # [E, d_e, d_v]
        R_vu = self.base_maps[graph.dir_vu]
        if self.rm_mode == "fixed":
            return EdgeGeometry(graph, R_uv, R_vu)
        # context: gather LoRA-факторы по рёбрам/направлению
        A = enc["A"].view(B, graph.N, self.num_directions, self.d_e, self.lora_rank)
        Bf = enc["B"].view(B, graph.N, self.num_directions, self.d_v, self.lora_rank)
        u, v, d_uv, d_vu = graph.u, graph.v, graph.dir_uv, graph.dir_vu
        A_u = A[:, u, d_uv]   # [B, E, d_e, r]
        A_v = A[:, v, d_vu]
        B_u = Bf[:, u, d_uv]  # [B, E, d_v, r]
        B_v = Bf[:, v, d_vu]
        gate_u = gate_v = None
        if "gate" in enc:
            gate = enc["gate"].view(B, graph.N, self.num_directions)
            gate_u = gate[:, u, d_uv]  # [B, E]
            gate_v = gate[:, v, d_vu]
        return EdgeGeometry(graph, R_uv, R_vu, scale=self.lora_scale,
                            A_u=A_u, A_v=A_v, B_u=B_u, B_v=B_v,
                            gate_u=gate_u, gate_v=gate_v)

    def _encode(self, feat_map):
        """feat_map [B,C,H,W] -> enc-dict с полями [B,N,...] и graph."""
        B, C, H, W = feat_map.shape
        graph = self._graph(H, W, feat_map.device)
        N = graph.N
        # [B,C,H,W] -> [B,N,C] (row-major, совпадает с nid = r*W + c)
        tokens = feat_map.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
        enc_flat = self.encoder(tokens.reshape(B * N, C))
        enc = {}
        for k, val in enc_flat.items():
            enc[k] = val.view(B, N, *val.shape[1:])
        enc["l1_weight"] = self.objective_l1
        return enc, graph, B, N

    def forward(self, feat_map, *, num_iters=None):
        """Возвращает (logits_window [W,B,N,C], agg_logits [B,C]).

        Координация (энкодер/ADMM/CG/декодер) гоняется в fp32 с выключенным autocast:
        разворачиваемый итеративный солвер численно чувствителен к bf16. Backbone при этом
        может работать в bf16 — сюда приходит его выход, приводимый к float32.
        """
        with torch.autocast(device_type=feat_map.device.type, enabled=False):
            feat_map = feat_map.float()
            return self._forward_fp32(feat_map, num_iters)

    def _forward_fp32(self, feat_map, num_iters):
        enc, graph, B, N = self._encode(feat_map)
        geometry = self._build_geometry(graph, enc, B)
        rho = self.rho
        z0 = enc["h"] if self.z_init_mode == "h" else torch.zeros_like(enc["h"])
        K = num_iters or (self.num_iters if self.training else self.eval_iters)
        x_window = run_admm(
            enc, geometry, rho, z0, K,
            z_mode=self.z_mode, gamma=self.gamma, cg_iters=self.cg_iters,
            tikhonov_eps=self.tikhonov_eps, loss_window=self.loss_window,
            grad_window=self.grad_window,
        )  # [W,B,N,d_v]
        Wn = x_window.shape[0]
        logits = self.decoder(x_window.reshape(Wn * B * N, self.d_v))
        logits = logits.view(Wn, B, N, self.num_classes)
        agg = self._aggregate(logits[-1])  # по последней итерации
        return logits, agg

    @staticmethod
    def _aggregate(logits_last):
        """Глобальные логиты = log(mean_agents softmax(per-agent logits)). [B,N,C]->[B,C]."""
        p = torch.softmax(logits_last, dim=-1).mean(dim=1)  # [B,C]
        return torch.log(p.clamp_min(1e-9))
