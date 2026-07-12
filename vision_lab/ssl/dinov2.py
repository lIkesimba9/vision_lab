"""DINOv2-стек: DINO + iBOT + KoLeo + Sinkhorn-Knopp (порт из прототипа, §5.2).

Компоненты (https://arxiv.org/abs/2304.07193):
  * DINO     — multi-crop self-distillation, teacher sharpening + centering;
  * iBOT     — masked image modeling: student видит block-masked глобальный кроп
               и матчит patch-токены учителя в замаскированных позициях;
  * KoLeo    — spread эмбеддингов на сфере (−log расстояние до соседа);
  * Sinkhorn — Sinkhorn–Knopp центрирование учителя (дефолт) vs EMA.

iBOT выравнивает маски по ВЫХОДНОЙ сетке токенов. Для ViT она равна patch-сетке
(настоящий iBOT); для Swin выход даунсемплится, поэтому маскируем ВХОД блоками,
выровненными по финальной сетке (SimMIM-style). ``ibot_weight=0`` → чистый
DINO+KoLeo (рекомендуется для Swin).

Расписания (EMA-tau, teacher-temp) — атрибуты ``current_tau``/``teacher_temp``,
их пишет ScheduleDriver; ``momentum_update`` зовёт трейнер. Центры (EMA/Sinkhorn)
и веса учителя — буферы/submodule (в state_dict → resume бесплатно).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.models.backbones import TokenBackbone
from vision_lab.ssl.base import MomentumTeacher, SSLMethod
from vision_lab.ssl.components import DINOHead, koleo_loss, sinkhorn_knopp


class DINOv2(SSLMethod):
    """DINOv2 SSL-метод. ``backbone`` — :class:`TokenBackbone`; ``views`` —
    :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment` (dino_v1: 2 глобальных +
    n локальных).
    """

    def __init__(self, backbone: TokenBackbone, views: nn.Module,
                 out_dim: int = 65536, student_temp: float = 0.1,
                 center_momentum: float = 0.9, center_mode: str = "sinkhorn",
                 koleo_weight: float = 0.1, ibot_weight: float = 1.0,
                 mask_ratio: float = 0.3, head_hidden_dim: int = 2048,
                 head_bottleneck_dim: int = 256, norm_last_layer: bool = True):
        super().__init__()
        self.views = views
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_mode = center_mode
        self.koleo_weight = koleo_weight
        self.ibot_weight = ibot_weight
        self.mask_ratio = mask_ratio

        self.student = backbone
        feat = backbone.out_dim

        def make_head():
            return DINOHead(feat, out_dim, hidden_dim=head_hidden_dim,
                            bottleneck_dim=head_bottleneck_dim, norm_last_layer=norm_last_layer)

        self.student_head = make_head()
        # TokenBackbone deepcopy'ится; DINOHead с weight_norm — нет (нужна factory).
        self.teacher = MomentumTeacher(self.student)
        self.teacher_head = MomentumTeacher(self.student_head, factory=make_head)

        if ibot_weight > 0:
            self.student_ibot_head = make_head()
            self.teacher_ibot_head = MomentumTeacher(self.student_ibot_head, factory=make_head)
        else:
            self.student_ibot_head = None
            self.teacher_ibot_head = None

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_ibot", torch.zeros(1, out_dim))

    def momentum_update(self) -> None:
        tau = float(self.current_tau)
        self.teacher.update(self.student, tau)
        self.teacher_head.update(self.student_head, tau)
        if self.teacher_ibot_head is not None:
            self.teacher_ibot_head.update(self.student_ibot_head, tau)

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled-вектор teacher-бэкбона (EMA лучше student для probe)."""
        return self.teacher.module.embed(images)

    def _block_mask(self, x: torch.Tensor, grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """SimMIM-style маска входа, выровненная по выходной сетке токенов.

        Гарантия «не всё и не ничего»: хотя бы один токен всегда виден.
        """
        b, _, h, w = x.shape
        gh, gw = grid
        m = torch.rand(b, gh, gw, device=x.device) < self.mask_ratio
        m_flat = m.reshape(b, gh * gw)
        m_flat[m_flat.all(dim=1), 0] = False  # не маскируем всё
        up = m.float().repeat_interleave(max(h // gh, 1), 1).repeat_interleave(max(w // gw, 1), 2)
        x_masked = x * (1.0 - up.unsqueeze(1))
        return x_masked, m_flat

    def _teacher_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.center_mode == "sinkhorn":
            return sinkhorn_knopp(logits)
        return F.softmax((logits - self.center) / float(self.teacher_temp), dim=-1)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x = batch["image"]
        view_set = self.views(x)
        if len(view_set.globals) < 2:
            raise ValueError("DINOv2 требует >= 2 глобальных вьюхи")
        g1, g2 = view_set.globals[0], view_set.globals[1]
        locals_ = view_set.locals
        b = x.shape[0]

        # teacher: 2 глобальных кропа (no grad)
        with torch.no_grad():
            t_out1, t_out2 = self.teacher(g1), self.teacher(g2)
            t_logits = self.teacher_head(torch.cat([t_out1.pooled, t_out2.pooled]))
            t_probs = self._teacher_probs(t_logits).detach()
            tp1, tp2 = t_probs.chunk(2)
            if self.ibot_weight > 0:
                t_itok = torch.cat([t_out1.tokens, t_out2.tokens])
                t_ilogits = self.teacher_ibot_head(t_itok)
                t_iprobs = F.softmax(
                    (t_ilogits - self.center_ibot) / float(self.teacher_temp), dim=-1).detach()

        # student: маскированные глобальные + локальные
        if self.ibot_weight > 0:
            grid = self.student.grid_size((g1.shape[-2], g1.shape[-1]))
            mg1, mask1 = self._block_mask(g1, grid)
            mg2, mask2 = self._block_mask(g2, grid)
        else:
            mg1, mg2 = g1, g2

        s_logits, s_tokens = [], []
        for view, want_tok in [(mg1, True), (mg2, True)] + [(c, False) for c in locals_]:
            s_out = self.student(view)
            s_logits.append(self.student_head(s_out.pooled))
            if want_tok:
                s_tokens.append(s_out.tokens)
        sp = [F.log_softmax(s / self.student_temp, dim=-1) for s in s_logits]

        # DINO: каждый student-кроп vs каждый teacher-глобал, кроме той же вьюхи
        dino_terms, n_terms = 0.0, 0
        for ti, tprob in enumerate([tp1, tp2]):
            for si, slog in enumerate(sp):
                if si == ti:
                    continue
                dino_terms = dino_terms - (tprob * slog).sum(dim=-1).mean()
                n_terms += 1
        out = {"dino_loss": dino_terms / max(n_terms, 1)}

        # iBOT: student masked-patch логиты vs teacher patch-probs в masked-позициях
        if self.ibot_weight > 0:
            ibot = 0.0
            for stok, tprob, mask in [(s_tokens[0], t_iprobs[:b], mask1),
                                      (s_tokens[1], t_iprobs[b:], mask2)]:
                s_il = F.log_softmax(self.student_ibot_head(stok) / self.student_temp, dim=-1)
                ce = -(tprob * s_il).sum(dim=-1)
                m = mask.float()
                ibot = ibot + (ce * m).sum() / m.sum().clamp(min=1.0)
            out["ibot_loss"] = ibot / 2.0

        # KoLeo на student global-эмбеддингах
        if self.koleo_weight > 0:
            g_emb = self.student.embed(torch.cat([mg1, mg2]))
            out["koleo_loss"] = koleo_loss(g_emb)

        total = out["dino_loss"]
        if "ibot_loss" in out:
            total = total + self.ibot_weight * out["ibot_loss"]
        if "koleo_loss" in out:
            total = total + self.koleo_weight * out["koleo_loss"]
        out["total_loss"] = total

        # EMA-центры (для center_mode='ema')
        if self.center_mode == "ema":
            self._update_center(t_logits, t_ilogits if self.ibot_weight > 0 else None)
        return out

    @torch.no_grad()
    def _update_center(self, t_logits: torch.Tensor, t_ilogits: torch.Tensor | None) -> None:
        b = self.center_momentum
        self.center.mul_(b).add_(t_logits.mean(dim=0, keepdim=True), alpha=1 - b)
        if t_ilogits is not None:
            flat = t_ilogits.reshape(-1, self.out_dim).mean(dim=0, keepdim=True)
            self.center_ibot.mul_(b).add_(flat, alpha=1 - b)
