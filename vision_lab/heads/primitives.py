"""Чистые функции для голов (ТЗ §4.2): margin/cosine/long-tail примитивы.

Головы остаются плоскими классами (наследуют только :class:`ClassifierHead`),
композируя эти функции — без промежуточных базовых классов. Причина: DBM
(инстанс-сложность) и Seesaw (накопительный буфер) сразу потребовали бы
escape-hatch у общего базового класса margin-голов.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def valid_rows(target: torch.Tensor) -> torch.Tensor:
    """Булева маска строк с валидной меткой (``target >= 0``)."""
    return target >= 0


def counts_to_tensor(class_counts) -> torch.Tensor:
    """Частоты классов → float-тензор с клампом >= 1 (защита от деления на 0)."""
    return torch.as_tensor(list(class_counts), dtype=torch.float).clamp_min(1.0)


def cosine_logits(embeddings: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Косинус между нормированными эмбеддингами и центрами: ``(B, C)`` в [-1, 1]."""
    return F.linear(F.normalize(embeddings), F.normalize(weight))


def additive_angular_margin(cosine: torch.Tensor, target: torch.Tensor, m: float) -> torch.Tensor:
    """ArcFace/AAM: на истинный класс cos(θ+m) с монотонной защитой (Wang 2018).

    Возвращает логиты БЕЗ масштаба ``s`` (умножение на s — снаружи).
    """
    sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
    cos_m, sin_m = math.cos(m), math.sin(m)
    phi = cosine * cos_m - sine * sin_m
    # монотонность: если θ+m > π, тянем линейно (иначе градиент разворачивается)
    th, mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
    phi = torch.where(cosine - th > 0, phi, cosine - mm)
    one_hot = F.one_hot(target, cosine.size(1)).to(cosine.dtype)
    return one_hot * phi + (1.0 - one_hot) * cosine


def subcenter_reduce(cosine_flat: torch.Tensor, n_class: int, k: int) -> torch.Tensor:
    """Sub-center ArcFace: max по k суб-центрам класса. Вход ``(B, C*k)`` → ``(B, C)``."""
    return cosine_flat.view(-1, n_class, k).max(dim=2)[0]


def ldam_margins(class_counts, max_m: float = 0.5) -> torch.Tensor:
    """LDAM-маржины (Cao 2019): ∝ n_c^{-1/4}, нормированы так, что макс = max_m."""
    counts = counts_to_tensor(class_counts)
    margins = 1.0 / torch.sqrt(torch.sqrt(counts))
    return margins * (max_m / margins.max())


def subtract_class_margin(cosine: torch.Tensor, target: torch.Tensor,
                          margins: torch.Tensor) -> torch.Tensor:
    """Вычитает класс-зависимый маржин из cos истинного класса (LDAM/DBM)."""
    onehot = F.one_hot(target, cosine.size(1)).bool()
    return torch.where(onehot, cosine - margins.unsqueeze(0), cosine)


def class_balanced_weights(class_counts, beta: float) -> torch.Tensor:
    """Class-Balanced веса (Cui 2019): (1-β)/(1-β^n_c), нормированы к среднему 1."""
    counts = counts_to_tensor(class_counts)
    eff = 1.0 - torch.pow(beta, counts)
    w = (1.0 - beta) / eff.clamp_min(1e-8)
    return w / w.mean()


def inverse_freq_weights(class_counts) -> torch.Tensor:
    """Обратная частота (аналог DRW без расписания), нормирована к среднему 1."""
    counts = counts_to_tensor(class_counts)
    w = counts.sum() / counts
    return w / w.mean()


def logit_prior(class_counts, tau: float = 1.0) -> torch.Tensor:
    """τ·log(приор класса) — сдвиг для Balanced Softmax / Logit Adjustment."""
    counts = counts_to_tensor(class_counts)
    return tau * (counts / counts.sum()).log()


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                         weight: torch.Tensor | None = None,
                         label_smoothing: float = 0.0) -> torch.Tensor:
    """CE только по строкам с валидной меткой (маскирование -1, ТЗ §5.3/multi-task).

    Если валидных строк нет — 0 (сохраняет граф, не роняет DDP).
    """
    mask = valid_rows(target)
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask], weight=weight,
                           label_smoothing=label_smoothing)


def masked_bce_with_logits(logits: torch.Tensor, target_multihot: torch.Tensor,
                           pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE по мульти-хот таргету ``(B, C)`` из {0, 1, -1} с ПОэлементным маскированием.

    ``-1`` означает «этот класс у сэмпла не размечен» (частичная разметка) —
    элемент не даёт ни лосса, ни градиента. Нет валидных элементов — 0
    (сохраняет граф, не роняет DDP).
    """
    mask = target_multihot >= 0
    if not mask.any():
        return logits.sum() * 0.0
    tgt = target_multihot.clamp_min(0).to(logits.dtype)
    pw = pos_weight.to(logits.dtype) if pos_weight is not None else None
    loss = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction="none")
    return loss[mask].mean()


def asymmetric_loss_with_logits(logits: torch.Tensor, target_multihot: torch.Tensor,
                                gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                                clip: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """Asymmetric Loss (Ridnik 2021) для multi-label; маскирует элементы с ``-1``.

    ``L+ = (1-p)^γ+·log p``; ``L- = p_m^γ-·log(1-p_m)``, где ``p_m = max(p-clip, 0)``
    (probability shifting: лёгкие негативы обнуляются). ``γ+=γ-=γ, clip=0`` —
    focal-BCE; ``γ=0, clip=0`` — обычный BCE. Фокусирующий вес участвует в
    градиенте (полноградиентный вариант, без detach оригинальной репы).
    """
    mask = target_multihot >= 0
    if not mask.any():
        return logits.sum() * 0.0
    y = target_multihot.clamp_min(0).to(logits.dtype)
    p = torch.sigmoid(logits)
    p_m = (p - clip).clamp(0.0, 1.0) if clip > 0 else p
    loss_pos = (1.0 - p).pow(gamma_pos) * torch.log(p.clamp_min(eps))
    loss_neg = p_m.pow(gamma_neg) * torch.log((1.0 - p_m).clamp_min(eps))
    loss = y * loss_pos + (1.0 - y) * loss_neg
    return -(loss[mask]).mean()


def has_positive_pairs(labels: torch.Tensor) -> bool:
    """Есть ли в батче хотя бы один класс с >= 2 валидными примерами (для metric-лоссов)."""
    valid = labels[valid_rows(labels)]
    if valid.numel() < 2:
        return False
    _, counts = valid.unique(return_counts=True)
    return bool((counts >= 2).any())
