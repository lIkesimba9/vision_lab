"""Flip-TTA: усреднение логитов по симметриям (ТЗ §5.5).

Фикс TTA или нормализации в одном месте чинит весь инференс (урок прототипа:
логика была размазана по ~20 скриптам).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Обратимые пиксельные симметрии (identity + флипы + поворот на 180°).
TTA_VIEWS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "hflip": lambda x: torch.flip(x, dims=[-1]),
    "vflip": lambda x: torch.flip(x, dims=[-2]),
    "rot180": lambda x: torch.flip(x, dims=[-2, -1]),
}


def tta_logits(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    views: tuple[str, ...] = ("identity", "hflip"),
) -> torch.Tensor:
    """Усредняет логиты по TTA-вьюхам.

    ``logits_fn`` — функция image -> логиты (B, C). Симметрии применимы, потому
    что классификация инвариантна к флипам/повороту (для дерматоскопии и т.п.).
    """
    unknown = set(views) - set(TTA_VIEWS)
    if unknown:
        raise KeyError(f"Неизвестные TTA-вьюхи: {unknown}. Доступны: {sorted(TTA_VIEWS)}")
    acc = None
    for name in views:
        out = logits_fn(TTA_VIEWS[name](images))
        acc = out if acc is None else acc + out
    return acc / len(views)
