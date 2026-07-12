"""DDP-утилиты: rank/world, дифференцируемый all_gather, DDP-mixin для batch-сэмплеров.

Инварианты DDP (ТЗ §11.1):

* кастомные сэмплеры сами шардят по рангам, трейнер выставляет
  ``use_distributed_sampler=False``;
* batch-зависимые лоссы (SimCLR/SupCon/KoLeo) зовут :func:`all_gather_grad`
  ЯВНО внутри своего forward — никакого неявного gather в трейнере
  (BYOL/DINO и positive-shuffle обязаны оставаться per-rank).
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def dist_info() -> tuple[int, int]:
    """(world_size, rank) — безопасно и до инициализации DDP (тогда (1, 0))."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def world_size() -> int:
    return dist_info()[0]


def rank() -> int:
    return dist_info()[1]


def is_main() -> bool:
    return rank() == 0


def all_gather_grad(x: torch.Tensor) -> torch.Tensor:
    """Дифференцируемый all_gather: ``(B, ...) -> (world*B, ...)``.

    Градиент течёт обратно в ЛОКАЛЬНЫЙ шард. При world_size==1 — identity,
    поэтому код contrastive-лоссов одинаков на одном GPU и в DDP, без веток.
    """
    if world_size() == 1:
        return x
    from torch.distributed import nn as dist_nn

    return torch.cat(list(dist_nn.functional.all_gather(x)), dim=0)


class DistributedBatchSamplerMixin:
    """Превращает любой batch-sampler в DDP-безопасный (порт из прототипа).

    Дочерний класс реализует:
        ``_build_epoch_batches() -> list[list[int]]``  — детерминирован по ``self._epoch_seed()``;
        ``_num_batches_total() -> int``                — число батчей ДО шардинга (стабильно).

    Гарантии:

    * каждый ранг строит ОДИН И ТОТ ЖЕ полный список батчей (один сид на
      эпоху) и берёт непересекающийся срез ``batches[rank::world]`` — нет
      дублей между GPU;
    * число батчей одинаково на всех рангах (остаток обрезается) — нет
      дедлока DDP;
    * reshuffle между эпохами через ``set_epoch`` (Lightning зовёт его сам).
    """

    seed: int = 42

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._epoch_set_externally = True

    def _epoch_seed(self) -> int:
        return self.seed + getattr(self, "_epoch", 0)

    def _sharded_batches(self) -> list[list[int]]:
        batches = self._build_epoch_batches()
        world, rank_ = dist_info()
        if world > 1:
            usable = (len(batches) // world) * world
            batches = batches[rank_:usable:world]
        # одиночный GPU: продвигаем эпоху сами, чтобы был reshuffle, если set_epoch не звали
        if world == 1 and not hasattr(self, "_epoch_set_externally"):
            self._epoch = getattr(self, "_epoch", -1) + 1
        return batches

    def __iter__(self):
        yield from self._sharded_batches()

    def __len__(self) -> int:
        world, _ = dist_info()
        n = self._num_batches_total()
        return n // world if world > 1 else n


__all__ = [
    "dist_info",
    "world_size",
    "rank",
    "is_main",
    "all_gather_grad",
    "DistributedBatchSamplerMixin",
]
