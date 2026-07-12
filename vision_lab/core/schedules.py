"""Расписания как first-class объекты + коллбек, который их применяет.

Ключевое решение: каждое расписание — ЧИСТАЯ функция ``(step, total_steps) -> float``
(frozen dataclass). Никакого персистентного состояния: на resume значение
пересчитывается из ``trainer.global_step`` до первого forward — расписания
не могут «разъехаться» с чекпоинтом в принципе.

:class:`ScheduleDriver` пишет значения в ОБЫЧНЫЕ float-атрибуты модуля в
``on_train_batch_start``:

* значение выставлено ДО forward (teacher_temp потребляется внутри forward);
* ``global_step`` не меняется между микро-батчами grad accumulation — все
  микро-батчи одного optimizer step видят одно значение;
* float-атрибут вместо буфера — нет device/dtype-возни под bf16 и ``fill_()``.

Настоящее состояние (DINO center, Seesaw cum_samples) — буферы модулей, не здесь.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@runtime_checkable
class Schedule(Protocol):
    """Протокол расписания: значение в точке ``step`` из ``total_steps``."""

    def __call__(self, step: int, total_steps: int) -> float: ...


@dataclass(frozen=True)
class ConstantSchedule:
    value: float

    def __call__(self, step: int, total_steps: int) -> float:
        return self.value


@dataclass(frozen=True)
class LinearSchedule:
    start: float
    end: float

    def __call__(self, step: int, total_steps: int) -> float:
        t = min(max(step, 0), total_steps) / max(total_steps, 1)
        return self.start + (self.end - self.start) * t


@dataclass(frozen=True)
class CosineSchedule:
    """Косинус от ``start`` к ``end`` за весь ран (EMA-tau, weight decay ramp)."""

    start: float
    end: float

    def __call__(self, step: int, total_steps: int) -> float:
        t = min(max(step, 0), total_steps) / max(total_steps, 1)
        return self.end - (self.end - self.start) * (math.cos(math.pi * t) + 1) / 2


@dataclass(frozen=True)
class LinearWarmupConstant:
    """Линейный прогрев ``start -> value`` за ``warmup_steps``, дальше константа
    (teacher temperature в DINO)."""

    start: float
    value: float
    warmup_steps: int

    def __call__(self, step: int, total_steps: int) -> float:
        if step >= self.warmup_steps or self.warmup_steps <= 0:
            return self.value
        return self.start + (self.value - self.start) * (step / self.warmup_steps)


class ScheduleDriver(pl.Callback):
    """Применяет расписания к атрибутам модуля каждый шаг.

    ``schedules`` — маппинг «путь -> расписание». Формы пути:

    * ``"method.current_tau"`` — атрибут (вложенный) LightningModule;
    * ``"optimizer/backbone.decay.weight_decay"`` — ключ param-группы с
      ``name == "backbone.decay"`` (группы именует :func:`vision_lab.core.optim.param_groups`).

    Пути валидируются на старте fit — опечатка в YAML падает сразу, а не
    молча не применяется.
    """

    def __init__(self, schedules: Mapping[str, Schedule], log_values: bool = True):
        super().__init__()
        self.schedules = dict(schedules)
        self.log_values = log_values
        self._total = 1

    # -- вспомогательные ---------------------------------------------------
    @staticmethod
    def _resolve_attr(pl_module: pl.LightningModule, path: str):
        """(объект, имя_атрибута) для пути вида ``a.b.attr``."""
        obj_path, _, attr = path.rpartition(".")
        obj = pl_module
        if obj_path:
            for part in obj_path.split("."):
                obj = getattr(obj, part)
        return obj, attr

    @staticmethod
    def _find_param_group(trainer: pl.Trainer, group_name: str) -> dict:
        for opt in trainer.optimizers:
            for g in opt.param_groups:
                if g.get("name") == group_name:
                    return g
        raise KeyError(f"param-группа {group_name!r} не найдена в оптимизаторах")

    def _assign(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                path: str, value: float) -> None:
        if path.startswith("optimizer/"):
            group_name, _, key = path[len("optimizer/"):].rpartition(".")
            self._find_param_group(trainer, group_name)[key] = value
        else:
            obj, attr = self._resolve_attr(pl_module, path)
            setattr(obj, attr, value)

    # -- hooks ---------------------------------------------------------------
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._total = max(1, int(trainer.estimated_stepping_batches))
        # fail-fast валидация путей (контракт §6.3: инварианты в коде, не в YAML)
        for path in self.schedules:
            if path.startswith("optimizer/"):
                group_name, _, _key = path[len("optimizer/"):].rpartition(".")
                self._find_param_group(trainer, group_name)
            else:
                obj, attr = self._resolve_attr(pl_module, path)
                if not hasattr(obj, attr):
                    raise AttributeError(
                        f"ScheduleDriver: у {type(obj).__name__} нет атрибута {attr!r} (путь {path!r})"
                    )

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                             batch, batch_idx: int) -> None:
        step = trainer.global_step
        for path, sched in self.schedules.items():
            value = float(sched(step, self._total))
            self._assign(trainer, pl_module, path, value)
            if self.log_values:
                pl_module.log(f"sched/{path}", value, on_step=True, on_epoch=False,
                              rank_zero_only=True)


def build_warmup_cosine(optimizer, warmup_steps: int, total_steps: int,
                        start_factor: float = 0.1) -> SequentialLR | CosineAnnealingLR:
    """LR-расписание: линейный прогрев -> косинусный отжиг (стандарт SSL/классификации)."""
    total_steps = max(int(total_steps), 1)
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
