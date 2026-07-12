"""Контракт SSL-метода (ТЗ §4.3) + переиспользуемый EMA-учитель.

Один контракт покрывает четыре семейства SSL. Различия — детали конкретного
метода, а не контракта:

============== ================= ================== ========== ==============
Семейство      EMA-teacher       gather по рангам    masking    probe-энкодер
============== ================= ================== ========== ==============
BYOL           да (online→tgt)   нет                нет        online backbone
DINO/iBOT      да (student→tgt)  нет                iBOT: да   teacher backbone
SimCLR/MoCo    MoCo: да          да (all_gather)    нет        online backbone
MAE            нет               нет                да         encoder tokens
I-JEPA         да (ctx→target)   нет                да         target encoder
============== ================= ================== ========== ==============

:class:`SSLMethod` владеет энкодерами, головами И GPU-вьюхами;
``forward(batch) -> dict лоссов``. Расписания (EMA-tau, teacher-temp) НЕ зашиты
в модуль — их выставляет :class:`~vision_lab.core.schedules.ScheduleDriver` в
атрибуты ``current_tau``/``teacher_temp``. EMA-обновление зовётся трейнером в
``on_before_zero_grad`` — ровно раз на optimizer step (корректно при grad
accumulation).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import nn


class MomentumTeacher(nn.Module):
    """EMA-копия student-модуля — ЕДИНСТВЕННАЯ реализация EMA в библиотеке.

    Учитель — submodule, поэтому его веса автоматически попадают в
    ``state_dict`` (resume бесплатно). ``factory`` — для модулей, которые нельзя
    deepcopy'ить (weight_norm-головы DINO): создаём чистого близнеца и копируем веса.
    """

    def __init__(self, student: nn.Module, factory: Callable[[], nn.Module] | None = None):
        super().__init__()
        self.module = factory() if factory is not None else copy.deepcopy(student)
        self.module.load_state_dict(student.state_dict())
        self.module.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def update(self, student: nn.Module, tau: float) -> None:
        """EMA-шаг: ``teacher = tau*teacher + (1-tau)*student``.

        Параметры усредняются, буферы (BN-статистики) копируются напрямую.
        """
        for pt, ps in zip(self.module.parameters(), student.parameters(), strict=True):
            pt.lerp_(ps.detach(), 1.0 - tau)
        for bt, bs in zip(self.module.buffers(), student.buffers(), strict=True):
            bt.copy_(bs)


class SSLMethod(nn.Module):
    """База SSL-методов. Подкласс владеет энкодерами/головами/вьюхами.

    Атрибуты-крутилки расписаний (обычные float, пишет ScheduleDriver):

    * ``current_tau`` — EMA-момент (методы с учителем);
    * ``teacher_temp`` — температура учителя (DINO-семейство).

    Обязательный контракт подкласса:

    * ``forward(batch) -> dict[str, Tensor]`` c ключом ``total_loss``; вьюхи
      генерируются ЗДЕСЬ на GPU (N независимых kornia-вьюх, §6.1); gather по
      рангам, если нужен, зовётся ЯВНО внутри forward (``all_gather_grad``);
    * ``extract_embeddings(images) -> (B, D)`` для kNN/linear-probe (докстринг
      обязан сказать, какой энкодер — он же экспортируется §4.4).

    Опциональный хук ``momentum_update()`` (no-op по умолчанию) трейнер зовёт
    ровно раз на optimizer step.
    """

    current_tau: float = 1.0
    teacher_temp: float = 0.04

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:  # pragma: no cover - интерфейс
        raise NotImplementedError

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def momentum_update(self) -> None:
        """EMA-обновление учителя. По умолчанию no-op (методы без учителя: MAE)."""
        return None
