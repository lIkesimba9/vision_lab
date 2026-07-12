"""Параметр-группы оптимизатора: именованные, с weight-decay исключениями.

Именованные группы («backbone.decay», «head.no_decay», ...) — часть контракта:
по имени их адресует ``ScheduleDriver`` (путь ``optimizer/<name>.<key>``),
например DINOv2-рамп weight decay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import nn


def default_no_decay(name: str, param: nn.Parameter) -> bool:
    """Исключения из weight decay: bias и все 1D-параметры (нормализации, gains).

    ``param.ndim <= 1`` покрывает LayerNorm/BatchNorm/RMSNorm веса и bias без
    перечисления типов слоёв.
    """
    return param.ndim <= 1 or name.endswith(".bias")


def param_groups(
    modules: Mapping[str, nn.Module],
    base_lr: float,
    weight_decay: float = 0.0,
    lr_overrides: Mapping[str, float | None] | None = None,
    no_decay: Callable[[str, nn.Parameter], bool] = default_no_decay,
) -> list[dict]:
    """Собирает param-группы из именованных модулей.

    ``lr_overrides[name]``: None → base_lr; 0.0 → параметры формально в
    оптимизаторе, но не обновляются (замороженный бэкбон, BN-статистики
    продолжают адаптироваться в train-режиме); иное число → свой LR
    (low-LR доменная адаптация бэкбона).

    Каждая группа получает ключ ``name`` — «<модуль>.decay» / «<модуль>.no_decay».
    """
    lr_overrides = dict(lr_overrides or {})
    groups: list[dict] = []
    for mod_name, module in modules.items():
        override = lr_overrides.get(mod_name)
        lr = base_lr if override is None else override
        decay_params, no_decay_params = [], []
        for pname, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay_params if no_decay(pname, p) else decay_params).append(p)
        if decay_params:
            groups.append({"params": decay_params, "lr": lr,
                           "weight_decay": weight_decay, "name": f"{mod_name}.decay"})
        if no_decay_params:
            groups.append({"params": no_decay_params, "lr": lr,
                           "weight_decay": 0.0, "name": f"{mod_name}.no_decay"})
    if not groups:
        raise ValueError("param_groups: ни одного обучаемого параметра")
    return groups
