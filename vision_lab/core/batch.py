"""Контракт батча — плоский dict, совместимый с ``default_collate``.

Единый контракт для классификации и SSL: один и тот же батч проходит через
любой трейнер, поэтому positive-shuffle/иерархические опции SSL получают
метки «бесплатно», а multi-task не меняет форму батча.

Инварианты (проверяются в манифест-датасете, не в трейнерах):

* **отсутствующая скалярная метка — всегда ``MISSING_LABEL`` (-1)**, никогда
  None и никогда «ключа нет у части сэмплов»: набор ключей постоянен для
  датасета (иначе ломается default_collate и появляются ветвления в коде);
* ``image`` — единственный гарантированный ключ;
* новые задачи/модальности = новые ключи по зарезервированным неймспейсам,
  ноль изменений в трейнерах:
  - ``image_<modality>`` — доп. входы (depth, cam2, volume, ...);
  - ``label_<task>``     — доп. разреженные таргеты (multi-task), маскируются -1;
  - ``target_<name>``    — плотные таргеты (mask, depth_gt, ...) для будущих задач.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from torch import Tensor

#: Единственная кодировка «метки нет» для разреженных таргетов.
MISSING_LABEL = -1

#: Префиксы входных ключей (не попадают в target_view).
INPUT_PREFIXES: tuple[str, ...] = ("image",)

#: Мета-ключи (не входы и не таргеты).
META_KEYS: tuple[str, ...] = ("sample_id", "source")


class Batch(TypedDict, total=False):
    """Типизация батча. В рантайме это обычный dict."""

    image: Tensor  # (B, 3, H, W) float32; SSL — до Normalize, CLS — нормализован на CPU
    label: Tensor  # (B,) long — id класса; multi-label: (B, C) мульти-хот из {0,1,-1}; -1 = не размечено
    levels: Tensor  # (B, L) long — уровни таксономии (грубый -> тонкий); -1 = нет метки
    # label_<level>: (B,) long — те же уровни плоскими ключами (для multi-task/иерархических голов)
    sample_id: list[str]
    source: list[str]  # домен/источник из колонки манифеста (§7.4), не из пути


def target_view(batch: Mapping[str, object]) -> dict[str, Tensor]:
    """Под-словарь таргетов батча: ``label``, ``levels``, ``label_*``, ``target_*``.

    Именно его получают головы (``ClassifierHead.forward(emb, targets)``) —
    голова сама выбирает нужный ключ через свой ``target_key``.
    """
    return {
        k: v  # type: ignore[misc]
        for k, v in batch.items()
        if not k.startswith(INPUT_PREFIXES) and k not in META_KEYS
    }
