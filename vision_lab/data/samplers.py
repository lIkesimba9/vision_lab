"""PK-семейство batch-сэмплеров (порт из прототипа, ТЗ §5.4) на DDP-mixin'е.

Структура батча — часть метода: triplet/SupCon/positive-shuffle не работают
без гарантии позитивов в батче. Все сэмплеры DDP-безопасны через
:class:`vision_lab.core.dist.DistributedBatchSamplerMixin`; трейнер обязан
выставить ``use_distributed_sampler=False``.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from torch.utils.data import Sampler

from vision_lab.core.dist import DistributedBatchSamplerMixin


def _pk_split(batch_size: int, n_labels_per_batch: int) -> int:
    """Проверка PK-аргументов; возвращает K — примеров на класс в батче."""
    if batch_size % n_labels_per_batch != 0:
        raise ValueError(
            f"batch_size={batch_size} должен делиться на n_labels_per_batch={n_labels_per_batch}"
        )
    return batch_size // n_labels_per_batch


def _group_by_class(labels: Sequence[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        groups[int(y)].append(idx)
    return dict(groups)


class PKCoverageBatchSampler(DistributedBatchSamplerMixin, Sampler):
    """PK-сэмплер с ПОЛНЫМ покрытием датасета за эпоху.

    Каждый класс перемешивается и режется на блоки по K = batch_size // P
    (последний блок добивается повтором того же класса). Каждый реальный
    пример попадает ровно в один блок ⇒ за эпоху модель видит ~100% данных.
    Батч = P блоков, по возможности из РАЗНЫХ классов. Частые классы дают
    больше блоков ⇒ распределение за эпоху ~натуральное (instance-balanced),
    при этом позитивы в батче гарантированы.

    -1 (no_label) — обычный «большой класс»: его блоки самопарятся в BYOL,
    triplet его игнорирует.
    """

    def __init__(self, labels: Sequence[int], batch_size: int, n_labels_per_batch: int,
                 drop_last: bool = False, seed: int = 42):
        self.batch_size = batch_size
        self.P = n_labels_per_batch
        self.K = _pk_split(batch_size, n_labels_per_batch)
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        self.class_to_indices = _group_by_class(labels)
        if self.P > len(self.class_to_indices):
            raise ValueError(
                f"n_labels_per_batch={self.P} > числа классов {len(self.class_to_indices)}"
            )

    def _num_batches_total(self) -> int:
        total_blocks = sum(
            math.ceil(len(idxs) / self.K) for idxs in self.class_to_indices.values()
        )
        return total_blocks // self.P if self.drop_last else math.ceil(total_blocks / self.P)

    def _build_epoch_batches(self) -> list[list[int]]:
        rng = random.Random(self._epoch_seed())

        # 1) режем каждый класс на блоки по K (полное покрытие, добивка повтором)
        blocks_by_class: dict[int, list[list[int]]] = {}
        for c, idxs in self.class_to_indices.items():
            idxs = idxs[:]
            rng.shuffle(idxs)
            blocks = []
            for i in range(0, len(idxs), self.K):
                blk = idxs[i:i + self.K]
                if len(blk) < self.K:
                    blk = blk + rng.choices(idxs, k=self.K - len(blk))
                blocks.append(blk)
            blocks_by_class[c] = blocks

        # 2) собираем батчи: P блоков, приоритет — разные классы
        batches: list[list[int]] = []
        remaining = sum(len(b) for b in blocks_by_class.values())
        all_classes = list(self.class_to_indices.keys())

        while remaining > 0:
            picked: list[list[int]] = []
            # проход 1: по одному блоку из разных классов (самые «полные» вперёд)
            for c in sorted(blocks_by_class, key=lambda c: len(blocks_by_class[c]), reverse=True):
                if len(picked) >= self.P:
                    break
                if blocks_by_class[c]:
                    picked.append(blocks_by_class[c].pop())
            # проход 2: добиваем слоты любыми оставшимися блоками (можно повтор класса)
            while len(picked) < self.P:
                avail = [c for c in blocks_by_class if blocks_by_class[c]]
                if not avail:
                    break
                c = max(avail, key=lambda c: len(blocks_by_class[c]))
                picked.append(blocks_by_class[c].pop())

            remaining = sum(len(b) for b in blocks_by_class.values())

            if len(picked) < self.P:
                if self.drop_last:
                    break
                # последний неполный батч: добиваем синтетическими блоками (повтор)
                while len(picked) < self.P:
                    c = rng.choice(all_classes)
                    picked.append(rng.choices(self.class_to_indices[c], k=self.K))

            batch = [i for blk in picked for i in blk]
            rng.shuffle(batch)
            batches.append(batch)

        return batches


class PositivePairsBatchSampler(DistributedBatchSamplerMixin, Sampler):
    """Простой PK: P случайных классов × K примеров; классы равновероятны."""

    def __init__(self, labels: Sequence[int], batch_size: int, n_labels_per_batch: int = 16,
                 drop_last: bool = True, replacement: bool = True, seed: int = 42):
        self.batch_size = batch_size
        self.n_labels_per_batch = n_labels_per_batch
        self.samples_per_class = _pk_split(batch_size, n_labels_per_batch)
        self.drop_last = drop_last
        self.replacement = replacement
        self.seed = seed
        self._epoch = 0

        self.class_to_indices = _group_by_class(labels)
        self.classes = list(self.class_to_indices.keys())
        if n_labels_per_batch > len(self.classes):
            raise ValueError(
                f"n_labels_per_batch={n_labels_per_batch} > числа классов {len(self.classes)}"
            )
        n = len(labels)
        self._len = n // batch_size if drop_last else (n + batch_size - 1) // batch_size

    def _num_batches_total(self) -> int:
        return self._len

    def _pick_class_samples(self, rng: random.Random, c: int) -> list[int]:
        idxs = self.class_to_indices[c]
        if len(idxs) >= self.samples_per_class:
            return rng.sample(idxs, self.samples_per_class)
        if not self.replacement:
            raise ValueError(
                f"У класса {c} мало примеров: {len(idxs)}, нужно {self.samples_per_class}"
            )
        return rng.choices(idxs, k=self.samples_per_class)

    def _build_epoch_batches(self) -> list[list[int]]:
        rng = random.Random(self._epoch_seed())
        batches = []
        for _ in range(self._len):
            batch: list[int] = []
            for c in rng.sample(self.classes, self.n_labels_per_batch):
                batch.extend(self._pick_class_samples(rng, c))
            rng.shuffle(batch)
            batches.append(batch)
        return batches


class PKBatchSampler(PositivePairsBatchSampler):
    """PK с температурным сэмплированием классов: P(class) ∝ counts**alpha.

    alpha=1 — натуральное распределение, alpha=0 — равновероятные классы
    (наследуемый :class:`PositivePairsBatchSampler` — его частный случай).
    """

    def __init__(self, labels: Sequence[int], batch_size: int, n_labels_per_batch: int,
                 alpha: float = 0.5, drop_last: bool = True, replacement: bool = True,
                 seed: int = 42):
        super().__init__(labels, batch_size, n_labels_per_batch,
                         drop_last=drop_last, replacement=replacement, seed=seed)
        counts = np.array([len(self.class_to_indices[c]) for c in self.classes], dtype=np.float64)
        weights = counts ** alpha
        self.class_probs = weights / weights.sum()

    def _build_epoch_batches(self) -> list[list[int]]:
        rng = random.Random(self._epoch_seed())
        nprng = np.random.RandomState(self._epoch_seed())
        batches = []
        for _ in range(self._len):
            batch: list[int] = []
            chosen = nprng.choice(len(self.classes), size=self.n_labels_per_batch,
                                  replace=False, p=self.class_probs)
            for ci in chosen:
                batch.extend(self._pick_class_samples(rng, self.classes[int(ci)]))
            rng.shuffle(batch)
            batches.append(batch)
        return batches
