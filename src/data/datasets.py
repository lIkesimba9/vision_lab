import torch
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from typing import List, Tuple, Dict
import os
from tqdm import tqdm
from torch import nn
import numpy as np
from src.data.utils import load_image

import random
import math
import torch.distributed as dist
from collections import defaultdict
from torch.utils.data import Sampler


def _dist_info():
    """world_size, rank — даже если DDP ещё не инициализирован."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


class _DistributedBatchSamplerMixin:
    """
    Превращает любой batch-sampler в DDP-безопасный.

    Дочерний класс реализует:
        _build_epoch_batches() -> list[list[int]]   # детерминирован по self._epoch_seed()
        _num_batches_total()   -> int               # сколько батчей до шардинга (стабильно)

    Гарантии:
      * каждый ранг строит ОДИН И ТОТ ЖЕ полный список батчей (один сид на эпоху) и
        берёт непересекающийся срез batches[rank::world] → нет дублей между GPU;
      * число батчей одинаково на всех рангах (обрезаем остаток) → нет дедлока DDP;
      * reshuffle между эпохами через set_epoch (Lightning вызывает его сам).
    """

    seed: int = 42

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._epoch_set_externally = True

    def _epoch_seed(self):
        return self.seed + getattr(self, "_epoch", 0)

    def _sharded_batches(self):
        batches = self._build_epoch_batches()
        world, rank = _dist_info()
        if world > 1:
            usable = (len(batches) // world) * world
            batches = batches[rank:usable:world]
        # одиночный GPU: продвигаем эпоху сами, чтобы был reshuffle, если set_epoch не звали
        if world == 1 and not hasattr(self, "_epoch_set_externally"):
            self._epoch = getattr(self, "_epoch", -1) + 1
        return batches

    def __iter__(self):
        for b in self._sharded_batches():
            yield b

    def __len__(self):
        world, _ = _dist_info()
        n = self._num_batches_total()
        return n // world if world > 1 else n


class PKCoverageBatchSampler(_DistributedBatchSamplerMixin, Sampler):
    """
    PK-семплер с ПОЛНЫМ покрытием датасета за эпоху + DDP-safe.

    Идея: каждый класс перемешивается и режется на блоки по K = batch_size // P примеров
    (последний блок добивается с повтором того же класса). Каждый реальный пример попадает
    ровно в один блок ⇒ за эпоху модель видит ~100% данных. Батч = P блоков, по возможности
    из РАЗНЫХ классов (PK-структура для triplet/positive-pair трюка). Частые классы дают
    больше блоков ⇒ распределение по эпохе ~натуральное (instance-balanced), что и нужно для
    обучения представлений, но при этом гарантировано покрытие и наличие позитивов в батче.

    -1 (no_label) — обычный большой класс: его блоки самопарятся в BYOL (triplet их игнорит).
    """

    def __init__(
        self,
        labels,
        batch_size: int,
        n_labels_per_batch: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if batch_size % n_labels_per_batch != 0:
            raise ValueError(
                f"batch_size={batch_size} должен делиться на n_labels_per_batch={n_labels_per_batch}"
            )
        self.batch_size = batch_size
        self.P = n_labels_per_batch
        self.K = batch_size // n_labels_per_batch
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(labels):
            self.class_to_indices[int(y)].append(idx)
        self.class_to_indices = dict(self.class_to_indices)

        if self.P > len(self.class_to_indices):
            raise ValueError(
                f"n_labels_per_batch={self.P} > num_classes={len(self.class_to_indices)}"
            )

    def _num_batches_total(self):
        total_blocks = sum(
            math.ceil(len(idxs) / self.K) for idxs in self.class_to_indices.values()
        )
        return total_blocks // self.P if self.drop_last else math.ceil(total_blocks / self.P)

    def _build_epoch_batches(self):
        rng = random.Random(self._epoch_seed())

        # 1) режем каждый класс на блоки по K (полное покрытие, добивка с повтором)
        blocks_by_class = {}
        for c, idxs in self.class_to_indices.items():
            idxs = idxs[:]
            rng.shuffle(idxs)
            blocks = []
            for i in range(0, len(idxs), self.K):
                blk = idxs[i : i + self.K]
                if len(blk) < self.K:
                    blk = blk + rng.choices(idxs, k=self.K - len(blk))
                blocks.append(blk)
            blocks_by_class[c] = blocks

        # 2) собираем батчи: P блоков, приоритет — разные классы
        batches = []
        remaining = sum(len(b) for b in blocks_by_class.values())
        all_classes = list(self.class_to_indices.keys())

        while remaining > 0:
            picked = []
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


class PositivePairsBatchSampler(_DistributedBatchSamplerMixin, Sampler):
    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        n_labels_per_batch: int = 16,
        drop_last: bool = True,
        replacement: bool = True,
        seed: int = 42,
    ):
        """
        labels: метка класса для каждого элемента датасета
        batch_size: размер батча
        n_labels_per_batch: сколько разных классов должно быть в батче
        drop_last: если True, число батчей берётся по floor
        replacement: если у класса мало примеров, добирать с повторением
        """
        if batch_size % n_labels_per_batch != 0:
            raise ValueError(
                f"batch_size={batch_size} должен делиться на n_labels_per_batch={n_labels_per_batch}"
            )

        self.batch_size = batch_size
        self.n_labels_per_batch = n_labels_per_batch
        self.samples_per_class = batch_size // n_labels_per_batch
        self.drop_last = drop_last
        self.replacement = replacement
        self.seed = seed
        self._epoch = 0

        self.speaker_to_indices = defaultdict(list)
        for idx, spk in enumerate(labels):
            self.speaker_to_indices[int(spk)].append(idx)

        self.labels = list(self.speaker_to_indices.keys())

        if drop_last:
            self._len = len(labels) // batch_size
        else:
            self._len = (len(labels) + batch_size - 1) // batch_size

    def _num_batches_total(self):
        return self._len

    def _build_epoch_batches(self):
        rng = random.Random(self._epoch_seed())
        batches = []
        for _ in range(self._len):
            batch = []
            chosen_classes = rng.sample(self.labels, self.n_labels_per_batch)
            for label in chosen_classes:
                indices = self.speaker_to_indices[label]
                if len(indices) >= self.samples_per_class:
                    picked = rng.sample(indices, self.samples_per_class)
                else:
                    if not self.replacement:
                        raise ValueError(
                            f"У класса {label} мало примеров: {len(indices)}, "
                            f"а нужно {self.samples_per_class}"
                        )
                    picked = rng.choices(indices, k=self.samples_per_class)
                batch.extend(picked)
            rng.shuffle(batch)
            batches.append(batch)
        return batches


class PKBatchSampler(_DistributedBatchSamplerMixin, Sampler):
    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        n_labels_per_batch: int,
        alpha: float = 0.5,
        drop_last: bool = True,
        replacement: bool = True,
        seed: int = 42,
    ):
        if batch_size % n_labels_per_batch != 0:
            raise ValueError(
                f"batch_size={batch_size} должен делиться на n_labels_per_batch={n_labels_per_batch}"
            )

        self.batch_size = batch_size
        self.n_labels_per_batch = n_labels_per_batch
        self.samples_per_class = batch_size // n_labels_per_batch
        self.drop_last = drop_last
        self.replacement = replacement
        self.seed = seed
        self._epoch = 0

        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(labels):
            self.class_to_indices[int(y)].append(idx)

        self.classes = list(self.class_to_indices.keys())
        if self.n_labels_per_batch > len(self.classes):
            raise ValueError(
                f"n_labels_per_batch={self.n_labels_per_batch} > num_classes={len(self.classes)}"
            )

        counts = np.array([len(self.class_to_indices[c]) for c in self.classes], dtype=np.float32)
        weights = counts ** alpha
        self.class_probs = weights / weights.sum()

        self._len = len(labels) // batch_size if drop_last else (len(labels) + batch_size - 1) // batch_size

    def _num_batches_total(self):
        return self._len

    def _build_epoch_batches(self):
        rng = random.Random(self._epoch_seed())
        nprng = np.random.RandomState(self._epoch_seed())
        batches = []
        for _ in range(self._len):
            batch = []
            chosen_classes = nprng.choice(
                self.classes,
                size=self.n_labels_per_batch,
                replace=False,
                p=self.class_probs,
            )
            for c in chosen_classes:
                c = int(c)
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.samples_per_class:
                    picked = rng.sample(idxs, self.samples_per_class)
                else:
                    if not self.replacement:
                        raise ValueError(
                            f"У класса {c} мало примеров: {len(idxs)}, нужно {self.samples_per_class}"
                        )
                    picked = rng.choices(idxs, k=self.samples_per_class)
                batch.extend(picked)
            rng.shuffle(batch)
            batches.append(batch)
        return batches



# Опечатки в исходных CSV. Нормализуем ДО маппинга в id: иначе строка либо молча
# выбрасывается (drop_unmapped=True), либо роняет `.astype(int)` на NaN.
# Сюда — только точные опечатки канонических меток, НЕ синонимы разных таксономий
# (для тех есть label_to_id в конфиге и VAL_LABEL2ID в src/data/hier.py).
LABEL_TYPOS = {
    "BLK": "BKL",   # derm7pt: 45 строк в train11/valid11/test11
}


def _read_metadata(csv_path, image_col, label_col, label_to_id, drop_unmapped):
    """Читает один CSV, делает 'image' абсолютным путём (parent/data/<x>) и мапит метку в id.
    Возвращает DataFrame со всеми исходными колонками (нужны diag-уровни в HierImageDataset)."""
    df = pd.read_csv(csv_path)
    df["image"] = df[image_col].apply(lambda x: Path(csv_path).parent / f"data/{x}")
    labels = df[label_col].map(lambda v: LABEL_TYPOS.get(v, v))
    df["label"] = labels.map(label_to_id)
    if drop_unmapped:
        df = df[df["label"].notna()]
    return df


def _read_no_label(no_label_metadata):
    df = pd.read_csv(no_label_metadata)
    df["image"] = df["image"].apply(lambda x: Path(no_label_metadata).parent / f"data/{x}")
    df["label"] = -1
    return df


class ImageDataset(Dataset):
    """Картинки + 11-классовая метка из набора CSV (+ опц. пул no_label с меткой -1).

    drop_unmapped=True — выбросить строки, чья метка не в label_to_id (NaN после map): нужно
    для валидации (melanoscope 'UNK', синонимы). По умолчанию False — путь обычной классификации
    не меняется. image_col/label_col — имена колонок (melanoscope: 'filename'/'melanoscope_labels').
    """

    def __init__(self, list_of_csv_path: List[str], label_to_id: Dict[str, int], image_size=None,
                 no_label_metadata=None, transforms=None, drop_unmapped: bool = False,
                 image_col: str = "image", label_col: str = "label",
                 label_col2: str = None, label_to_id2: Dict[str, int] = None,
                 color_constancy: int = None, cc_align_json: str = None):
        super().__init__()
        frames = [_read_metadata(p, image_col, label_col, label_to_id, drop_unmapped)
                  for p in list_of_csv_path]
        if no_label_metadata is not None:
            frames.append(_read_no_label(no_label_metadata))

        df = pd.concat(frames, ignore_index=True)
        self.path_to_images = df["image"].values
        self.labels = df["label"].astype(int).values
        # опц. вторая метка (multi-task aux): нет колонки / NaN / вне словаря -> -1 (маскируется в лоссе)
        self.labels2 = None
        if label_col2 is not None and label_to_id2 is not None:
            col = df[label_col2] if label_col2 in df.columns else pd.Series([None] * len(df))
            self.labels2 = col.map(label_to_id2).fillna(-1).astype(int).values
        self.image_size = image_size
        self.transforms = transforms
        self.color_constancy = color_constancy
        # per-attribution (per-source) colour alignment: map each image from its
        # source's SoG stats to a global reference, so between-device colour bias is
        # removed while within-device relative colour is kept. Source derived from path.
        self.cc_align = None
        if cc_align_json is not None:
            import json
            d = json.load(open(cc_align_json))
            ref = d["_global_train"]
            self.cc_align = {
                "ref": (np.asarray(ref["mean"], np.float32), np.asarray(ref["std"], np.float32)),
                "by_source": {k: (np.asarray(v["mean"], np.float32), np.asarray(v["std"], np.float32))
                              for k, v in d.items() if not k.startswith("_")},
            }

    def __len__(self):
        return len(self.labels)

    def _load(self, index):
        path = self.path_to_images[index]
        image = load_image(path, self.image_size)
        if self.color_constancy:
            from src.color_constancy import shades_of_gray
            image = shades_of_gray(image, power=self.color_constancy)
        if self.cc_align is not None:
            m_ref, s_ref = self.cc_align["ref"]
            src = Path(path).parent.parent.name          # .../<source>/data/<image>
            m_src, s_src = self.cc_align["by_source"].get(src, (m_ref, s_ref))
            image = np.clip((image - m_src) / s_src * s_ref + m_ref, 0.0, 1.0).astype(np.float32)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return image

    def __getitem__(self, index) -> Dict:
        item = {"image": self._load(index), "label": int(self.labels[index])}
        if self.labels2 is not None:
            item["label11"] = int(self.labels2[index])
        return item


class HierImageDataset(ImageDataset):
    """ImageDataset + иерархические per-image метки из diag-колонок ISIC-метадата.

    Дополнительно отдаёт ``levels`` (LongTensor (L,)) — id по каждому уровню из ``level_columns``
    (по умолчанию diag1/2/3, см. src.data.hier). Отсутствующая колонка / NaN / значение вне
    словаря -> -1 (игнор в иерархическом лоссе). self.labels (для PK-сэмплера) — 11-классовые.

    Строки БЕЗ 11-класса НЕ выбрасываются: label=-1 (игнор в target-члене лосса), но diag-уровни
    читаются как есть. Так milk_test / unlabeled / частично-размеченные external участвуют в BYOL
    всегда, а в diag-уровнях — там, где у них есть diag-колонки. no_label -> все уровни -1.
    """

    def __init__(self, list_of_csv_path: List[str], label_to_id: Dict[str, int],
                 level_columns: List[str] = None, level_vocabs: Dict[str, Dict[str, int]] = None,
                 image_size=None, no_label_metadata=None, transforms=None, sampler_level: str = None):
        from src.data.hier import LEVEL_COLUMNS, LEVEL_VOCABS
        self.level_columns = list(LEVEL_COLUMNS if level_columns is None else level_columns)
        self.level_vocabs = LEVEL_VOCABS if level_vocabs is None else level_vocabs

        # drop_unmapped=False: строки без 11-класса оставляем (label NaN -> -1), diag читаем отдельно
        frames = [_read_metadata(p, "image", "label", label_to_id, drop_unmapped=False)
                  for p in list_of_csv_path]
        if no_label_metadata is not None:
            frames.append(_read_no_label(no_label_metadata))
        df = pd.concat(frames, ignore_index=True)
        df["label"] = df["label"].fillna(-1)

        self.path_to_images = df["image"].values
        self.labels = df["label"].astype(int).values
        self.level_ids = self._map_levels(df)
        self.image_size = image_size
        self.transforms = transforms

        # метки для PK-сэмплера: по умолчанию 11-класс (self.labels); либо по diag-уровню
        # (sampler_level, напр. 'diagnosis_3') — батчи с PK-структурой на этом уровне.
        # 11-классовый target в __getitem__ при этом НЕ меняется (всегда self.labels).
        if sampler_level is None:
            self.sampler_labels = self.labels
        else:
            self.sampler_labels = self.level_ids[:, self.level_columns.index(sampler_level)]

        # sanity покрытия: n с 11-классом и доля -1 по diag-уровням (высокая доля = либо данные
        # без 11-класса/diag — это норм для milk_test/unlabeled, либо рассинхрон словаря hier.py)
        n_labeled = int((self.labels != -1).sum())
        frac_missing = (self.level_ids == -1).mean(axis=0).round(3)
        print(f"[HierImageDataset] N={len(self.labels)} с 11-классом={n_labeled} "
              f"({n_labeled/max(len(self.labels),1):.1%}); доля -1 по уровням "
              f"{dict(zip(self.level_columns, frac_missing.tolist()))}")

    def _map_levels(self, df) -> np.ndarray:
        """(N, L) int: id уровня для каждой строки, -1 если колонки нет / NaN / нет в словаре."""
        cols = []
        for c in self.level_columns:
            voc = self.level_vocabs[c]
            if c in df.columns:
                cols.append(df[c].map(lambda v: voc.get(v, -1) if isinstance(v, str) else -1).to_numpy())
            else:
                cols.append(np.full(len(df), -1))
        return np.stack(cols, axis=1).astype(np.int64)

    def __getitem__(self, index) -> Dict:
        item = super().__getitem__(index)
        item["levels"] = torch.from_numpy(self.level_ids[index]).long()
        return item