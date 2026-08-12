"""Сэмпл-центричный манифест-датасет (ТЗ §7.1–7.2).

Пиксели и разметка раздельны, связь — через parquet-манифест; никаких
``ImageFolder`` (структура папок не кодирует ни метки, ни домен).

Строка манифеста = сэмпл. Обязательные колонки: ``sample_id``, ``source``,
``split``, ``input_<modality>_path``. Опциональные: ``source_format_<modality>``,
``label_<task>`` (nullable-строки — канонические имена классов), прочие
переносятся по запросу. Отсутствие метки в батче — всегда ``-1``
(:data:`vision_lab.core.batch.MISSING_LABEL`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vision_lab.core.batch import MISSING_LABEL
from vision_lab.data.decoders import decode_image
from vision_lab.data.taxonomy import Taxonomy

log = logging.getLogger(__name__)

#: Препроцессинг-шаг: (изображение HWC float32 [0,1], строка манифеста) -> изображение.
Preprocessing = Callable[[np.ndarray, Mapping], np.ndarray]


def _class_to_id(classes: Sequence[str] | Mapping[str, int]) -> dict[str, int]:
    if isinstance(classes, Mapping):
        return dict(classes)
    return {name: i for i, name in enumerate(classes)}


def map_labels(
    values: pd.Series,
    class_to_id: Mapping[str, int],
    unknown: str = "error",
) -> np.ndarray:
    """Строковые метки -> id; NaN/None -> -1.

    ``unknown``: 'error' — неизвестная метка роняет загрузку (дефолт, ловит
    рассинхрон словаря); 'missing' — считается неразмеченной (-1). Вариант
    «тихо выбросить строку» намеренно не поддержан — фильтруйте через ``where``.
    """
    ids = values.map(lambda v: class_to_id.get(v) if isinstance(v, str) else MISSING_LABEL)
    bad = ids.isna()
    if bad.any():
        unknown_values = sorted(values[bad].unique().tolist())
        if unknown == "error":
            raise KeyError(
                f"Метки вне словаря классов: {unknown_values[:10]} "
                f"(всего строк: {int(bad.sum())}). Обновите словарь или задайте unknown='missing'."
            )
        ids = ids.fillna(MISSING_LABEL)
    return ids.astype(np.int64).to_numpy()


def map_multilabel(
    values: pd.Series,
    class_to_id: Mapping[str, int],
    unknown: str = "error",
) -> np.ndarray:
    """Списки имён классов -> мульти-хот ``(N, C)``; null -> строка из -1.

    Колонка манифеста — ``list<string>`` (parquet). Присутствие списка означает
    ПОЛНУЮ разметку строки: не перечисленные классы = 0; пустой список валиден
    («ни один класс»). Отсутствие списка (null) — строка не размечена: -1 по
    всем классам (маскируется головой поэлементно).

    ``unknown``: 'error' — имя вне словаря роняет загрузку; 'missing' — имя
    игнорируется (строка остаётся размеченной по остальным классам).
    """
    n_class = len(class_to_id)
    out = np.full((len(values), n_class), MISSING_LABEL, dtype=np.int64)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        row = np.zeros(n_class, dtype=np.int64)
        for name in v:
            idx = class_to_id.get(name)
            if idx is None:
                if unknown == "error":
                    raise KeyError(
                        f"Метка {name!r} (строка {i}) вне словаря классов. "
                        f"Обновите словарь или задайте unknown='missing'."
                    )
                continue
            row[idx] = 1
        out[i] = row
    return out


class ManifestDataset(Dataset):
    """Датасет поверх parquet-манифеста; один на все форматы и задачи.

    Параметры:
        manifest: путь к .parquet либо готовый DataFrame.
        root: префикс для относительных путей манифеста (пути в манифесте
            хранятся относительными — переносимость между машинами).
        split: фильтр по колонке ``split`` (train/val/test); None — без фильтра.
        where: дополнительный pandas-query фильтр,
            например ``"label_diag.notna()"`` (labeled-подмножество для
            semi-supervised) или ``"source == 'clinic_a'"``.
        modality: какой вход читать (``input_<modality>_path``).
        label_column: колонка основного таргета -> ключ батча ``label``;
            None — чистый SSL без меток.
        label_kind: ``multiclass`` (строка-имя класса -> int id) или
            ``multilabel`` (list<string> -> мульти-хот ``(C,)`` из {0, 1, -1},
            см. :func:`map_multilabel`).
        classes: список имён классов (id по порядку) либо готовый словарь
            имя -> id. Обязателен при label_column.
        extra_label_columns: {колонка -> ключ батча ``label_<task>``} для multi-task.
        extra_classes: словари классов для extra-колонок (по имени колонки).
        taxonomy: добавляет в батч ``levels`` (L,) и по ключу ``label_<level>``
            на каждый уровень (для иерархических голов, см.
            :func:`~vision_lab.heads.hierarchical_head`); предки выводятся из
            метки label_column (только multiclass).
        transform: albumentations Compose (ожидается ключ ``image``); None —
            изображение конвертируется в CHW-тензор как есть (SSL-путь).
        preprocessing: детерминированные шаги до аугментаций
            (color constancy, пер-source выравнивание) — одинаковы на train
            и inference.
        image_size: (H, W) resize на декоде (SSL: decode -> resize -> tensor).
        max_side: декодировать JPEG сразу в уменьшенном масштабе, пока длинная
            сторона не меньше этого значения. В отличие от image_size не
            разворачивает кадр целиком и сохраняет соотношение сторон; на
            многомегапиксельных снимках именно декод, а не аугментации,
            оказывается узким местом обучения.
        unknown: политика неизвестных меток, см. :func:`map_labels`.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        root: str | Path | None = None,
        split: str | None = None,
        where: str | None = None,
        modality: str = "rgb",
        label_column: str | None = None,
        label_kind: str = "multiclass",
        classes: Sequence[str] | Mapping[str, int] | None = None,
        extra_label_columns: Mapping[str, str] | None = None,
        extra_classes: Mapping[str, Sequence[str] | Mapping[str, int]] | None = None,
        taxonomy: Taxonomy | str | Path | None = None,
        transform=None,
        preprocessing: Sequence[Preprocessing] = (),
        image_size: tuple[int, int] | None = None,
        max_side: int | None = None,
        unknown: str = "error",
    ):
        super().__init__()
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_parquet(manifest)

        self.input_column = f"input_{modality}_path"
        self.format_column = f"source_format_{modality}"
        required = ["sample_id", "source", "split", self.input_column]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"В манифесте нет обязательных колонок: {missing_cols}")

        if split is not None:
            df = df[df["split"] == split]
        if where is not None:
            df = df.query(where)
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"Манифест пуст после фильтров split={split!r}, where={where!r}")
        self._df = df

        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.preprocessing = tuple(preprocessing)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.max_side = int(max_side) if max_side is not None else None

        if isinstance(taxonomy, (str, Path)):
            taxonomy = Taxonomy.from_yaml(taxonomy)
        self.taxonomy = taxonomy

        # --- основной таргет -------------------------------------------------
        if label_kind not in {"multiclass", "multilabel"}:
            raise ValueError(f"label_kind={label_kind!r} (multiclass|multilabel)")
        self.label_kind = label_kind
        self.labels: np.ndarray | None = None
        self._label_names: pd.Series | None = None
        if label_column is not None:
            if classes is None:
                raise ValueError("label_column задан — нужен и словарь classes")
            if label_column not in df.columns:
                raise ValueError(f"Колонка {label_column!r} отсутствует в манифесте")
            self.class_to_id = _class_to_id(classes)
            if label_kind == "multilabel":
                self.labels = map_multilabel(df[label_column], self.class_to_id, unknown)
            else:
                self.labels = map_labels(df[label_column], self.class_to_id, unknown)
                self._label_names = df[label_column]

        # --- extra-таргеты (multi-task) --------------------------------------
        self._extra: dict[str, np.ndarray] = {}
        for col, batch_key in (extra_label_columns or {}).items():
            if not batch_key.startswith("label_"):
                raise ValueError(f"Ключ батча {batch_key!r} обязан начинаться с 'label_' (§7.2)")
            spec = (extra_classes or {}).get(col)
            if spec is None:
                raise ValueError(f"Для extra-колонки {col!r} не задан словарь extra_classes")
            self._extra[batch_key] = map_labels(df[col], _class_to_id(spec), unknown)

        # --- levels из таксономии --------------------------------------------
        self._levels: np.ndarray | None = None
        self._level_keys: tuple[str, ...] = ()
        if self.taxonomy is not None:
            if self._label_names is None:
                raise ValueError(
                    "taxonomy требует label_column с label_kind='multiclass' "
                    "(levels выводятся из имени метки)"
                )
            self._levels = np.stack([
                self.taxonomy.levels_vector(v if isinstance(v, str) else None)
                for v in self._label_names
            ])
            self._level_keys = tuple(f"label_{lv}" for lv in self.taxonomy.levels)
            clash = set(self._level_keys) & set(self._extra)
            if clash:
                raise ValueError(
                    f"extra_label_columns конфликтуют с ключами уровней таксономии: {sorted(clash)}"
                )

    # -- служебное ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._df)

    def level_labels(self, level: str) -> np.ndarray:
        """Метки уровня таксономии — например, для PK-сэмплера по уровню."""
        if self._levels is None:
            raise ValueError("Датасет создан без taxonomy")
        return self._levels[:, list(self.taxonomy.levels).index(level)]

    def _resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if self.root is None or p.is_absolute() else self.root / p

    # -- контракт Dataset ------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        row = self._df.iloc[index]
        fmt = row.get(self.format_column) if self.format_column in self._df.columns else None
        image = decode_image(self._resolve_path(row[self.input_column]), fmt,
                             self.image_size, self.max_side)

        for prep in self.preprocessing:
            image = prep(image, row)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

        item: dict = {
            "image": image,
            "sample_id": str(row["sample_id"]),
            "source": str(row["source"]),
        }
        if self.labels is not None:
            if self.label_kind == "multilabel":
                item["label"] = torch.from_numpy(self.labels[index])
            else:
                item["label"] = int(self.labels[index])
        for batch_key, ids in self._extra.items():
            item[batch_key] = int(ids[index])
        if self._levels is not None:
            item["levels"] = torch.from_numpy(self._levels[index])
            for j, key in enumerate(self._level_keys):
                item[key] = int(self._levels[index, j])
        return item
