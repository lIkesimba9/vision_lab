"""Semi-supervised: размеченные + неразмеченные данные (ТЗ §5.3).

Два подмножества ОДНОГО манифеста (фильтр по null-меткам), два даталоадера,
``CombinedLoader``. Без фейковых меток и отдельных папок: неразмеченные — это
строки, где ``label_column`` пуст (в батче попадают как ``-1``, но для чистого
SSL-члена метки не нужны вовсе).
"""

from __future__ import annotations

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from vision_lab.data.manifest import ManifestDataset


def combined_semi_supervised_loader(
    manifest,
    label_column: str,
    classes,
    labeled_kwargs: dict | None = None,
    unlabeled_kwargs: dict | None = None,
    mode: str = "max_size_cycle",
    **common,
) -> CombinedLoader:
    """Строит CombinedLoader {'labeled': ..., 'unlabeled': ...} из одного манифеста.

    ``common`` — общие аргументы ManifestDataset (root, split, transform,
    image_size, preprocessing...). ``labeled_kwargs``/``unlabeled_kwargs`` —
    доп. параметры DataLoader соответствующей ветки (batch_size, sampler,
    num_workers). Размеченные = ``{label_column}.notna()``, неразмеченные =
    ``.isna()`` (грузятся без меток).

    В ``training_step`` батч приходит словарём:
    ``batch["labeled"]`` и ``batch["unlabeled"]``.
    """
    labeled_kwargs = dict(labeled_kwargs or {})
    unlabeled_kwargs = dict(unlabeled_kwargs or {})

    labeled_ds = ManifestDataset(
        manifest, where=f"{label_column}.notna()",
        label_column=label_column, classes=classes, **common)
    unlabeled_ds = ManifestDataset(
        manifest, where=f"{label_column}.isna()", **common)

    def build_loader(ds, kwargs):
        sampler = kwargs.pop("batch_sampler", None)
        if sampler is not None:
            return DataLoader(ds, batch_sampler=sampler, **kwargs)
        return DataLoader(ds, **kwargs)

    return CombinedLoader(
        {"labeled": build_loader(labeled_ds, labeled_kwargs),
         "unlabeled": build_loader(unlabeled_ds, unlabeled_kwargs)},
        mode=mode,
    )
