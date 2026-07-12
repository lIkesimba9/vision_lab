import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from vision_lab.core.batch import MISSING_LABEL, target_view
from vision_lab.data.manifest import ManifestDataset, map_labels


def make_ds(tiny_dataset, **kwargs):
    defaults = dict(
        manifest=tiny_dataset["manifest"],
        root=tiny_dataset["root"],
        split="train",
        label_column="label",
        classes=tiny_dataset["classes"],
    )
    defaults.update(kwargs)
    return ManifestDataset(**defaults)


def test_item_contract_and_missing_label(tiny_dataset):
    ds = make_ds(tiny_dataset)
    item = ds[0]
    assert item["image"].shape == (3, 16, 16) and item["image"].dtype == torch.float32
    assert item["image"].max() <= 1.0
    assert item["source"] in {"dev_a", "dev_b"}
    assert item["label"] == tiny_dataset["classes"].index("melanoma")
    assert ds[9]["label"] == MISSING_LABEL  # null в манифесте -> -1


def test_default_collate_and_target_view(tiny_dataset):
    ds = make_ds(tiny_dataset, taxonomy=tiny_dataset["taxonomy"])
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch["image"].shape == (4, 3, 16, 16)
    assert batch["label"].dtype == torch.int64
    assert batch["levels"].shape == (4, 2)
    assert isinstance(batch["sample_id"], list)
    assert set(target_view(batch)) == {"label", "levels"}


def test_levels_derived_from_taxonomy(tiny_dataset):
    ds = make_ds(tiny_dataset, taxonomy=tiny_dataset["taxonomy"])
    item = ds[0]  # melanoma -> coarse=malignant
    # coarse{benign,malignant}: malignant=1; fine{bcc,melanoma,nevus}: melanoma=1
    assert item["levels"].tolist() == [1, 1]
    assert ds[9]["levels"].tolist() == [MISSING_LABEL, MISSING_LABEL]
    # метки уровня для PK-сэмплера
    assert ds.level_labels("coarse").shape == (10,)


def test_split_and_where_filters(tiny_dataset):
    assert len(make_ds(tiny_dataset)) == 10
    assert len(make_ds(tiny_dataset, split="val", where=None)) == 2
    labeled = make_ds(tiny_dataset, where="label.notna()")
    assert len(labeled) == 9 and (labeled.labels != MISSING_LABEL).all()
    unlabeled = make_ds(tiny_dataset, where="label.isna()", label_column=None, classes=None)
    assert len(unlabeled) == 1


def test_unknown_label_error_and_missing_policy():
    s = pd.Series(["a", "b", None, "TYPO"])
    with pytest.raises(KeyError):
        map_labels(s, {"a": 0, "b": 1}, unknown="error")
    ids = map_labels(s, {"a": 0, "b": 1}, unknown="missing")
    np.testing.assert_array_equal(ids, [0, 1, -1, -1])


def test_missing_required_column_fails_fast(tiny_dataset):
    df = pd.read_parquet(tiny_dataset["manifest"]).drop(columns=["source"])
    with pytest.raises(ValueError, match="source"):
        ManifestDataset(df, root=tiny_dataset["root"])


def test_ssl_mode_without_labels(tiny_dataset):
    ds = ManifestDataset(tiny_dataset["manifest"], root=tiny_dataset["root"],
                         split="train", image_size=(8, 8))
    item = ds[0]
    assert item["image"].shape == (3, 8, 8)
    assert "label" not in item
