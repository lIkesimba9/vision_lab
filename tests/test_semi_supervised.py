from vision_lab.data.semi_supervised import combined_semi_supervised_loader


def test_combined_loader_splits_labeled_unlabeled(tiny_dataset):
    loader = combined_semi_supervised_loader(
        manifest=tiny_dataset["manifest"],
        label_column="label",
        classes=tiny_dataset["classes"],
        root=tiny_dataset["root"],
        split="train",
        labeled_kwargs={"batch_size": 4},
        unlabeled_kwargs={"batch_size": 4},
    )
    seen_labeled, seen_unlabeled = 0, 0
    for batch, _, _ in loader:
        lab = batch["labeled"]
        unl = batch["unlabeled"]
        # размеченная ветка: все метки валидны
        assert (lab["label"] >= 0).all()
        # неразмеченная ветка: меток нет вовсе (чистый SSL-член)
        assert "label" not in unl
        seen_labeled += lab["image"].size(0)
        seen_unlabeled += unl["image"].size(0)
    assert seen_labeled > 0 and seen_unlabeled > 0


def test_unlabeled_count_matches_manifest(tiny_dataset):
    from vision_lab.data.manifest import ManifestDataset

    unlabeled = ManifestDataset(tiny_dataset["manifest"], root=tiny_dataset["root"],
                                split="train", where="label.isna()")
    # в train (10 строк) 1 без метки (индекс 9)
    assert len(unlabeled) == 1
