import torch

from vision_lab.core.batch import MISSING_LABEL, target_view


def test_target_view_splits_targets_from_inputs_and_meta():
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "image_depth": torch.zeros(2, 1, 8, 8),
        "label": torch.tensor([1, MISSING_LABEL]),
        "label_aux": torch.tensor([0, 2]),
        "levels": torch.tensor([[0, 1], [MISSING_LABEL, 3]]),
        "target_mask": torch.zeros(2, 8, 8),
        "sample_id": ["a", "b"],
        "source": ["dev1", "dev2"],
    }
    targets = target_view(batch)
    assert set(targets) == {"label", "label_aux", "levels", "target_mask"}


def test_target_view_empty_for_ssl_batch_without_labels():
    assert target_view({"image": torch.zeros(1, 3, 4, 4), "sample_id": ["x"]}) == {}
