import numpy as np
import pytest

from vision_lab.data.taxonomy import Taxonomy

SPEC = {
    "levels": ["coarse", "fine"],
    "nodes": {
        "malignant": {"level": "coarse"},
        "benign": {"level": "coarse"},
        "melanoma": {"level": "fine", "parent": "malignant"},
        "nevus": {"level": "fine", "parent": "benign"},
    },
}


def test_vocab_deterministic_alphabetical():
    t = Taxonomy.from_dict(SPEC)
    assert t.vocab["coarse"] == {"benign": 0, "malignant": 1}
    assert t.vocab["fine"] == {"melanoma": 0, "nevus": 1}
    assert t.num_classes("fine") == 2


def test_levels_vector_full_and_partial_depth():
    t = Taxonomy.from_dict(SPEC)
    np.testing.assert_array_equal(t.levels_vector("melanoma"), [1, 0])  # malignant, melanoma
    np.testing.assert_array_equal(t.levels_vector("benign"), [0, -1])   # уровень тоньше метки = -1
    np.testing.assert_array_equal(t.levels_vector(None), [-1, -1])


def test_unknown_label_is_error_not_silent():
    t = Taxonomy.from_dict(SPEC)
    with pytest.raises(KeyError):
        t.levels_vector("UNK")


def test_invalid_specs_rejected():
    bad_level = {"levels": ["a"], "nodes": {"x": {"level": "b"}}}
    with pytest.raises(ValueError):
        Taxonomy.from_dict(bad_level)

    parent_not_coarser = {
        "levels": ["a", "b"],
        "nodes": {"p": {"level": "b"}, "c": {"level": "b", "parent": "p"}},
    }
    with pytest.raises(ValueError):
        Taxonomy.from_dict(parent_not_coarser)
