import json

import numpy as np
import pytest

from vision_lab.data.preprocessing import (
    ColorConstancy,
    SourceAlignment,
    compute_source_stats,
    shades_of_gray,
)


def test_shades_of_gray_neutralizes_color_cast():
    rng = np.random.RandomState(0)
    base = rng.uniform(0.2, 0.8, size=(32, 32, 3)).astype(np.float32)
    tinted = np.clip(base * np.array([1.3, 1.0, 0.7], np.float32), 0, 1)  # тёплый заливающий свет

    corrected = shades_of_gray(tinted, power=6)
    assert corrected.dtype == np.float32
    assert corrected.min() >= 0.0 and corrected.max() <= 1.0
    # дисбаланс каналов после коррекции меньше, чем до
    def imbalance(x):
        m = x.mean(axis=(0, 1))
        return m.max() - m.min()
    assert imbalance(corrected) < imbalance(tinted)


def test_color_constancy_step_signature():
    img = np.random.RandomState(1).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    out = ColorConstancy(power=6)(img, {"source": "dev"})
    assert out.shape == img.shape


def test_source_alignment_moves_source_to_reference():
    stats = {
        "_global": ([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        "dev_a": ([0.3, 0.3, 0.3], [0.1, 0.1, 0.1]),
    }
    align = SourceAlignment(stats)
    img = np.full((4, 4, 3), 0.3, np.float32)  # ровно среднее dev_a
    out = align(img, {"source": "dev_a"})
    np.testing.assert_allclose(out, 0.5, atol=1e-6)  # уехало в среднее референса


def test_missing_source_is_error_not_fallback():
    align = SourceAlignment({"_global": ([0.5] * 3, [0.2] * 3)})
    with pytest.raises(KeyError, match="dev_x"):
        align(np.zeros((2, 2, 3), np.float32), {"source": "dev_x"})


def test_compute_stats_roundtrip_with_dataset(tiny_dataset, tmp_path):
    from vision_lab.data.manifest import ManifestDataset

    ds = ManifestDataset(tiny_dataset["manifest"], root=tiny_dataset["root"], split="train")
    stats = compute_source_stats(ds)
    assert {"dev_a", "dev_b", "_global"} <= set(stats)

    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    align = SourceAlignment(path)
    out = align(np.full((4, 4, 3), 0.4, np.float32), {"source": "dev_b"})
    assert out.shape == (4, 4, 3) and out.dtype == np.float32
