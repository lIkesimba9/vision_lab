import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Мини-датасет: 12 PNG 16x16, parquet-манифест, taxonomy.yaml.

    Классы: melanoma/nevus/bcc (fine) под malignant/benign (coarse).
    3 сэмпла без метки (null) — для semi-supervised/SSL путей.
    2 источника: dev_a, dev_b.
    """
    rng = np.random.RandomState(0)
    img_dir = tmp_path / "processed"
    img_dir.mkdir()

    labels = ["melanoma", "nevus", "bcc"] * 3 + [None] * 3
    rows = []
    for i, label in enumerate(labels):
        name = f"img_{i:02d}.png"
        img = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / name), img)
        rows.append({
            "sample_id": f"s{i:02d}",
            "input_rgb_path": f"processed/{name}",
            "source_format_rgb": "png",
            "source": "dev_a" if i % 2 == 0 else "dev_b",
            "label": label,
            "split": "train" if i < 10 else "val",
        })
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest_path)

    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        """
levels: [coarse, fine]
nodes:
  malignant: {level: coarse}
  benign: {level: coarse}
  melanoma: {level: fine, parent: malignant}
  bcc: {level: fine, parent: malignant}
  nevus: {level: fine, parent: benign}
""",
        encoding="utf-8",
    )
    return {
        "root": tmp_path,
        "manifest": manifest_path,
        "taxonomy": taxonomy_path,
        "classes": ["bcc", "melanoma", "nevus"],
    }
