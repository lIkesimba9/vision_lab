"""Shared inference scaffold: load a checkpoint, build the eval transform, run
(optionally TTA-averaged) logits through the head contract.

This exact `load_module -> build transform -> TTA-average predict_logits`
sequence was copy-pasted across predict.py, submit.py, evaluate.py, eval_embeddings.py,
infer_scc_binary.py and ~14 `scripts/*` (the eval_*/oof_*/make_* cluster). Fixing
the TTA math or the normalization once, here, now fixes it everywhere.

Every head follows `criterion.predict_logits(model(x)["embeddings"]) -> (B, C)`
(see docs/FRAMEWORK.md), so this works for any LinearHead/AAMHead/SubCenterHead/...
checkpoint without special-casing.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import albumentations as A
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.utils import load_image
from src.hydra_setup import register_all_resolvers

# Composing a run's saved config needs the custom resolvers (${derma_root:},
# ${len:...}, ${config_names:...}). Register them once at import.
register_all_resolvers()


def tta_views(x: torch.Tensor):
    """4-view flip TTA (identity, hflip, vflip, rot180)."""
    yield x
    yield torch.flip(x, dims=[3])     # hflip
    yield torch.flip(x, dims=[2])     # vflip
    yield torch.flip(x, dims=[2, 3])  # rot180


def build_eval_transform(cfg) -> A.Compose:
    """Deterministic eval transform (Resize -> Normalize -> ToTensor) from a run cfg."""
    return A.Compose([
        A.Resize(cfg.height, cfg.width, interpolation=1, p=1.0),
        A.Normalize(mean=list(cfg.mean), std=list(cfg.std), max_pixel_value=1, p=1.0),
        A.ToTensorV2(p=1.0),
    ])


def load_module(ckpt_path, device):
    """Instantiate the LightningModule from `<run>/.hydra/config.yaml` and load weights.

    Returns (module.eval().to(device), cfg). The run dir is inferred as the
    checkpoint's grandparent (…/<run>/checkpoints/<ckpt>).
    """
    run_dir = Path(ckpt_path).parent.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    module = hydra.utils.instantiate(
        cfg.trainer, _recursive_=False,
        monitor_metric=cfg.monitor_metric, num_classes=cfg.num_classes, val_loader_names=["v"],
    )
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    module.load_state_dict(state, strict=True)
    return module.eval().to(device), cfg


@torch.no_grad()
def tta_logits(module, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Mean logits over TTA views for one batch (single view if `tta` is False)."""
    views = list(tta_views(x)) if tta else [x]
    return sum(
        module.criterion.predict_logits(module.model(v)["embeddings"]) for v in views
    ) / len(views)


class PathListDataset(Dataset):
    """Minimal dataset over a list of image paths + an albumentations transform."""

    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": self.transform(image=load_image(self.paths[i]))["image"]}


def select_top_epochs(run_dir, k: int = 3, metric: str = "val/f1_macro") -> list[int]:
    """Top-`k` epoch indices for a run by a TensorBoard scalar.

    Mirrors the `top3()` helper duplicated across ~15 scripts: read `metric` from
    the run's TB events and take the k best epochs; if TB is unavailable, fall back
    to the last k epochs by checkpoint filename (robust for losses that never
    logged the metric). Returned indices are sorted ascending.
    """
    run_dir = str(run_dir)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = (glob.glob(run_dir + "/tb_logs/version_0/events*")
                  or glob.glob(run_dir + "/**/events*", recursive=True))
        ea = EventAccumulator(events[0])
        ea.Reload()
        vals = [s.value for s in sorted(ea.Scalars(metric), key=lambda s: s.step)]
        if vals:
            return sorted(sorted(range(len(vals)), key=lambda i: -vals[i])[:k])
    except Exception:
        pass
    epochs = sorted(
        int(os.path.basename(c).split("_")[0][2:])
        for c in glob.glob(run_dir + "/checkpoints/ep*.ckpt")
    )
    return epochs[-k:]
