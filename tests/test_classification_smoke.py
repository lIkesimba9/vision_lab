"""Смоук: 1 батч через ClassificationTrainer на синтетике (CPU, §11.3).

Покрывает все типы пайплайнов §5.1: бинарный, многоклассовый, multi-label,
multi-task, иерархический.
"""

from functools import partial

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from vision_lab.core.module import ClassificationTrainer
from vision_lab.data.taxonomy import Taxonomy
from vision_lab.heads import AAMHead, LinearHead, MultiLabelHead, MultiTaskHead, hierarchical_head


class SyntheticImages(Dataset):
    def __init__(self, n=16, num_classes=4, size=32, aux=False,
                 multilabel=False, levels=False):
        self.n, self.c, self.size, self.aux = n, num_classes, size, aux
        self.multilabel, self.levels = multilabel, levels

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        item = {
            "image": torch.rand(3, self.size, self.size, generator=g),
            "sample_id": f"s{i}",
            "source": "synthetic",
        }
        if self.multilabel:
            item["label"] = torch.randint(0, 2, (self.c,), generator=g)
        else:
            label = int(torch.randint(0, self.c, (1,), generator=g))
            item["label"] = label
            if self.levels:  # иерархия: coarse выводится из fine-метки
                item["label_fine"] = label
                item["label_coarse"] = label % 2
        if self.aux:
            item["label_aux"] = int(torch.randint(0, 3, (1,), generator=g))
        return item


class TinyBackbone(torch.nn.Module):
    """Заглушка бэкбона (без timm-скачивания): conv -> global pool -> (B, D)."""

    out_dim = 16

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


def run_one_epoch(head, num_classes=4, task="multiclass", **ds_kwargs):
    ds = SyntheticImages(num_classes=num_classes, **ds_kwargs)
    val = SyntheticImages(n=8, num_classes=num_classes,
                          **{k: v for k, v in ds_kwargs.items() if k != "n"})
    trainer_module = ClassificationTrainer(
        backbone=TinyBackbone(),
        head=head,
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        num_classes=num_classes,
        task=task,
        warmup_steps=2,
    )
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                         enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(trainer_module, DataLoader(ds, batch_size=8),
                DataLoader(val, batch_size=8))
    return trainer_module


def test_ce_pipeline_smoke():
    run_one_epoch(LinearHead(4, 16, mode="ce"))


def test_aam_pipeline_smoke():
    run_one_epoch(AAMHead(4, 16))


def test_multitask_pipeline_smoke():
    head = MultiTaskHead(
        heads={"main": LinearHead(4, 16, target_key="label"),
               "aux": LinearHead(3, 16, target_key="label_aux")},
        primary="main",
    )
    run_one_epoch(head, aux=True)


def test_binary_pipeline_smoke():
    """Бинарная классификация = multiclass с num_classes=2 (baseline-first)."""
    run_one_epoch(LinearHead(2, 16, mode="ce"), num_classes=2)


def test_multilabel_pipeline_smoke():
    """Multi-label: мульти-хот таргет, task="multilabel" (Multilabel-метрики в val)."""
    module = run_one_epoch(MultiLabelHead(4, 16, mode="asl"),
                           task="multilabel", multilabel=True)
    assert module.val_f1_macro.__class__.__name__ == "MultilabelF1Score"


def test_hierarchical_pipeline_smoke():
    """Иерархия: под-голова на уровень таксономии, метрики по primary (fine)."""
    taxonomy = Taxonomy.from_dict({
        "levels": ["coarse", "fine"],
        "nodes": {
            "even": {"level": "coarse"}, "odd": {"level": "coarse"},
            "c0": {"level": "fine", "parent": "even"},
            "c1": {"level": "fine", "parent": "odd"},
            "c2": {"level": "fine", "parent": "even"},
            "c3": {"level": "fine", "parent": "odd"},
        },
    })
    head = hierarchical_head(taxonomy, embedding_dim=16, weights={"coarse": 0.3})
    assert head.target_key == "label_fine"  # метрики в val идут по primary-ключу
    run_one_epoch(head, levels=True)


def test_frozen_backbone_lr_zero():
    module = ClassificationTrainer(
        backbone=TinyBackbone(),
        head=LinearHead(4, 16, mode="ce"),
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        num_classes=4,
        backbone_lr=0.0,
    )
    module.trainer = type("T", (), {"estimated_stepping_batches": 10})()
    cfg = module.configure_optimizers()
    groups = {g["name"]: g for g in cfg["optimizer"].param_groups}
    assert groups["backbone.decay"]["lr"] == 0.0
    assert groups["head.decay"]["lr"] == 1e-3
