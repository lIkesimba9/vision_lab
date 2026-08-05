"""Конфиг-смоук-тест (ТЗ §8): каждый shipped-конфиг инстанцируется и делает шаг.

Ловит опечатки ключей (§8, «trasform»-урок), дрейф сигнатур, нарушения порядка
kornia. Тяжёлые бэкбоны подменяются крошечными timm test-моделями (test_convnext
/test_vit, 64 фичи) через оверрайд — без скачивания весов.
"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from vision_lab.configs import CONFIG_ROOT, register_resolvers

register_resolvers()

EXPERIMENTS = sorted((CONFIG_ROOT / "experiment").glob("*.yaml"))

# подмена бэкбонов на крошечные test-модели (pretrained=false, 64 фичи)
TINY = {"convnextv2_tiny": ("test_convnext", 64), "vit_small_patch16_224": ("test_vit", 64)}


def _shrink(cfg):
    """Рекурсивно заменяет model_name на test-модель и pretrained=false."""
    if OmegaConf.is_config(cfg) and OmegaConf.is_dict(cfg):
        if "model_name" in cfg and cfg.model_name in TINY:
            cfg.model_name = TINY[cfg.model_name][0]
        if "pretrained" in cfg:
            cfg.pretrained = False
        if "image_size" in cfg:
            cfg.image_size = 32
        if "local_size" in cfg:
            cfg.local_size = 16
        if "n_local" in cfg:
            cfg.n_local = 2
        if "out_dim" in cfg:
            cfg.out_dim = 128
        for key in cfg:
            if OmegaConf.is_missing(cfg, key):  # пропускаем ??? (embedding_dim и т.п.)
                continue
            _shrink(cfg[key])
    return cfg


class SyntheticDS(Dataset):
    def __init__(self, n=8, task="multiclass"):
        self.n, self.task = n, task

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        if self.task == "multilabel":
            label = torch.randint(0, 2, (3,), generator=g)  # мульти-хот (C,)
        else:
            label = int(i % 3)
        return {
            "image": torch.rand(3, 32, 32, generator=g),
            "label": label,
            "sample_id": f"s{i}",
            "source": "syn",
        }


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_config_instantiates_and_steps(path: Path):
    cfg = _shrink(OmegaConf.load(path))
    task = cfg.module.get("task", "multiclass")

    # embedding_dim = out_dim крошечной модели (64), там где голова его требует.
    # .keys() важен: OmegaConf `in` возвращает False для MISSING (???)-значений.
    if "head" in cfg.module.keys() and "embedding_dim" in cfg.module.head.keys():
        cfg.module.head.embedding_dim = 64

    module = instantiate(cfg.module, _convert_="all")

    # расписания навешиваем ScheduleDriver'ом (как в бою)
    callbacks = []
    if "schedules" in cfg:
        from vision_lab.core import ScheduleDriver

        schedules = {k: instantiate(v) for k, v in cfg.schedules.items()}
        callbacks.append(ScheduleDriver(schedules, log_values=False))

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, callbacks=callbacks,
                         limit_train_batches=2, limit_val_batches=0,
                         enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, DataLoader(SyntheticDS(task=task), batch_size=4))


def test_all_component_configs_are_valid_yaml():
    """Все YAML в группах читаются (нет синтаксических ошибок / битых якорей)."""
    for yaml_path in CONFIG_ROOT.rglob("*.yaml"):
        cfg = OmegaConf.load(yaml_path)
        assert cfg is not None, yaml_path
