"""Тесты двух режимов запуска (§8): Python-модуль (train.run) и CLI
(python -m vision_lab.train --config-name ...)."""

import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from vision_lab.train import run


def train_cfg(tiny_dataset) -> dict:
    """Полный конфиг обучения на tiny-датасете (крошечная timm test-модель)."""
    return {
        "seed": 0,
        "module": {
            "_target_": "vision_lab.core.ClassificationTrainer",
            "num_classes": 3,
            "backbone": {
                "_target_": "vision_lab.models.backbones.EmbeddingBackbone",
                "model_name": "test_convnext",
                "pretrained": False,
            },
            "head": {
                "_target_": "vision_lab.heads.LinearHead",
                "mode": "ce",
                "n_class": 3,
                "embedding_dim": 64,
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "_partial_": True,
                "lr": 1e-3,
            },
        },
        "data": {
            "train_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 4,
                "dataset": {
                    "_target_": "vision_lab.data.ManifestDataset",
                    "manifest": str(tiny_dataset["manifest"]),
                    "root": str(tiny_dataset["root"]),
                    "split": "train",
                    "label_column": "label",
                    "classes": tiny_dataset["classes"],
                    "image_size": [32, 32],
                },
            },
        },
        "trainer": {
            "_target_": "lightning.pytorch.Trainer",
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "limit_train_batches": 2,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    }


def test_run_as_python_module(tiny_dataset):
    """Python-режим: run(cfg) собирает компоненты и делает fit."""
    trainer = run(OmegaConf.create(train_cfg(tiny_dataset)))
    assert trainer.global_step > 0


def test_run_requires_data_section(tiny_dataset):
    cfg = train_cfg(tiny_dataset)
    del cfg["data"]
    with pytest.raises(ValueError, match="data.train_dataloader"):
        run(OmegaConf.create(cfg))


def test_cli_config_name(tiny_dataset, tmp_path):
    """CLI-режим: python -m vision_lab.train --config-dir ... --config-name ..."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    OmegaConf.save(OmegaConf.create(train_cfg(tiny_dataset)), cfg_dir / "train_smoke.yaml")
    result = subprocess.run(
        [sys.executable, "-m", "vision_lab.train",
         "--config-dir", str(cfg_dir), "--config-name", "train_smoke"],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_cli_override_from_command_line(tiny_dataset, tmp_path):
    """Hydra-оверрайды из командной строки доходят до компонентов."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    OmegaConf.save(OmegaConf.create(train_cfg(tiny_dataset)), cfg_dir / "train_smoke.yaml")
    result = subprocess.run(
        [sys.executable, "-m", "vision_lab.train",
         "--config-dir", str(cfg_dir), "--config-name", "train_smoke",
         "module.optimizer.lr=1e-4", "trainer.limit_train_batches=1"],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
