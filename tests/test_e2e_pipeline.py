"""Сквозной приёмочный тест (ТЗ §13, этап 9): SSL → классификация → инференс.

Проверяет весь цикл на реальном (крошечном) timm-бэкбоне:
1. SSL-претрейн BYOL пару шагов → Lightning-чекпоинт;
2. загрузка бэкбона стадии SSL в классификацию через чекпоинт-контракт §4.4
   (strip_prefixes находит SSL-префикс автоматически);
3. дообучение головы + инференс с flip-TTA.

Это регрессионный тест миграции: доказывает, что контракты (backbone→тензор,
head, SSLMethod, checkpoint) стыкуются без ручной правки ключей.
"""

from __future__ import annotations

import copy
from functools import partial

import lightning.pytorch as pl
import timm
import torch
from torch.utils.data import DataLoader, Dataset

from vision_lab.core import ClassificationTrainer, SSLTrainer
from vision_lab.core.checkpoint import load_backbone, strip_prefixes
from vision_lab.heads import LinearHead
from vision_lab.inference import Predictor
from vision_lab.models.backbones import EmbeddingBackbone
from vision_lab.ssl.byol import BYOL
from vision_lab.ssl.gpu_augs import MultiViewAugment, build_view_pipeline

MODEL = "test_convnext"  # крошечная timm-модель (64 фичи), без скачивания


class SynDS(Dataset):
    def __init__(self, n=16, num_classes=3):
        self.n, self.c = n, num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"image": torch.rand(3, 32, 32, generator=g),
                "label": int(i % self.c), "sample_id": f"s{i}", "source": "syn"}


def _trainer(**kw):
    kw.setdefault("enable_checkpointing", False)
    return pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                      limit_train_batches=2, limit_val_batches=0,
                      enable_progress_bar=False, enable_model_summary=False,
                      logger=False, **kw)


def test_ssl_to_classification_to_inference(tmp_path):
    # --- 1. SSL-претрейн BYOL -> чекпоинт ---
    view = build_view_pipeline(32, scale=(0.5, 1.0))
    method = BYOL(EmbeddingBackbone(MODEL, pretrained=False),
                  MultiViewAugment([view, copy.deepcopy(view)]),
                  hidden_dim=64, projection_dim=32)
    ssl_module = SSLTrainer(method, optimizer=partial(torch.optim.AdamW, lr=1e-3))
    ssl_trainer = _trainer(enable_checkpointing=False, default_root_dir=str(tmp_path))
    ssl_trainer.fit(ssl_module, DataLoader(SynDS(), batch_size=8))
    ckpt_path = tmp_path / "ssl.ckpt"
    ssl_trainer.save_checkpoint(ckpt_path)

    # --- 2. чекпоинт-контракт §4.4: SSL-префикс распознаётся автоматически ---
    sd = torch.load(ckpt_path, weights_only=False)["state_dict"]
    _, prefix = strip_prefixes(sd)
    assert prefix == "method.backbone.net.", f"неожиданный префикс: {prefix}"

    # --- перенос бэкбона стадии SSL в чистую классификацию ---
    cls_backbone = EmbeddingBackbone(MODEL, pretrained=False)
    report = load_backbone(cls_backbone.net, ckpt_path, weights_only=False)
    assert report.prefix == "method.backbone.net."
    assert not report.missing and not report.unexpected

    # веса действительно перенеслись (совпадают с online-бэкбоном SSL)
    for k, v in method.backbone.net.state_dict().items():
        assert torch.allclose(cls_backbone.net.state_dict()[k], v)

    # --- 3. дообучение головы классификации ---
    head = LinearHead(n_class=3, embedding_dim=cls_backbone.out_dim, mode="ce")
    cls_module = ClassificationTrainer(cls_backbone, head,
                                       optimizer=partial(torch.optim.AdamW, lr=1e-3),
                                       num_classes=3, backbone_lr=1e-5)  # low-LR адаптация
    _trainer().fit(cls_module, DataLoader(SynDS(), batch_size=8),
                   DataLoader(SynDS(8), batch_size=8))

    # --- инференс с flip-TTA ---
    predictor = Predictor(cls_backbone, head, tta_views=("identity", "hflip", "vflip"),
                          image_size=32)
    logits = predictor.predict_tensor(torch.rand(4, 3, 32, 32))
    assert logits.shape == (4, 3)
    proba = torch.softmax(logits, dim=1)
    assert torch.allclose(proba.sum(dim=1), torch.ones(4), atol=1e-5)


def test_backbone_ckpt_loads_via_config_path(tmp_path):
    """EmbeddingBackbone(ckpt_path=...) — путь загрузки бэкбона из конфига."""
    src = timm.create_model(MODEL, pretrained=False, num_classes=0)
    ckpt = tmp_path / "bb.pt"
    torch.save(src.state_dict(), ckpt)
    bb = EmbeddingBackbone(MODEL, pretrained=False, ckpt_path=str(ckpt))
    for k, v in src.state_dict().items():
        assert torch.allclose(bb.net.state_dict()[k], v)
