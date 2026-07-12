"""Единая точка инференса (ТЗ §5.5): load_module → transform → predict.

Собирает предсказатель из компонентов классификации (backbone + head) и
eval-трансформа, с опциональным flip-TTA. Один и тот же путь для любого
чекпоинта: ``head.predict_logits(backbone(image))``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vision_lab.data.decoders import decode_image
from vision_lab.data.transforms import build_classification_transform
from vision_lab.inference.tta import tta_logits


class Predictor:
    """Инференс поверх обученных backbone + head.

    Параметры:
        backbone, head — инстанцированные компоненты (веса уже загружены);
        transform — albumentations eval-трансформ (по умолчанию resize+normalize);
        tta_views — симметрии для усреднения логитов ((), чтобы выключить TTA);
        device — куда переносить модель.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, transform=None,
                 tta_views: tuple[str, ...] = ("identity", "hflip"),
                 device: str = "cpu", image_size: int = 224):
        self.backbone = backbone.to(device).eval()
        self.head = head.to(device).eval()
        self.transform = transform or build_classification_transform(
            "eval", image_size=image_size, train=False)
        self.tta_views = tta_views
        self.device = device

    def _logits_fn(self, images: torch.Tensor) -> torch.Tensor:
        return self.head.predict_logits(self.backbone(images))

    @torch.no_grad()
    def predict_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Логиты для батча тензоров ``(B, 3, H, W)`` с TTA."""
        images = images.to(self.device)
        if not self.tta_views:
            return self._logits_fn(images)
        return tta_logits(self._logits_fn, images, self.tta_views)

    @torch.no_grad()
    def predict_paths(self, paths: Sequence[str | Path], batch_size: int = 32) -> torch.Tensor:
        """Логиты для списка путей к изображениям (decode → transform → predict)."""
        all_logits = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            tensors = [self.transform(image=decode_image(p))["image"] for p in chunk]
            all_logits.append(self.predict_tensor(torch.stack(tensors)).cpu())
        return torch.cat(all_logits) if all_logits else torch.empty(0)

    def predict_proba(self, paths: Sequence[str | Path], batch_size: int = 32) -> np.ndarray:
        return torch.softmax(self.predict_paths(paths, batch_size), dim=1).numpy()
