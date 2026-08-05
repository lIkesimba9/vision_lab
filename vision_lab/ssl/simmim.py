"""SimMIM (Xie 2021, https://arxiv.org/abs/2111.09886) — простейший masked
image modeling: маскируем блоки входа, предсказываем пиксели замаскированных
патчей линейной головой, L1 только на маске.

В отличие от MAE энкодер видит ПОЛНУЮ сетку (маскированные блоки занулены на
входе) — поэтому работает с любым бэкбоном, дающим токены с равномерной
сеткой (ViT, Swin, CNN через :class:`~vision_lab.models.backbones.TokenBackbone`);
этим SimMIM претрейнился SwinV2-G. Плата — нет MAE-экономии на выкидывании
токенов.

Маска выравнивается по ВЫХОДНОЙ сетке токенов (:func:`components.block_mask`,
та же механика, что у iBOT-ветки DINOv2). Без учителя (``momentum_update`` —
no-op). ``extract_embeddings`` — pooled-вектор бэкбона без маски.
"""

from __future__ import annotations

import torch
from torch import nn

from vision_lab.models.backbones import TokenBackbone
from vision_lab.ssl.base import SSLMethod
from vision_lab.ssl.components import block_mask, patchify


class SimMIM(SSLMethod):
    """SimMIM: block-маска входа + линейное предсказание пикселей патча.

    Компоненты передаются инстанцированными:
        ``backbone`` — :class:`TokenBackbone` (ViT/Swin/CNN);
        ``views``    — :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment`
                       (1 вьюха, рецепт ``mim_v1`` — только crop+flip).

    ``patch_px`` — сторона пиксельного патча одного токена = image_size //
    grid (задаётся явно, чтобы голова имела фиксированный выход).
    """

    def __init__(self, backbone: TokenBackbone, views: nn.Module,
                 patch_px: int, mask_ratio: float = 0.6):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio={mask_ratio} должен быть в (0, 1)")
        self.backbone = backbone
        self.views = views
        self.mask_ratio = mask_ratio
        self.patch_px = patch_px
        self.head = nn.Linear(backbone.out_dim, patch_px * patch_px * 3)

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled-вектор бэкбона на НЕмаскированном входе."""
        return self.backbone.embed(images)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x = self.views(batch["image"]).globals[0]
        grid = self.backbone.grid_size((x.shape[-2], x.shape[-1]))
        if x.shape[-2] // grid[0] != self.patch_px:
            raise ValueError(
                f"patch_px={self.patch_px} не совпадает с фактическим "
                f"{x.shape[-2]}//{grid[0]}={x.shape[-2] // grid[0]} — проверьте image_size вьюх"
            )

        x_masked, mask = block_mask(x, grid, self.mask_ratio)
        out = self.backbone(x_masked)
        if out.tokens.shape[1] != mask.shape[1]:
            raise ValueError(
                f"Число токенов {out.tokens.shape[1]} != размеру маски {mask.shape[1]} "
                f"(сетка {grid}) — бэкбон без равномерной сетки не поддержан"
            )

        pred = self.head(out.tokens)                      # (B, N, ph*ph*3)
        target = patchify(x, grid)                        # пиксели ОРИГИНАЛЬНОЙ вьюхи
        loss_per_patch = (pred - target).abs().mean(dim=-1)
        m = mask.float()
        loss = (loss_per_patch * m).sum() / m.sum().clamp(min=1.0)
        return {"simmim_loss": loss, "total_loss": loss}
