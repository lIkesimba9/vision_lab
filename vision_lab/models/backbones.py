"""Обёртки бэкбонов над timm (ТЗ §4.1): три формы выхода, выбираются конфигом.

* :class:`EmbeddingBackbone` → ``(B, D)`` — пулинг-вектор (классификация, metric);
* :class:`SpatialBackbone`   → ``(B, C, H', W')`` + ``reduction`` — фичемапа
  (сегментация/детекция в перспективе);
* :class:`TokenBackbone`     → ``(pooled, tokens, grid)`` — унифицирует ViT
  (CLS+patch), Swin (channels-last) и CNN; нужен DINO/iBOT/MAE/I-JEPA.

Единое имя внутреннего атрибута — ``net`` (timm-модель). Именно оно делает
чекпоинт-контракт §4.4 простым: любой стадийный чекпоинт срезается до ``net.``.
Обёртки НЕ грузят веса сами — это делает :func:`create_timm_backbone` через
``vision_lab.core.checkpoint.load_backbone`` (единственная точка загрузки).
"""

from __future__ import annotations

from typing import NamedTuple

import timm
import torch
from torch import nn

from vision_lab.core.checkpoint import load_backbone


def create_timm_net(model_name: str, pretrained: bool = True,
                    global_pool: str | None = None, **timm_kwargs) -> nn.Module:
    """timm.create_model с ``num_classes=0`` (feature extractor).

    ``global_pool=""`` → фичемапа без пулинга (для SpatialBackbone).
    """
    kwargs = dict(timm_kwargs)
    if global_pool is not None:
        kwargs["global_pool"] = global_pool
    return timm.create_model(model_name, pretrained=pretrained, num_classes=0, **kwargs)


class EmbeddingBackbone(nn.Module):
    """timm-модель → пулинг-вектор ``(B, D)``.

    Возврат — голый тензор (не dict): полиморфизм живёт в выборе обёртки,
    а не в ключах словаря. Опциональная проекция ``embedding_dim`` (Linear+BN)
    и dropout поверх фич бэкбона.
    """

    def __init__(self, model_name: str, pretrained: bool = True,
                 embedding_dim: int | None = None, dropout: float = 0.0,
                 ckpt_path: str | None = None, **timm_kwargs):
        super().__init__()
        self.net = create_timm_net(model_name, pretrained, **timm_kwargs)
        in_features = self.net.num_features
        if ckpt_path is not None:
            load_backbone(self.net, ckpt_path)

        if embedding_dim is not None:
            self.proj = nn.Sequential(nn.Linear(in_features, embedding_dim),
                                      nn.BatchNorm1d(embedding_dim))
            self.out_dim = embedding_dim
        else:
            self.proj = nn.Identity()
            self.out_dim = in_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.net(x)))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Вектор для probe — совпадает с forward (единый интерфейс с TokenBackbone)."""
        return self.forward(x)


class SpatialBackbone(nn.Module):
    """timm-CNN → фичемапа ``(B, C, H', W')`` (без global pooling).

    Экспонирует ``out_dim`` (каналы) и ``reduction`` (фактор даунсемпла из
    timm ``feature_info``) — ровно то, что нужно smp-style декодеру.
    Опциональная 1×1-проекция каналов.
    """

    def __init__(self, model_name: str, pretrained: bool = True,
                 proj_dim: int | None = None, ckpt_path: str | None = None, **timm_kwargs):
        super().__init__()
        self.net = create_timm_net(model_name, pretrained, global_pool="", **timm_kwargs)
        in_features = self.net.num_features
        if ckpt_path is not None:
            load_backbone(self.net, ckpt_path)

        try:
            self.reduction = int(self.net.feature_info.reduction()[-1])
        except (AttributeError, TypeError):
            self.reduction = 32  # типичный CNN-даунсемпл, если timm не сообщил

        if proj_dim is not None:
            self.proj = nn.Sequential(nn.Conv2d(in_features, proj_dim, 1),
                                      nn.BatchNorm2d(proj_dim), nn.GELU())
            self.out_dim = proj_dim
        else:
            self.proj = nn.Identity()
            self.out_dim = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.net(x)
        if fmap.dim() != 4:
            raise ValueError(
                f"{type(self.net).__name__}: ожидалась 4D фичемапа, получено {tuple(fmap.shape)}. "
                "SpatialBackbone рассчитан на CNN-бэкбоны (ConvNeXt/EffNetV2/...)."
            )
        return self.proj(fmap)


class TokenOutput(NamedTuple):
    pooled: torch.Tensor    # (B, D) — глобальный вектор
    tokens: torch.Tensor    # (B, N, D) — patch-токены
    grid: tuple[int, int]   # (H', W') — сетка токенов


class TokenBackbone(nn.Module):
    """Унифицированный токен-выход для ViT / Swin / CNN (порт логики из прототипа).

    ``forward`` → :class:`TokenOutput` (pooled, tokens, grid). ``embed`` даёт
    pooled-вектор для probe — та же timm-модель, без двойного оборачивания.
    ``grid_size(hw)`` считает сетку токенов ДО forward (нужно для input-space
    масок iBOT/SimMIM/MAE/I-JEPA).
    """

    def __init__(self, model_name: str, pretrained: bool = True,
                 ckpt_path: str | None = None, **timm_kwargs):
        super().__init__()
        self.net = create_timm_net(model_name, pretrained, global_pool="", **timm_kwargs)
        self.out_dim = self.net.num_features
        self.num_prefix_tokens = int(getattr(self.net, "num_prefix_tokens", 0))
        if ckpt_path is not None:
            load_backbone(self.net, ckpt_path)

    def grid_size(self, image_hw: tuple[int, int]) -> tuple[int, int]:
        """Сетка токенов для входного размера (из patch_embed / reduction)."""
        h, w = image_hw
        pe = getattr(self.net, "patch_embed", None)
        if pe is not None and hasattr(pe, "patch_size"):
            ps = pe.patch_size
            ps = ps if isinstance(ps, tuple) else (ps, ps)
            return h // ps[0], w // ps[1]
        try:
            red = int(self.net.feature_info.reduction()[-1])
        except (AttributeError, TypeError):
            red = 32
        return h // red, w // red

    def forward(self, x: torch.Tensor) -> TokenOutput:
        feats = self.net.forward_features(x)
        h, w = self.grid_size((x.shape[-2], x.shape[-1]))

        if feats.dim() == 3:                       # ViT: (B, prefix+N, D)
            tokens = feats[:, self.num_prefix_tokens:, :]
            pooled = feats[:, 0, :] if self.num_prefix_tokens else tokens.mean(dim=1)
        elif feats.dim() == 4:
            if feats.shape[1] == self.out_dim:     # CNN: (B, C, H, W)
                b, c, hh, ww = feats.shape
                h, w = hh, ww
                tokens = feats.flatten(2).transpose(1, 2)
            else:                                  # Swin channels-last: (B, H, W, C)
                b, hh, ww, c = feats.shape
                h, w = hh, ww
                tokens = feats.reshape(b, hh * ww, c)
            pooled = tokens.mean(dim=1)
        else:
            raise ValueError(f"Неожиданная форма forward_features: {tuple(feats.shape)}")

        # выравниваем grid по фактическому числу токенов (робастно к округлениям)
        if h * w != tokens.shape[1]:
            h = w = int(round(tokens.shape[1] ** 0.5))
        return TokenOutput(pooled=pooled, tokens=tokens, grid=(h, w))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).pooled
