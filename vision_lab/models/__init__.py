"""Бэкбоны: три формы выхода поверх timm (§4.1)."""

from vision_lab.models.backbones import (
    EmbeddingBackbone,
    SpatialBackbone,
    TokenBackbone,
    TokenOutput,
    create_timm_net,
)

__all__ = [
    "EmbeddingBackbone",
    "SpatialBackbone",
    "TokenBackbone",
    "TokenOutput",
    "create_timm_net",
]
