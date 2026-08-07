"""Бэкбоны: три формы выхода поверх timm (§4.1)."""

from vision_lab.models.backbones import (
    EmbeddingBackbone,
    SpatialBackbone,
    TokenBackbone,
    TokenOutput,
    create_timm_net,
)
from vision_lab.models.lora import (
    DEFAULT_TARGETS,
    LoRALinear,
    apply_lora,
    merge_lora,
    trainable_parameters,
)

__all__ = [
    "EmbeddingBackbone",
    "SpatialBackbone",
    "TokenBackbone",
    "TokenOutput",
    "create_timm_net",
    "LoRALinear",
    "apply_lora",
    "merge_lora",
    "trainable_parameters",
    "DEFAULT_TARGETS",
]
