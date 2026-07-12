"""CPU-аугментации (albumentations) для классификации."""

from vision_lab.data.transforms.classification import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_classification_transform,
)

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "build_classification_transform"]
