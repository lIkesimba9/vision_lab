"""CPU-рецепты аугментаций для классификации (albumentations, ТЗ §6.1).

Одна вьюха на шаг, тяжёлая стохастика на CPU в DataLoader-воркере. Рецепты
именованы и версионированы (``heavy_v1``) — имя фиксируется в логе эксперимента.
Канонический стек классификации — albumentations (порт между стеками не
поддерживается, ТЗ §6.2).

Урок прототипа: рецепт аугментаций влияет на качество сильнее выбора лосса и
бэкбона — это полноценный объект проектирования, а не обвязка.

Вход: HWC RGB float32 [0,1] (из декодера/препроцессинга). Выход: CHW-тензор,
нормализованный ImageNet-статистикой.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _finalize(mean, std) -> list:
    return [A.Normalize(mean=mean, std=std, max_pixel_value=1.0), ToTensorV2()]


def build_classification_transform(
    recipe: str,
    image_size: int = 224,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    train: bool = True,
) -> A.Compose:
    """Собирает albumentations-Compose по имени рецепта.

    Рецепты: ``eval`` (только resize+normalize), ``light``, ``medium``,
    ``heavy_v1`` (порт «hugeron»: геометрия + blur/noise + дисторсии +
    цветовые + CLAHE). ``train=False`` всегда даёт eval-стек независимо от recipe.
    """
    if not train or recipe == "eval":
        return A.Compose([A.Resize(image_size, image_size), *_finalize(mean, std)])

    builders = {"light": _light, "medium": _medium, "heavy_v1": _heavy_v1}
    if recipe not in builders:
        raise KeyError(f"Неизвестный рецепт {recipe!r}. Доступны: {['eval', *builders]}")
    return A.Compose([*builders[recipe](image_size), *_finalize(mean, std)])


def _light(image_size: int) -> list:
    return [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]


def _medium(image_size: int) -> list:
    return [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-30, 30), shear=(-8, 8), scale=(0.9, 1.1), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(p=0.2),
    ]


def _heavy_v1(image_size: int) -> list:
    """Порт «hugeron» — самый агрессивный рецепт (главный драйвер качества)."""
    return [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.4, 1.0), ratio=(0.7, 1.43)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-45, 45), shear=(-12, 12), scale=(0.85, 1.15),
                 translate_percent=(0.0, 0.1), p=0.6),
        A.OneOf([
            A.ElasticTransform(alpha=1.0, sigma=50.0),
            A.GridDistortion(),
            A.OpticalDistortion(),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.CLAHE(clip_limit=3.0, p=0.3),
        A.CoarseDropout(p=0.3),
    ]
