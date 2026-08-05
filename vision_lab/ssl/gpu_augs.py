"""GPU-аугментации для SSL (kornia, ТЗ §6.1–6.3).

Почему на GPU и внутри SSL-модуля: нужны N *независимо* аугментированных вьюх
одного тензора. CPU-augs дали бы всем вьюхам одинаковую аугментацию и убили бы
контраст. CPU-путь для SSL минимален (decode → resize → tensor), вся стохастика
и Normalize — здесь.

Инварианты порядка — в КОДЕ, а не в комментариях YAML (§6.3):

* ``Normalize`` — строго последним (после него значения вне [0,1]);
* стохастические value-range операции (шум, соляризация, blur) — ДО Normalize,
  иначе device-assert в gamma/CLAHE/grayscale и порча статистик.

:func:`build_ssl_views` валидирует порядок assert'ом и делает коэрцию
Hydra-списков (ListConfig ломает interpolate/kornia) — контракт нельзя
нарушить правкой YAML.
"""

from __future__ import annotations

from collections.abc import Sequence

import kornia.augmentation as K
import torch
from torch import nn

from vision_lab.data.transforms.classification import IMAGENET_MEAN, IMAGENET_STD

#: Операции, которые обязаны стоять до Normalize (работают в шкале [0,1]).
_VALUE_RANGE_OPS = (
    K.RandomGaussianNoise,
    K.RandomSolarize,
    K.RandomGaussianBlur,
    K.RandomMotionBlur,
    K.ColorJitter,
    K.RandomGrayscale,
    K.RandomChannelShuffle,
)


class ViewSet(nn.Module):
    """Результат :class:`MultiViewAugment`: списки глобальных и локальных вьюх.

    Не NamedTuple, чтобы .to(device)/state_dict не требовались; доступ по
    ``.globals`` / ``.locals`` (списки тензоров).
    """

    def __init__(self, globals_: list[torch.Tensor], locals_: list[torch.Tensor]):
        super().__init__()
        self.globals = globals_
        self.locals = locals_


def _coerce_size(size) -> tuple[int, int]:
    """OmegaConf ListConfig / int -> (H, W) из нативных int (иначе kornia падает)."""
    if isinstance(size, int):
        return (size, size)
    return tuple(int(s) for s in size)


def _assert_order(pipeline: nn.Module) -> None:
    """Проверяет, что Normalize последний и value-range ops стоят до него."""
    ops = list(pipeline) if isinstance(pipeline, (nn.Sequential, K.AugmentationSequential)) \
        else list(pipeline.children())
    norm_positions = [i for i, op in enumerate(ops) if isinstance(op, K.Normalize)]
    if len(norm_positions) > 1:
        raise ValueError("В пайплайне вьюхи больше одного Normalize")
    if not norm_positions:
        return
    norm_i = norm_positions[0]
    if norm_i != len(ops) - 1:
        raise ValueError(
            f"Normalize должен быть последним (позиция {norm_i} из {len(ops) - 1}); "
            "value-range операции обязаны стоять до него (§6.3)"
        )
    for i, op in enumerate(ops):
        if i > norm_i and isinstance(op, _VALUE_RANGE_OPS):
            raise ValueError(f"{type(op).__name__} стоит после Normalize (§6.3)")


def build_view_pipeline(
    image_size,
    scale: tuple[float, float] = (0.3, 1.0),
    blur_p: float = 1.0,
    solarize_p: float = 0.0,
    grayscale_p: float = 0.2,
    color_jitter_p: float = 0.8,
    hflip_p: float = 0.5,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> K.AugmentationSequential:
    """Один стек вьюхи. Порядок фиксирован: crop/flip → color → blur/solarize →
    Normalize. Асимметрия BYOL (разный blur/solarize по вьюхам) задаётся
    параметрами при сборке двух пайплайнов.
    """
    size = _coerce_size(image_size)
    # ядро blur ~10% размера вьюхи, нечётное и строго меньше вьюхи (иначе pad > input)
    k = max(3, int(0.1 * min(size)) | 1)
    k = min(k, min(size) - 1 if min(size) % 2 == 0 else min(size) - 2)
    ops: list[nn.Module] = [
        K.RandomResizedCrop(size=size, scale=scale, same_on_batch=False),
        K.RandomHorizontalFlip(p=hflip_p),
    ]
    if color_jitter_p > 0:
        ops.append(K.ColorJitter(0.4, 0.4, 0.2, 0.1, p=color_jitter_p))
    if grayscale_p > 0:
        ops.append(K.RandomGrayscale(p=grayscale_p))
    if blur_p > 0:
        ops.append(K.RandomGaussianBlur((k, k), (0.1, 2.0), p=blur_p))
    if solarize_p > 0:
        ops.append(K.RandomSolarize(0.1, p=solarize_p))
    ops.append(K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    pipeline = K.AugmentationSequential(*ops, same_on_batch=False)
    _assert_order(pipeline)
    return pipeline


class MultiViewAugment(nn.Module):
    """N независимых вьюх одного батча (§6.1). Универсальна для BYOL/DINO/MAE.

    ``global_views`` — список стеков (len>=1). Асимметричный BYOL = 2 разных
    стека; DINO = 2 глобальных. ``local_view`` + ``n_local`` — multi-crop
    (локальные вьюхи меньшего размера).
    """

    def __init__(self, global_views: Sequence[nn.Module],
                 local_view: nn.Module | None = None, n_local: int = 0):
        super().__init__()
        if len(global_views) < 1:
            raise ValueError("Нужен хотя бы один global view")
        self.global_views = nn.ModuleList(global_views)
        self.local_view = local_view
        self.n_local = n_local if local_view is not None else 0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> ViewSet:
        globals_ = [aug(x) for aug in self.global_views]
        locals_ = [self.local_view(x) for _ in range(self.n_local)] if self.local_view else []
        return ViewSet(globals_, locals_)


def build_ssl_views(
    recipe: str,
    image_size=224,
    local_size=96,
    n_local: int = 0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> MultiViewAugment:
    """Именованные+версионированные рецепты SSL-вьюх.

    * ``byol_v1`` — 2 асимметричных глобальных вьюхи (BYOL/MoCo v3):
      вьюха-1 с blur, вьюха-2 с blur+solarize.
    * ``dino_v1`` — 2 глобальных (crop 0.4–1.0) + ``n_local`` локальных
      (crop 0.05–0.4, меньший размер).
    * ``simclr_v1`` — 2 СИММЕТРИЧНЫЕ вьюхи (Chen 2020: crop 0.2–1.0, jitter,
      grayscale, blur 0.5) — SimCLR/SimSiam.
    * ``mim_v1`` — 1 вьюха, только crop 0.2–1.0 + flip (MAE/SimMIM:
      маскирование заменяет тяжёлые аугментации).
    """
    if recipe == "byol_v1":
        view1 = build_view_pipeline(image_size, scale=(0.3, 1.0), blur_p=1.0,
                                    solarize_p=0.0, mean=mean, std=std)
        view2 = build_view_pipeline(image_size, scale=(0.3, 1.0), blur_p=0.1,
                                    solarize_p=0.2, mean=mean, std=std)
        return MultiViewAugment([view1, view2])

    if recipe == "dino_v1":
        g1 = build_view_pipeline(image_size, scale=(0.4, 1.0), blur_p=1.0, mean=mean, std=std)
        g2 = build_view_pipeline(image_size, scale=(0.4, 1.0), blur_p=0.1,
                                 solarize_p=0.2, mean=mean, std=std)
        local = build_view_pipeline(local_size, scale=(0.05, 0.4), blur_p=0.5,
                                    mean=mean, std=std)
        return MultiViewAugment([g1, g2], local_view=local, n_local=n_local)

    if recipe == "simclr_v1":
        views = [build_view_pipeline(image_size, scale=(0.2, 1.0), blur_p=0.5,
                                     mean=mean, std=std) for _ in range(2)]
        return MultiViewAugment(views)

    if recipe == "mim_v1":
        view = build_view_pipeline(image_size, scale=(0.2, 1.0), blur_p=0.0,
                                   color_jitter_p=0.0, grayscale_p=0.0,
                                   mean=mean, std=std)
        return MultiViewAugment([view])

    raise KeyError(
        f"Неизвестный SSL-рецепт вьюх {recipe!r}. "
        "Доступны: byol_v1, dino_v1, simclr_v1, mim_v1"
    )
