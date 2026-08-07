"""LoRA-адаптеры для бэкбонов (§4.1, models.md «Linear probe → LoRA → файнтюн»).

Зачем. Плейбук ставит PEFT средней ступенью лестницы между linear probe и полным
файнтюном: «по точности не хуже файнтюна, не лучше», но обучаемых параметров
около процента. Три ситуации, где это решает:

* **большой бэкбон при малых данных** — полный файнтюн ViT-L на нескольких
  тысячах примеров переобучается, linear probe недобирает;
* **много задач на один бэкбон** — общие веса плюс дешёвый адаптер на задачу
  вместо копии модели на каждую;
* **нехватка памяти** — оптимизатор хранит моменты только для адаптеров, что
  освобождает место под больший батч или большее разрешение входа.

Третий пункт — типичный повод при адаптации к разрешению, на которое весов не
выпускали: у DINOv3, например, все чекпоинты идут только в 256, и подъём до 384
и выше возможен лишь дообучением с интерполяцией позиционных эмбеддингов.

Устройство. :class:`LoRALinear` оборачивает ``nn.Linear``, оставляя исходную
матрицу замороженной, и добавляет к ней низкоранговую поправку
``B @ A * (alpha / r)``. ``B`` инициализируется нулями, поэтому **сразу после
установки адаптеров модель считает ровно то же, что и до неё** — обучение
стартует не с испорченных фич.

Совместимость с трейнером бесплатная:
:func:`~vision_lab.core.optim.param_groups` пропускает параметры с
``requires_grad=False``, так что в оптимизатор попадут только адаптеры и голова.

Пример::

    from vision_lab.models import EmbeddingBackbone, apply_lora

    backbone = EmbeddingBackbone("vit_large_patch16_dinov3.lvd1689m", img_size=384)
    n = apply_lora(backbone.net, r=16, alpha=32)   # -> сколько слоёв обёрнуто
    # дальше как обычно: ClassificationTrainer(backbone, head, ...)

Перед экспортом модели адаптеры схлопываются в исходные веса
(:func:`merge_lora`) — инференс не платит ни временем, ни зависимостью от этого
модуля.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
from torch import nn

#: Имена Linear-слоёв, которые адаптируются по умолчанию.
#: Покрывают ViT-семейство timm (``qkv``/``proj`` во внимании, ``fc1``/``fc2`` в MLP)
#: и раздельные проекции внимания у EVA-02 и подобных.
DEFAULT_TARGETS = ("qkv", "proj", "fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """``nn.Linear`` с низкоранговой поправкой; исходные веса заморожены.

    ``y = base(x) + dropout(x) @ A^T @ B^T * (alpha / r)``

    ``A`` инициализируется по Каймингу, ``B`` — нулями: поправка стартует с нуля,
    и выход совпадает с выходом исходного слоя.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"r должен быть положительным, получено {r}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        w = base.weight
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, dtype=w.dtype, device=w.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, dtype=w.dtype, device=w.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_dropout(x) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + delta * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Исходный ``nn.Linear`` с вплавленной поправкой (для инференса/экспорта)."""
        self.base.weight.add_((self.lora_b @ self.lora_a) * self.scaling)
        self.base.weight.requires_grad_(True)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(True)
        return self.base

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}"


def _walk(model: nn.Module, cls) -> Iterator[tuple[nn.Module, str, nn.Module]]:
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, cls):
                yield parent, name, child


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Iterable[str] = DEFAULT_TARGETS,
    freeze_base: bool = True,
) -> int:
    """Оборачивает подходящие ``nn.Linear`` в :class:`LoRALinear`; возвращает их число.

    ``target_modules`` сверяется с ИМЕНЕМ слоя внутри родителя (``qkv``, ``fc1``, …),
    а не с полным путём — так один и тот же набор имён работает для любой глубины.

    ``freeze_base=True`` дополнительно замораживает все остальные параметры
    модели: остаются обучаемыми только адаптеры. Голова живёт вне бэкбона и этим
    не затрагивается.

    Ошибка, если не нашлось ни одного слоя: молча обучать нечего — это почти
    всегда опечатка в ``target_modules`` под незнакомую архитектуру.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    targets = set(target_modules)
    wrapped = 0
    for parent, name, child in _walk(model, nn.Linear):
        if name in targets:
            setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            wrapped += 1

    if wrapped == 0:
        raise ValueError(
            f"ни один слой не подошёл под target_modules={sorted(targets)}. "
            "Проверьте имена подмодулей: "
            f"{sorted({n for _, n, _ in _walk(model, nn.Linear)})[:12]}"
        )
    return wrapped


def merge_lora(model: nn.Module) -> int:
    """Вплавляет все адаптеры обратно в ``nn.Linear``; возвращает их число."""
    merged = 0
    for parent, name, child in _walk(model, LoRALinear):
        setattr(parent, name, child.merged())
        merged += 1
    return merged


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """``(обучаемых, всего)`` — для лога и для проверки, что заморозка сработала."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
