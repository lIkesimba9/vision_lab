import pytest
import torch
from torch import nn

from vision_lab.models import (
    LoRALinear,
    apply_lora,
    merge_lora,
    trainable_parameters,
)


class TinyViTLike(nn.Module):
    """Мини-модель с именами слоёв как у ViT из timm."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(d, 3 * d),
                "proj": nn.Linear(d, d),
                "fc1": nn.Linear(d, 2 * d),
                "fc2": nn.Linear(2 * d, d),
            })
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        for b in self.blocks:
            x = b["proj"](b["qkv"](x)[..., : x.shape[-1]])
            x = b["fc2"](b["fc1"](x))
        return self.norm(x)


def test_adapters_start_as_identity():
    """B инициализируется нулями: выход до обучения совпадает с исходным."""
    torch.manual_seed(0)
    model = TinyViTLike().eval()
    x = torch.randn(4, 16)
    before = model(x)

    apply_lora(model, r=4, alpha=8)
    after = model.eval()(x)
    torch.testing.assert_close(before, after)


def test_only_adapters_are_trainable():
    model = TinyViTLike(d=256)
    apply_lora(model, r=4)
    trainable, total = trainable_parameters(model)

    assert 0 < trainable < total
    assert trainable / total < 0.05
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names)


def test_trainable_share_shrinks_as_model_widens():
    """Свойство низкорангового адаптера: чем шире слои, тем дешевее адаптация."""
    shares = []
    for d in (64, 256, 1024):
        model = TinyViTLike(d=d)
        apply_lora(model, r=4)
        trainable, total = trainable_parameters(model)
        shares.append(trainable / total)
    assert shares[0] > shares[1] > shares[2]


def test_wraps_expected_layers_and_counts_them():
    model = TinyViTLike()
    n = apply_lora(model, r=2, target_modules=("qkv", "fc1"))
    assert n == 4  # два блока × два целевых слоя
    assert isinstance(model.blocks[0]["qkv"], LoRALinear)
    assert isinstance(model.blocks[0]["proj"], nn.Linear)
    assert not isinstance(model.blocks[0]["proj"], LoRALinear)


def test_unknown_targets_raise_instead_of_silently_doing_nothing():
    model = TinyViTLike()
    with pytest.raises(ValueError, match="ни один слой не подошёл"):
        apply_lora(model, target_modules=("attention_that_does_not_exist",))


def test_merge_restores_plain_linear_and_keeps_output():
    torch.manual_seed(1)
    model = TinyViTLike()
    apply_lora(model, r=4, alpha=8)
    # сдвигаем адаптеры от нуля, иначе проверка вырождается в тождество
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)

    model.eval()
    x = torch.randn(4, 16)
    before = model(x)

    merged = merge_lora(model)
    assert merged == 8
    assert isinstance(model.blocks[0]["qkv"], nn.Linear)
    torch.testing.assert_close(before, model(x), rtol=1e-4, atol=1e-5)


def test_param_groups_picks_up_only_adapters():
    """Трейнер собирает группы через param_groups — тот пропускает замороженное."""
    from vision_lab.core.optim import param_groups

    model = TinyViTLike()
    apply_lora(model, r=4)
    head = nn.Linear(16, 2)

    groups = param_groups({"backbone": model, "head": head},
                          base_lr=1e-3, lr_overrides={"backbone": 1e-4})
    picked = {id(p) for g in groups for p in g["params"]}
    adapters = {id(p) for n, p in model.named_parameters() if "lora_" in n}
    frozen = {id(p) for n, p in model.named_parameters() if "lora_" not in n}

    assert adapters <= picked
    assert not (frozen & picked)


def test_rejects_non_positive_rank():
    with pytest.raises(ValueError, match="r должен быть положительным"):
        LoRALinear(nn.Linear(4, 4), r=0)
