import pytest
from torch import nn

from vision_lab.core.optim import default_no_decay, param_groups


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_no_decay_covers_bias_and_norm():
    m = Toy()
    named = dict(m.named_parameters())
    assert default_no_decay("fc.bias", named["fc.bias"])
    assert default_no_decay("norm.weight", named["norm.weight"])  # 1D
    assert not default_no_decay("fc.weight", named["fc.weight"])


def test_param_groups_names_lrs_and_decay_split():
    backbone, head = Toy(), Toy()
    groups = param_groups(
        {"backbone": backbone, "head": head},
        base_lr=1e-3,
        weight_decay=0.05,
        lr_overrides={"backbone": 1e-5},
    )
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"backbone.decay", "backbone.no_decay", "head.decay", "head.no_decay"}
    assert by_name["backbone.decay"]["lr"] == 1e-5
    assert by_name["head.decay"]["lr"] == 1e-3
    assert by_name["backbone.no_decay"]["weight_decay"] == 0.0
    assert by_name["head.decay"]["weight_decay"] == 0.05
    # fc.weight в decay, всё остальное (bias, norm) в no_decay
    assert len(by_name["head.decay"]["params"]) == 1
    assert len(by_name["head.no_decay"]["params"]) == 3


def test_param_groups_skips_frozen_and_raises_on_empty():
    m = Toy()
    for p in m.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError):
        param_groups({"m": m}, base_lr=1e-3)
