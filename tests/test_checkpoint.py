import pytest
import torch
from torch import nn

from vision_lab.core.checkpoint import (
    BACKBONE_PREFIXES,
    extract_fc_weights,
    load_backbone,
    strip_prefixes,
)


def make_net():
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def prefixed(sd, prefix):
    return {prefix + k: v for k, v in sd.items()}


@pytest.mark.parametrize("prefix", BACKBONE_PREFIXES)
def test_every_registry_prefix_strips(prefix):
    net = make_net()
    sd = prefixed(net.state_dict(), prefix)
    stripped, matched = strip_prefixes(sd)
    assert matched == prefix
    assert stripped.keys() == net.state_dict().keys()


def test_bare_state_dict_passes_through():
    sd = make_net().state_dict()
    stripped, matched = strip_prefixes(sd)
    assert matched is None
    assert stripped.keys() == sd.keys()


def test_byol_checkpoint_prefers_online_over_teacher():
    net = make_net()
    online = prefixed(net.state_dict(), "method.backbone.net.")
    teacher = prefixed(net.state_dict(), "method.teacher.module.backbone.net.")
    _, matched = strip_prefixes({**teacher, **online})
    assert matched == "method.backbone.net."


def test_dino_checkpoint_prefers_teacher_over_student():
    net = make_net()
    student = prefixed(net.state_dict(), "method.student.net.")
    teacher = prefixed(net.state_dict(), "method.teacher.module.net.")
    _, matched = strip_prefixes({**student, **teacher})
    assert matched == "method.teacher.module.net."


def test_load_backbone_from_lightning_file(tmp_path):
    src, dst = make_net(), make_net()
    ckpt = {
        "state_dict": prefixed(src.state_dict(), "backbone.net."),
        "epoch": 3,
        "global_step": 100,
    }
    path = tmp_path / "epoch3.ckpt"
    torch.save(ckpt, path)

    report = load_backbone(dst, path)  # weights_only=True по умолчанию
    assert report.prefix == "backbone.net."
    assert not report.missing and not report.unexpected
    for a, b in zip(dst.parameters(), src.parameters()):
        assert torch.equal(a, b)


def test_extract_fc_weights_from_tensor_dict_and_file(tmp_path):
    w = torch.randn(5, 8)
    b = torch.randn(5)

    got_w, got_b = extract_fc_weights(w, 5, 8)
    assert torch.equal(got_w, w) and got_b is None

    got_w, got_b = extract_fc_weights({"fc.weight": w, "fc.bias": b}, 5, 8)
    assert torch.equal(got_w, w) and torch.equal(got_b, b)

    # Lightning-чекпоинт: ключ с нестандартным именем, ищется по форме
    ckpt = {"state_dict": {"model.head.proto.weight": w}}
    got_w, got_b = extract_fc_weights(ckpt, 5, 8)
    assert torch.equal(got_w, w) and got_b is None

    path = tmp_path / "fc.pt"
    torch.save({"weight": w, "bias": b}, path)
    got_w, got_b = extract_fc_weights(path, 5, 8)
    assert torch.equal(got_w, w) and torch.equal(got_b, b)


def test_extract_fc_weights_wrong_shape_raises():
    with pytest.raises(ValueError):
        extract_fc_weights(torch.randn(3, 3), 5, 8)
    with pytest.raises(KeyError):
        extract_fc_weights({"fc.weight": torch.randn(3, 3)}, 5, 8)
