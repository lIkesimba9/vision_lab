"""Смоук экспериментальных голов — изолированы от боевых (§3.1), но должны работать."""

import torch

from vision_lab.experimental.proto import ProtoHead
from vision_lab.experimental.speaker import SpeakerHead

N, D, C = 16, 8, 4


def pk_targets():
    # PK-структура: 4 класса по 4 примера (позитивы гарантированы)
    labels = torch.arange(C).repeat_interleave(4)
    return {"label": labels}


def test_proto_head_contract():
    torch.manual_seed(0)
    emb = torch.randn(N, D, requires_grad=True)
    head = ProtoHead(C, D, k=3)
    out = head(emb, pk_targets())
    assert {"total_loss", "ce", "clst", "sep"} <= set(out)
    out["total_loss"].backward()
    assert head.predict_logits(emb.detach()).shape == (N, C)


def test_speaker_head_contract():
    torch.manual_seed(1)
    emb = torch.randn(N, D, requires_grad=True)
    head = SpeakerHead(C, D)
    out = head(emb, pk_targets())
    assert {"total_loss", "aam_loss", "sv_loss"} <= set(out)
    assert torch.isfinite(out["total_loss"])
    out["total_loss"].backward()
    assert head.predict_logits(emb.detach()).shape == (N, C)


def test_experimental_not_imported_by_production():
    """experimental не должен подтягиваться при импорте боевых пакетов (§3.1)."""
    import sys

    for mod in list(sys.modules):
        if mod.startswith("vision_lab.experimental"):
            del sys.modules[mod]
    import vision_lab.heads  # noqa: F401
    import vision_lab.ssl  # noqa: F401

    assert not any(m.startswith("vision_lab.experimental") for m in sys.modules)
