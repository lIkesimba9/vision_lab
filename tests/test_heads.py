import math

import pytest
import torch

from vision_lab.core.batch import MISSING_LABEL
from vision_lab.heads import (
    AAMHead,
    AAMTripletHead,
    CosineCEHead,
    DBMHead,
    FocalHead,
    LDAMHead,
    LinearHead,
    LogitAdjustHead,
    MultiTaskHead,
    SeesawHead,
    SubCenterHead,
    VSHead,
)
from vision_lab.heads import primitives as P

N, D, C = 16, 8, 4
COUNTS = [50, 30, 12, 8]


def batch(target_key="label", n=N):
    torch.manual_seed(0)
    emb = torch.randn(n, D, requires_grad=True)
    labels = torch.randint(0, C, (n,))
    return emb, {target_key: labels}


ALL_HEADS = [
    lambda: LinearHead(C, D, mode="ce"),
    lambda: LinearHead(C, D, mode="bce", class_counts=COUNTS),
    lambda: LinearHead(C, D, mode="balanced_softmax", class_counts=COUNTS),
    lambda: CosineCEHead(C, D),
    lambda: AAMHead(C, D),
    lambda: SubCenterHead(C, D, k=3),
    lambda: LDAMHead(C, D, class_counts=COUNTS, class_balanced=True),
    lambda: FocalHead(C, D, class_counts=COUNTS, cb_beta=0.99),
    lambda: LogitAdjustHead(C, D, class_counts=COUNTS),
    lambda: VSHead(C, D, class_counts=COUNTS),
    lambda: SeesawHead(C, D),
    lambda: DBMHead(C, D, class_counts=COUNTS),
    lambda: AAMTripletHead(C, D),
]


@pytest.mark.parametrize("make", ALL_HEADS, ids=lambda f: f().__class__.__name__)
def test_head_contract_forward_and_predict(make):
    head = make()
    emb, targets = batch()
    out = head(emb, targets)
    assert "total_loss" in out
    loss = out["total_loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()  # градиент проходит

    logits = head.predict_logits(emb.detach())
    assert logits.shape == (N, C)
    assert head.classifier_weight.shape == (C, D)


@pytest.mark.parametrize("make", ALL_HEADS, ids=lambda f: f().__class__.__name__)
def test_head_masks_unlabeled(make):
    """Строки с меткой -1 не должны ронять лосс (маскирование §5.3)."""
    head = make()
    emb, targets = batch()
    targets["label"][:4] = MISSING_LABEL
    loss = head(emb, targets)["total_loss"]
    assert torch.isfinite(loss)


def test_all_unlabeled_gives_zero_loss():
    head = LinearHead(C, D, mode="ce")
    emb, targets = batch()
    targets["label"][:] = MISSING_LABEL
    loss = head(emb, targets)["total_loss"]
    assert loss.item() == 0.0


# --- сверка формул примитивов ---------------------------------------------
def test_aam_margin_matches_arcface_formula():
    torch.manual_seed(1)
    emb = torch.randn(3, D)
    weight = torch.randn(C, D)
    target = torch.tensor([0, 1, 2])
    m = 0.3
    got = P.additive_angular_margin(P.cosine_logits(emb, weight), target, m)

    # ручной ArcFace на истинном классе для строки 0
    cos = torch.nn.functional.linear(
        torch.nn.functional.normalize(emb), torch.nn.functional.normalize(weight))
    c = cos[0, 0].item()
    s = math.sqrt(max(1 - c * c, 0))
    expected = c * math.cos(m) - s * math.sin(m)
    assert got[0, 0].item() == pytest.approx(expected, abs=1e-5)
    # нецелевые классы не тронуты
    assert got[0, 1].item() == pytest.approx(cos[0, 1].item(), abs=1e-6)


def test_ldam_margins_scale_and_order():
    margins = P.ldam_margins(COUNTS, max_m=0.5)
    assert margins.max().item() == pytest.approx(0.5)
    # редкий класс (меньше примеров) -> больший маржин
    assert margins[3] > margins[0]


def test_class_balanced_weights_normalized():
    w = P.class_balanced_weights(COUNTS, beta=0.99)
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)
    assert w[3] > w[0]  # редкий класс — больший вес


def test_subcenter_reduce_takes_max():
    cos = torch.tensor([[0.1, 0.9, 0.2, 0.3]])  # C=2, k=2
    reduced = P.subcenter_reduce(cos, n_class=2, k=2)
    assert reduced.squeeze().tolist() == pytest.approx([0.9, 0.3])


def test_multitask_head_composition():
    torch.manual_seed(2)
    emb = torch.randn(N, D, requires_grad=True)
    targets = {"label": torch.randint(0, C, (N,)), "label_aux": torch.randint(0, 3, (N,))}
    targets["label_aux"][:5] = MISSING_LABEL  # часть без aux-метки
    head = MultiTaskHead(
        heads={
            "main": LinearHead(C, D, mode="ce", target_key="label"),
            "aux": LinearHead(3, D, mode="ce", target_key="label_aux"),
        },
        primary="main",
        weights={"aux": 0.3},
    )
    out = head(emb, targets)
    assert {"total_loss", "main_loss", "aux_loss"} <= set(out)
    out["total_loss"].backward()
    assert head.predict_logits(emb.detach()).shape == (N, C)


def test_fc_weight_transfer_between_stages(tmp_path):
    src = AAMHead(C, D)
    path = tmp_path / "fc.pt"
    src.save_fc_weights(path)
    dst = LinearHead(C, D, mode="ce")
    dst.load_fc_weights(path)
    assert torch.allclose(dst.classifier_weight, src.classifier_weight)
