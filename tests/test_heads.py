import math

import pytest
import torch

from vision_lab.core.batch import MISSING_LABEL
from vision_lab.data.taxonomy import Taxonomy
from vision_lab.heads import (
    AAMHead,
    AAMTripletHead,
    CosFaceHead,
    CosineCEHead,
    DBMHead,
    FocalHead,
    GCEHead,
    LDAMHead,
    LinearHead,
    LogitAdjustHead,
    MultiLabelHead,
    MultiTaskHead,
    PolyHead,
    SCEHead,
    SeesawHead,
    SubCenterHead,
    VSHead,
    hierarchical_head,
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
    lambda: LinearHead(C, D, mode="ce", class_counts=COUNTS, weighting="inverse"),
    lambda: LinearHead(C, D, mode="ce", class_counts=COUNTS, weighting="cb"),
    lambda: LinearHead(C, D, mode="bce", class_counts=COUNTS),
    lambda: LinearHead(C, D, mode="balanced_softmax", class_counts=COUNTS),
    lambda: PolyHead(C, D),
    lambda: CosineCEHead(C, D),
    lambda: AAMHead(C, D),
    lambda: CosFaceHead(C, D),
    lambda: SubCenterHead(C, D, k=3),
    lambda: LDAMHead(C, D, class_counts=COUNTS, class_balanced=True),
    lambda: FocalHead(C, D, class_counts=COUNTS, cb_beta=0.99),
    lambda: LogitAdjustHead(C, D, class_counts=COUNTS),
    lambda: VSHead(C, D, class_counts=COUNTS),
    lambda: SeesawHead(C, D),
    lambda: DBMHead(C, D, class_counts=COUNTS),
    lambda: GCEHead(C, D),
    lambda: SCEHead(C, D),
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


# --- сверка формул новых голов с референсами ---------------------------------
def test_cosface_subtracts_margin_on_target_only():
    torch.manual_seed(3)
    head = CosFaceHead(C, D, m=0.35, s=1.0)  # s=1: сравниваем чистые косинусы
    emb, targets = batch()
    cos = P.cosine_logits(emb, head.weight)
    out_logits = head.s * P.subtract_class_margin(cos, targets["label"], head.margins)
    row, y = 0, targets["label"][0].item()
    assert out_logits[row, y].item() == pytest.approx(cos[row, y].item() - 0.35, abs=1e-6)
    other = (y + 1) % C
    assert out_logits[row, other].item() == pytest.approx(cos[row, other].item(), abs=1e-6)


def test_poly_eps_zero_equals_ce():
    torch.manual_seed(4)
    emb, targets = batch()
    poly = PolyHead(C, D, epsilon=0.0)
    ce = torch.nn.functional.cross_entropy(poly.fc(emb), targets["label"])
    assert poly(emb, targets)["total_loss"].item() == pytest.approx(ce.item(), abs=1e-6)


def test_gce_q_one_is_mae():
    torch.manual_seed(5)
    emb, targets = batch()
    head = GCEHead(C, D, q=1.0)
    p = torch.softmax(head.fc(emb), dim=1)
    p_t = p.gather(1, targets["label"][:, None]).squeeze(1)
    assert head(emb, targets)["total_loss"].item() == pytest.approx(
        (1.0 - p_t).mean().item(), abs=1e-6)


def test_sce_beta_zero_is_alpha_ce():
    torch.manual_seed(6)
    emb, targets = batch()
    head = SCEHead(C, D, alpha=0.5, beta=0.0)
    ce = torch.nn.functional.cross_entropy(head.fc(emb), targets["label"])
    assert head(emb, targets)["total_loss"].item() == pytest.approx(0.5 * ce.item(), abs=1e-6)


def test_weighted_ce_matches_reference():
    torch.manual_seed(7)
    emb, targets = batch()
    head = LinearHead(C, D, mode="ce", class_counts=COUNTS, weighting="inverse")
    ref = torch.nn.functional.cross_entropy(
        head.fc(emb), targets["label"], weight=P.inverse_freq_weights(COUNTS))
    assert head(emb, targets)["total_loss"].item() == pytest.approx(ref.item(), abs=1e-6)


def test_weighting_requires_class_counts():
    with pytest.raises(ValueError, match="class_counts"):
        LinearHead(C, D, mode="ce", weighting="cb")


# --- multi-label --------------------------------------------------------------
def multihot_batch(n=N):
    torch.manual_seed(8)
    emb = torch.randn(n, D, requires_grad=True)
    t = torch.randint(0, 2, (n, C))
    return emb, {"label": t}


def test_multilabel_contract_forward_and_predict():
    for mode in ("bce", "asl"):
        head = MultiLabelHead(C, D, mode=mode)
        emb, targets = multihot_batch()
        loss = head(emb, targets)["total_loss"]
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert head.predict_logits(emb.detach()).shape == (N, C)
        assert head.classifier_weight.shape == (C, D)


def test_multilabel_masks_elementwise():
    """-1 в отдельном элементе мульти-хота: класс не даёт ни лосса, ни градиента."""
    head = MultiLabelHead(C, D, mode="bce")
    emb, targets = multihot_batch()
    targets["label"][:, 0] = MISSING_LABEL  # класс 0 не размечен ни у кого
    head(emb, targets)["total_loss"].backward()
    assert torch.allclose(head.fc.weight.grad[0], torch.zeros(D))
    assert head.fc.weight.grad[1:].abs().sum() > 0


def test_multilabel_all_missing_zero_loss():
    head = MultiLabelHead(C, D)
    emb, targets = multihot_batch()
    targets["label"][:] = MISSING_LABEL
    assert head(emb, targets)["total_loss"].item() == 0.0


def test_multilabel_bce_matches_reference():
    torch.manual_seed(9)
    emb, targets = multihot_batch()
    head = MultiLabelHead(C, D, mode="bce")
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        head.fc(emb), targets["label"].float())
    assert head(emb, targets)["total_loss"].item() == pytest.approx(ref.item(), abs=1e-6)


def test_asl_zero_gammas_zero_clip_equals_bce():
    emb, targets = multihot_batch()
    bce = MultiLabelHead(C, D, mode="bce")
    asl = MultiLabelHead(C, D, mode="asl", gamma_neg=0.0, gamma_pos=0.0, clip=0.0)
    asl.fc.load_state_dict(bce.fc.state_dict())
    assert asl(emb, targets)["total_loss"].item() == pytest.approx(
        bce(emb, targets)["total_loss"].item(), abs=1e-5)


def test_multilabel_binary_single_logit():
    """n_class=1 + плоский (B,)-таргет {0,1,-1} — бинарная классификация одним логитом."""
    head = MultiLabelHead(1, D, mode="bce")
    emb = torch.randn(N, D, requires_grad=True)
    t = torch.randint(0, 2, (N,))
    t[:3] = MISSING_LABEL
    loss = head(emb, {"label": t})["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert head.predict_logits(emb.detach()).shape == (N, 1)


def test_multilabel_rejects_flat_target_for_multiclass_shape():
    head = MultiLabelHead(C, D)
    emb, targets = batch()  # (B,) int-метки
    with pytest.raises(ValueError, match="мульти-хот"):
        head(emb, targets)


# --- иерархическая классификация ----------------------------------------------
TAXONOMY = Taxonomy.from_dict({
    "levels": ["coarse", "fine"],
    "nodes": {
        "malignant": {"level": "coarse"},
        "benign": {"level": "coarse"},
        "melanoma": {"level": "fine", "parent": "malignant"},
        "bcc": {"level": "fine", "parent": "malignant"},
        "nevus": {"level": "fine", "parent": "benign"},
    },
})


def test_hierarchical_head_builder():
    head = hierarchical_head(TAXONOMY, embedding_dim=D, weights={"coarse": 0.3})
    assert set(head.heads) == {"coarse", "fine"}
    assert head.primary == "fine"  # дефолт — самый тонкий уровень
    assert head.heads["coarse"].n_class == 2 and head.heads["fine"].n_class == 3
    assert head.heads["coarse"].target_key == "label_coarse"

    torch.manual_seed(10)
    emb = torch.randn(N, D, requires_grad=True)
    targets = {"label_coarse": torch.randint(0, 2, (N,)),
               "label_fine": torch.randint(0, 3, (N,))}
    targets["label_fine"][:5] = MISSING_LABEL  # разметка только до coarse
    out = head(emb, targets)
    assert {"total_loss", "coarse_loss", "fine_loss"} <= set(out)
    out["total_loss"].backward()
    assert head.predict_logits(emb.detach()).shape == (N, 3)


def test_hierarchical_head_custom_factory():
    head = hierarchical_head(
        TAXONOMY, embedding_dim=D, primary_level="coarse",
        head_factory=lambda n_class, embedding_dim, target_key: AAMHead(
            n_class, embedding_dim, target_key=target_key),
    )
    assert isinstance(head.heads["fine"], AAMHead)
    assert head.primary == "coarse"
