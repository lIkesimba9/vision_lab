import torch

from vision_lab.core.batch import MISSING_LABEL
from vision_lab.losses import SupConLoss, TripletSemiHardLoss, pairwise_distance


def test_pairwise_distance_zero_diagonal_and_symmetric():
    x = torch.randn(5, 4)
    d = pairwise_distance(x)
    assert torch.allclose(torch.diagonal(d), torch.zeros(5), atol=1e-5)
    assert torch.allclose(d, d.t(), atol=1e-5)


def test_triplet_zero_without_positive_pairs():
    emb = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])  # все уникальны — позитивов нет
    loss = TripletSemiHardLoss()(emb, labels)
    assert loss.item() == 0.0


def test_triplet_separates_clusters():
    # два хорошо разделённых кластера -> малый лосс; перемешанные -> больше
    torch.manual_seed(0)
    good = torch.cat([torch.zeros(4, 8) + 5, torch.zeros(4, 8) - 5])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    trip = TripletSemiHardLoss(margin=1.0)
    loss_good = trip(good, labels)
    bad = torch.randn(8, 8)
    loss_bad = trip(bad, labels)
    assert loss_good < loss_bad


def test_triplet_ignores_unlabeled():
    emb = torch.randn(6, 8)
    labels = torch.tensor([0, 0, 1, 1, MISSING_LABEL, MISSING_LABEL])
    loss = TripletSemiHardLoss()(emb, labels)
    assert torch.isfinite(loss)


def test_supcon_pulls_same_class_together():
    torch.manual_seed(1)
    emb = torch.cat([torch.randn(3, 8) + 3, torch.randn(3, 8) - 3])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    sup = SupConLoss(temperature=0.1)
    loss_sep = sup(emb, labels)
    loss_mixed = sup(torch.randn(6, 8), labels)
    assert torch.isfinite(loss_sep)
    assert loss_sep < loss_mixed


def test_supcon_zero_without_positives():
    emb = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    loss = SupConLoss()(emb, labels)
    assert loss.item() == 0.0
    loss.backward()  # граф сохранён
