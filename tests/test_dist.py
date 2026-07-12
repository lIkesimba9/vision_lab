import torch

import vision_lab.core.dist as dist_mod
from vision_lab.core.dist import DistributedBatchSamplerMixin, all_gather_grad


def test_all_gather_grad_identity_single_process():
    x = torch.randn(4, 3, requires_grad=True)
    y = all_gather_grad(x)
    assert y is x  # world=1 -> identity, нулевые накладные расходы


class ToySampler(DistributedBatchSamplerMixin):
    """20 батчей по 2 индекса, порядок зависит от эпохи."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _num_batches_total(self):
        return 20

    def _build_epoch_batches(self):
        import random

        rng = random.Random(self._epoch_seed())
        batches = [[2 * i, 2 * i + 1] for i in range(20)]
        rng.shuffle(batches)
        return batches


def test_ranks_get_disjoint_equal_slices(monkeypatch):
    def batches_for(rank, world):
        monkeypatch.setattr(dist_mod, "dist_info", lambda: (world, rank))
        s = ToySampler()
        s.set_epoch(0)
        return list(s)

    b0, b1 = batches_for(0, 2), batches_for(1, 2)
    assert len(b0) == len(b1) == 10
    flat0 = {i for b in b0 for i in b}
    flat1 = {i for b in b1 for i in b}
    assert flat0.isdisjoint(flat1)


def test_same_epoch_deterministic_different_epochs_reshuffled(monkeypatch):
    monkeypatch.setattr(dist_mod, "dist_info", lambda: (1, 0))
    s = ToySampler()
    s.set_epoch(3)
    a = list(s)
    s.set_epoch(3)
    assert list(s) == a  # детерминизм внутри эпохи
    s.set_epoch(4)
    assert list(s) != a  # reshuffle между эпохами


def test_len_accounts_for_world(monkeypatch):
    monkeypatch.setattr(dist_mod, "dist_info", lambda: (2, 0))
    assert len(ToySampler()) == 10
    monkeypatch.setattr(dist_mod, "dist_info", lambda: (1, 0))
    assert len(ToySampler()) == 20
