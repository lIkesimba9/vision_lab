from collections import Counter

import pytest

import vision_lab.core.dist as dist_mod
from vision_lab.data.samplers import (
    PKBatchSampler,
    PKCoverageBatchSampler,
    PositivePairsBatchSampler,
)

# 4 класса с перекосом + пул -1 (no_label)
LABELS = [0] * 50 + [1] * 30 + [2] * 12 + [3] * 8 + [-1] * 20


def flat(batches):
    return [i for b in batches for i in b]


def test_pk_coverage_sees_every_sample():
    s = PKCoverageBatchSampler(LABELS, batch_size=16, n_labels_per_batch=4)
    s.set_epoch(0)
    batches = list(s)
    assert all(len(b) == 16 for b in batches)
    assert set(flat(batches)) == set(range(len(LABELS)))  # полное покрытие за эпоху


def test_pk_coverage_batches_have_positives():
    s = PKCoverageBatchSampler(LABELS, batch_size=16, n_labels_per_batch=4)
    s.set_epoch(1)
    for batch in s:
        counts = Counter(LABELS[i] for i in batch)
        assert max(counts.values()) >= 2  # в каждом батче есть позитивная пара


def test_positive_pairs_pk_structure():
    s = PositivePairsBatchSampler(LABELS, batch_size=16, n_labels_per_batch=4, seed=1)
    s.set_epoch(0)
    for batch in s:
        counts = Counter(LABELS[i] for i in batch)
        assert len(counts) == 4  # ровно P разных классов
        assert all(v == 4 for v in counts.values())  # по K примеров


def test_pk_alpha_zero_uniform_alpha_one_natural():
    n_iter = 200
    labels = [0] * 90 + [1] * 10

    def class_freq(alpha):
        s = PKBatchSampler(labels, batch_size=4, n_labels_per_batch=2, alpha=alpha)
        hits = Counter()
        for e in range(n_iter):
            s.set_epoch(e)
            for batch in s:
                hits.update({labels[i] for i in batch})
        return hits

    freq = class_freq(alpha=0.0)
    # P=2 из 2 классов => оба класса в каждом батче независимо от alpha;
    # проверяем сам механизм вероятностей на 3 классах
    labels3 = [0] * 80 + [1] * 15 + [2] * 5
    s = PKBatchSampler(labels3, batch_size=4, n_labels_per_batch=2, alpha=1.0, seed=7)
    hits = Counter()
    for e in range(n_iter):
        s.set_epoch(e)
        for batch in s:
            hits.update({labels3[i] for i in batch})
    assert hits[0] > hits[1] > hits[2]  # натуральный перекос при alpha=1
    assert freq[0] > 0 and freq[1] > 0


def test_ddp_ranks_disjoint_and_equal(monkeypatch):
    def batches_for(rank):
        monkeypatch.setattr(dist_mod, "dist_info", lambda: (2, rank))
        s = PKCoverageBatchSampler(LABELS, batch_size=16, n_labels_per_batch=4, seed=5)
        s.set_epoch(2)
        return list(s)

    b0, b1 = batches_for(0), batches_for(1)
    assert len(b0) == len(b1)  # равное число батчей — нет дедлока DDP
    ids0, ids1 = set(map(tuple, b0)), set(map(tuple, b1))
    assert ids0.isdisjoint(ids1)  # нет дублей между рангами


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        PKCoverageBatchSampler(LABELS, batch_size=15, n_labels_per_batch=4)
    with pytest.raises(ValueError):
        PositivePairsBatchSampler([0, 1], batch_size=8, n_labels_per_batch=4)
