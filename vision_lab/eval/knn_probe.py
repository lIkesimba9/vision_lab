"""kNN и linear-probe оценка представлений (порт из прототипа, ТЗ §5.6).

Для SSL — онлайн-отбор чекпоинтов по целевой метрике (macro-F1), а не по
proxy-лоссу: SSL-лосс не коррелирует с качеством.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def stratified_gallery_split(labels: np.ndarray, gallery_frac: float,
                             seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированный сплит индексов на галерею/квери (по classам)."""
    rng = np.random.RandomState(seed)
    gallery_idx, query_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * gallery_frac)))
        gallery_idx.extend(idx[:cut])
        query_idx.extend(idx[cut:])
    return np.array(gallery_idx), np.array(query_idx)


def knn_macro_f1(gallery_x, gallery_y, query_x, query_y, k: int = 20) -> float:
    """kNN macro-F1: косинусное similarity-взвешенное голосование."""
    g = gallery_x / (np.linalg.norm(gallery_x, axis=1, keepdims=True) + 1e-8)
    q = query_x / (np.linalg.norm(query_x, axis=1, keepdims=True) + 1e-8)
    sims = q @ g.T
    k = min(k, g.shape[0])
    topk = np.argsort(-sims, axis=1)[:, :k]
    classes = np.unique(gallery_y)
    preds = np.empty(len(query_y), dtype=gallery_y.dtype)
    for i in range(len(query_y)):
        neigh_y = gallery_y[topk[i]]
        neigh_w = sims[i, topk[i]].clip(min=0)
        votes = {c: neigh_w[neigh_y == c].sum() for c in classes}
        preds[i] = max(votes, key=votes.get)
    return float(f1_score(query_y, preds, average="macro"))


def linear_probe_macro_f1(gallery_x, gallery_y, query_x, query_y) -> float:
    """Linear-probe macro-F1: балансированная логистическая регрессия на галерее."""
    if len(np.unique(gallery_y)) < 2:
        return 0.0
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(gallery_x, gallery_y)
    return float(f1_score(query_y, clf.predict(query_x), average="macro"))
