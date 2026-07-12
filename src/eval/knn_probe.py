"""Лёгкие прокси качества SSL-фич для отбора чекпоинтов на лету: kNN macro-F1 и
linear-probe macro-F1. Считаются по эмбеддингам backbone (а не SSL-loss, который
мисранкует относительно downstream).

Галерея — held-out milk_train (модель её НЕ видела в предобучении); квери —
вторая половина milk_train и (опц.) melanoscope. Всё на cosine-пространстве
(эмбеддинги L2-нормируются).
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def knn_macro_f1(gallery_x, gallery_y, query_x, query_y, k: int = 20) -> float:
    """Косинусный kNN (взвешивание по похожести), macro-F1 на квери."""
    g, q = _l2(gallery_x), _l2(query_x)
    sims = q @ g.T                                   # (Nq, Ng)
    k = min(k, g.shape[0])
    topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    preds = np.empty(q.shape[0], dtype=np.int64)
    for i in range(q.shape[0]):
        idx = topk[i]
        # голосование соседей с весом по косинусной похожести
        votes = {}
        for jj, ny in enumerate(gallery_y[idx]):
            votes[ny] = votes.get(ny, 0.0) + max(sims[i, idx[jj]], 0.0)
        preds[i] = max(votes, key=votes.get)
    return float(f1_score(query_y, preds, average="macro"))


def linear_probe_macro_f1(gallery_x, gallery_y, query_x, query_y, max_iter: int = 200,
                          random_state: int = 42) -> float:
    """Логрег поверх замороженных эмбеддингов (галерея=train), macro-F1 на квери."""
    g, q = _l2(gallery_x), _l2(query_x)
    if np.unique(gallery_y).size < 2:
        return 0.0
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", C=1.0,
                             n_jobs=-1, random_state=random_state)
    clf.fit(g, gallery_y)
    preds = clf.predict(q)
    return float(f1_score(query_y, preds, average="macro"))


def stratified_gallery_split(labels: np.ndarray, gallery_frac: float = 0.5, seed: int = 42):
    """Детерминированный стратифицированный сплит индексов на (gallery, query)."""
    rng = np.random.RandomState(seed)
    gallery_idx, query_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_g = max(1, int(round(len(idx) * gallery_frac)))
        gallery_idx.extend(idx[:n_g].tolist())
        query_idx.extend(idx[n_g:].tolist())
    return np.array(sorted(gallery_idx)), np.array(sorted(query_idx))
