"""Зоопарк голов-классификаторов (порт из прототипа, единый контракт §4.2).

Все головы читают свою метку через ``target_key`` из словаря таргетов,
маскируют ``-1`` и возвращают ``{"total_loss": ..., <компоненты>}``. Общий
margin/cosine/long-tail код — в :mod:`vision_lab.heads.primitives`.

Семейства (ТЗ §5.1):
  * softmax: :class:`LinearHead` (ce/bce/balanced_softmax, label smoothing,
    weighted CE), :class:`PolyHead` (Poly-1);
  * multi-label: :class:`MultiLabelHead` (BCE / ASL, мульти-хот таргет);
  * angular: :class:`CosineCEHead`, :class:`AAMHead`, :class:`CosFaceHead`,
    :class:`SubCenterHead`;
  * long-tail: :class:`FocalHead`, :class:`LDAMHead`, :class:`LogitAdjustHead`,
    :class:`SeesawHead`, :class:`VSHead`, :class:`DBMHead`;
  * noise-robust: :class:`GCEHead`, :class:`SCEHead`;
  * метрические: :class:`AAMTripletHead` (AAM + triplet).

Принцип baseline-first (§5.1): эталон — ``LinearHead(mode="ce")`` + сильные
аугментации; зоопарк доступен, но сравнивается с этим бейзлайном.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vision_lab.heads import primitives as P
from vision_lab.heads.base import ClassifierHead
from vision_lab.losses.metric import TripletSemiHardLoss


class LinearHead(ClassifierHead):
    """Линейный классификатор: mode ∈ {ce, bce, balanced_softmax}.

    ce — обычный CrossEntropy; bce — one-vs-rest из ОДИНОЧНОЙ int-метки +
    class-balanced pos_weight (настоящий multi-label с мульти-хот таргетом —
    :class:`MultiLabelHead`); balanced_softmax — long-tail softmax со сдвигом
    на log-приор.

    ``weighting`` — веса классов для ce/balanced_softmax (требует class_counts):
    ``"inverse"`` — обратная частота, ``"cb"`` — Class-Balanced (Cui 2019,
    параметр ``cb_beta``). Это базовый long-tail бейзлайн до всякого зоопарка.
    """

    def __init__(self, n_class: int, embedding_dim: int, mode: str = "ce",
                 class_counts=None, label_smoothing: float = 0.0, pos_weight=None,
                 weighting: str | None = None, cb_beta: float = 0.999,
                 target_key: str = "label", fc_weight_path: str | None = None):
        super().__init__()
        if mode not in {"ce", "bce", "balanced_softmax"}:
            raise ValueError(f"mode={mode!r} (ce|bce|balanced_softmax)")
        if weighting not in {None, "inverse", "cb"}:
            raise ValueError(f"weighting={weighting!r} (None|inverse|cb)")
        if weighting is not None and class_counts is None:
            raise ValueError("weighting задан — нужен и class_counts")
        if weighting is not None and mode == "bce":
            raise ValueError("weighting применим к ce/balanced_softmax; для bce — pos_weight")
        self.mode = mode
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.target_key = target_key
        self.label_smoothing = label_smoothing
        self.weighting = weighting
        self.fc = nn.Linear(embedding_dim, n_class)

        log_prior = P.logit_prior(class_counts) if class_counts is not None else torch.zeros(n_class)
        if pos_weight is not None:
            cb_pos = torch.as_tensor(pos_weight, dtype=torch.float)
        elif class_counts is not None:
            counts = P.counts_to_tensor(class_counts)
            cb_pos = (counts.sum() - counts) / counts
        else:
            cb_pos = torch.ones(n_class)
        if weighting == "inverse":
            ce_weight = P.inverse_freq_weights(class_counts)
        elif weighting == "cb":
            ce_weight = P.class_balanced_weights(class_counts, cb_beta)
        else:
            ce_weight = torch.ones(n_class)
        self.register_buffer("log_prior", log_prior)
        self.register_buffer("pos_weight", cb_pos)
        self.register_buffer("ce_weight", ce_weight)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    @property
    def classifier_bias(self):
        return self.fc.bias

    def forward(self, embeddings, targets: Mapping[str, torch.Tensor]):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        logits = self.fc(embeddings[mask])
        y = labels[mask]
        if self.mode == "bce":
            tgt = F.one_hot(y, self.n_class).to(logits.dtype)
            loss = F.binary_cross_entropy_with_logits(
                logits, tgt, pos_weight=self.pos_weight.to(logits.dtype))
        elif self.mode == "balanced_softmax":
            loss = F.cross_entropy(logits + self.log_prior, y,
                                   weight=self.ce_weight.to(logits.dtype),
                                   label_smoothing=self.label_smoothing)
        else:
            loss = F.cross_entropy(logits, y, weight=self.ce_weight.to(logits.dtype),
                                   label_smoothing=self.label_smoothing)
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class MultiLabelHead(ClassifierHead):
    """Настоящий multi-label: таргет — мульти-хот ``(B, C)`` из {0, 1, -1}.

    ``-1`` маскируется ПОэлементно (частичная разметка: неизвестен конкретный
    класс сэмпла, а не вся строка). mode: ``bce`` — BCEWithLogits (+ явный
    pos_weight); ``asl`` — Asymmetric Loss (Ridnik 2021; γ+=γ-=γ, clip=0 даёт
    focal-BCE, γ=0 — BCE). ``n_class=1`` — бинарная классификация одним
    логитом, допускается плоский ``(B,)``-таргет из {0, 1, -1}.
    """

    def __init__(self, n_class: int, embedding_dim: int, mode: str = "bce",
                 gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05,
                 pos_weight=None, target_key: str = "label",
                 fc_weight_path: str | None = None):
        super().__init__()
        if mode not in {"bce", "asl"}:
            raise ValueError(f"mode={mode!r} (bce|asl)")
        self.mode = mode
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip
        self.target_key = target_key
        self.fc = nn.Linear(embedding_dim, n_class)
        pw = (torch.as_tensor(pos_weight, dtype=torch.float)
              if pos_weight is not None else torch.ones(n_class))
        self.register_buffer("pos_weight", pw)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    @property
    def classifier_bias(self):
        return self.fc.bias

    def forward(self, embeddings, targets):
        t = self.take_target(targets)
        if t.ndim == 1:
            if self.n_class != 1:
                raise ValueError(
                    f"MultiLabelHead(n_class={self.n_class}) ждёт мульти-хот (B, C); "
                    f"плоский (B,)-таргет допустим только при n_class=1 "
                    f"(int-метки классов — это LinearHead)"
                )
            t = t.unsqueeze(1)
        logits = self.fc(embeddings)
        if self.mode == "asl":
            loss = P.asymmetric_loss_with_logits(logits, t, self.gamma_neg,
                                                 self.gamma_pos, self.clip)
        else:
            loss = P.masked_bce_with_logits(logits, t, self.pos_weight)
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class PolyHead(ClassifierHead):
    """Poly-1 loss (Leng 2022): CE + ε·(1 - p_t); ε=0 — ровно CE."""

    def __init__(self, n_class, embedding_dim, epsilon=1.0,
                 target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.epsilon = epsilon
        self.target_key = target_key
        self.fc = nn.Linear(embedding_dim, n_class)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        y = labels[mask]
        logp = F.log_softmax(self.fc(embeddings[mask]), dim=1)
        logp_t = logp.gather(1, y[:, None]).squeeze(1)
        loss = (-logp_t + self.epsilon * (1.0 - logp_t.exp())).mean()
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class GCEHead(ClassifierHead):
    """Generalized CE (Zhang 2018): (1 - p_t^q)/q — устойчив к шумным меткам.

    q→0 — обычный CE, q=1 — MAE; q интерполирует между ними (дефолт из статьи 0.7).
    """

    def __init__(self, n_class, embedding_dim, q=0.7,
                 target_key="label", fc_weight_path=None):
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q={q} должен быть в (0, 1]")
        self.n_class, self.embedding_dim, self.q = n_class, embedding_dim, q
        self.target_key = target_key
        self.fc = nn.Linear(embedding_dim, n_class)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        y = labels[mask]
        p = F.softmax(self.fc(embeddings[mask]), dim=1)
        p_t = p.gather(1, y[:, None]).squeeze(1).clamp_min(1e-8)
        loss = ((1.0 - p_t.pow(self.q)) / self.q).mean()
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class SCEHead(ClassifierHead):
    """Symmetric CE (Wang 2019): α·CE + β·RCE — устойчив к шумным меткам.

    RCE (reverse CE) с log(0) := A даёт замкнутую форму -A·(1 - p_t); A=-4 из
    статьи. β=0 — чистый α·CE.
    """

    def __init__(self, n_class, embedding_dim, alpha=0.1, beta=1.0, A=-4.0,
                 target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.alpha, self.beta, self.A = alpha, beta, A
        self.target_key = target_key
        self.fc = nn.Linear(embedding_dim, n_class)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        y = labels[mask]
        logits = self.fc(embeddings[mask])
        ce = F.cross_entropy(logits, y)
        p_t = F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        rce = (-self.A * (1.0 - p_t)).mean()
        return {"total_loss": self.alpha * ce + self.beta * rce}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class _CosineHead(ClassifierHead):
    """Общий фундамент косинусных голов: обучаемая [C, D]-матрица центров."""

    def __init__(self, n_class: int, embedding_dim: int, s: float,
                 target_key: str, fc_weight_path: str | None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.s = s
        self.target_key = target_key
        self.weight = nn.Parameter(torch.empty(n_class, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.weight

    def predict_logits(self, embeddings):
        return self.s * P.cosine_logits(embeddings, self.weight)


class CosineCEHead(_CosineHead):
    """Нормированная косинусная CE (без маржина)."""

    def __init__(self, n_class, embedding_dim, s=30.0, label_smoothing=0.0,
                 target_key="label", fc_weight_path=None):
        super().__init__(n_class, embedding_dim, s, target_key, fc_weight_path)
        self.label_smoothing = label_smoothing

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        logits = self.s * P.cosine_logits(embeddings, self.weight)
        return {"total_loss": P.masked_cross_entropy(logits, labels,
                                                     label_smoothing=self.label_smoothing)}


class AAMHead(_CosineHead):
    """Additive Angular Margin (ArcFace)."""

    def __init__(self, n_class, embedding_dim, m=0.2, s=30.0, label_smoothing=0.0,
                 target_key="label", fc_weight_path=None):
        super().__init__(n_class, embedding_dim, s, target_key, fc_weight_path)
        self.m = m
        self.label_smoothing = label_smoothing

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        cosine = P.cosine_logits(embeddings[mask], self.weight)
        logits = self.s * P.additive_angular_margin(cosine, labels[mask], self.m)
        return {"total_loss": F.cross_entropy(logits, labels[mask],
                                              label_smoothing=self.label_smoothing)}


class CosFaceHead(_CosineHead):
    """CosFace / AM-Softmax (Wang 2018): аддитивный КОСИНУСНЫЙ маржин cos(θ) - m.

    Проще ArcFace (маржин в косинусе, не в угле) — нет тригонометрии и
    защиты монотонности; типичный m выше арк-фейсовского (0.35 vs 0.2).
    """

    def __init__(self, n_class, embedding_dim, m=0.35, s=30.0, label_smoothing=0.0,
                 target_key="label", fc_weight_path=None):
        super().__init__(n_class, embedding_dim, s, target_key, fc_weight_path)
        self.m = m
        self.label_smoothing = label_smoothing
        self.register_buffer("margins", torch.full((n_class,), float(m)))

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        cosine = P.cosine_logits(embeddings[mask], self.weight)
        logits = self.s * P.subtract_class_margin(cosine, labels[mask], self.margins)
        return {"total_loss": F.cross_entropy(logits, labels[mask],
                                              label_smoothing=self.label_smoothing)}


class SubCenterHead(ClassifierHead):
    """Sub-center ArcFace: k суб-центров на класс (устойчив к шумным меткам)."""

    def __init__(self, n_class, embedding_dim, m=0.3, s=30.0, k=2, label_smoothing=0.0,
                 target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.m, self.s, self.k = m, s, k
        self.target_key = target_key
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.empty(n_class * k, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        # для переноса — первый суб-центр каждого класса -> [C, D]
        return self.weight.view(self.n_class, self.k, self.embedding_dim)[:, 0, :]

    def _cosine(self, embeddings):
        return P.subcenter_reduce(P.cosine_logits(embeddings, self.weight), self.n_class, self.k)

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        cosine = self._cosine(embeddings[mask])
        logits = self.s * P.additive_angular_margin(cosine, labels[mask], self.m)
        return {"total_loss": F.cross_entropy(logits, labels[mask],
                                              label_smoothing=self.label_smoothing)}

    def predict_logits(self, embeddings):
        return self.s * self._cosine(embeddings)


class LDAMHead(_CosineHead):
    """LDAM (Cao 2019): косинусная голова + класс-зависимый маржин ∝ n_c^{-1/4}."""

    def __init__(self, n_class, embedding_dim, class_counts, max_m=0.5, s=30.0,
                 label_smoothing=0.0, class_balanced=False, target_key="label",
                 fc_weight_path=None):
        super().__init__(n_class, embedding_dim, s, target_key, fc_weight_path)
        self.label_smoothing = label_smoothing
        self.register_buffer("margins", P.ldam_margins(class_counts, max_m))
        w = P.inverse_freq_weights(class_counts) if class_balanced else torch.ones(n_class)
        self.register_buffer("cls_weight", w)

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        cos = P.cosine_logits(embeddings[mask], self.weight)
        cos_m = P.subtract_class_margin(cos, labels[mask], self.margins)
        loss = F.cross_entropy(self.s * cos_m, labels[mask],
                               weight=self.cls_weight.to(cos.dtype),
                               label_smoothing=self.label_smoothing)
        return {"total_loss": loss}


class FocalHead(ClassifierHead):
    """Focal loss (Lin 2017); с class_counts+cb_beta — Class-Balanced Focal."""

    def __init__(self, n_class, embedding_dim, gamma=2.0, class_counts=None,
                 cb_beta=None, target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim, self.gamma = n_class, embedding_dim, gamma
        self.target_key = target_key
        self.fc = nn.Linear(embedding_dim, n_class)
        alpha = (P.class_balanced_weights(class_counts, cb_beta)
                 if class_counts is not None and cb_beta is not None else torch.ones(n_class))
        self.register_buffer("alpha", alpha)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        y = labels[mask]
        logp = F.log_softmax(self.fc(embeddings[mask]), dim=1)
        logp_t = logp.gather(1, y[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -(self.alpha[y] * (1.0 - p_t).pow(self.gamma) * logp_t).mean()
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class LogitAdjustHead(ClassifierHead):
    """Logit Adjustment (Menon 2021) / обобщённый Balanced Softmax: CE(z + τ·log prior).

    adjust_inference=True → post-hoc LA: на инференсе вычитаем τ·log prior.
    """

    def __init__(self, n_class, embedding_dim, class_counts, tau=1.0, label_smoothing=0.0,
                 adjust_inference=False, target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.tau, self.adjust_inference = tau, adjust_inference
        self.target_key = target_key
        self.label_smoothing = label_smoothing
        self.fc = nn.Linear(embedding_dim, n_class)
        self.register_buffer("log_prior", P.logit_prior(class_counts))
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        logits = self.fc(embeddings) + self.tau * self.log_prior
        return {"total_loss": P.masked_cross_entropy(logits, labels,
                                                     label_smoothing=self.label_smoothing)}

    def predict_logits(self, embeddings):
        z = self.fc(embeddings)
        return z - self.tau * self.log_prior if self.adjust_inference else z


class VSHead(ClassifierHead):
    """VS loss (Kini 2021): z' = Δ·z + ι, Δ_c=(n_c/n_max)^γ (мультипл.), ι_c=τ·log prior (аддит.)."""

    def __init__(self, n_class, embedding_dim, class_counts, gamma=0.3, tau=1.0,
                 label_smoothing=0.0, target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.target_key = target_key
        self.label_smoothing = label_smoothing
        self.fc = nn.Linear(embedding_dim, n_class)
        counts = P.counts_to_tensor(class_counts)
        self.register_buffer("delta", (counts / counts.max()).pow(gamma))
        self.register_buffer("iota", P.logit_prior(class_counts, tau))
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        logits = self.delta * self.fc(embeddings) + self.iota
        return {"total_loss": P.masked_cross_entropy(logits, labels,
                                                     label_smoothing=self.label_smoothing)}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class SeesawHead(ClassifierHead):
    """Seesaw loss (Wang 2021): mitigation (накопл. частоты) × compensation (мис-классиф.)."""

    def __init__(self, n_class, embedding_dim, p=0.8, q=2.0, eps=1e-2,
                 target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.p, self.q, self.eps = p, q, eps
        self.target_key = target_key
        self.fc = nn.Linear(embedding_dim, n_class)
        self.register_buffer("cum_samples", torch.zeros(n_class))
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        y = labels[mask]
        logits = self.fc(embeddings[mask])
        self.cum_samples += torch.bincount(y, minlength=self.n_class).float()
        onehot = F.one_hot(y, self.n_class).float()
        weights = logits.new_ones(onehot.size())
        cum = self.cum_samples.clamp_min(1.0)
        ratio = cum[None, :] / cum[:, None]
        idx = (ratio < 1.0).float()
        weights = weights * (ratio.pow(self.p) * idx + (1.0 - idx))[y, :]
        if self.q > 0:
            scores = F.softmax(logits.detach(), dim=1)
            self_s = scores.gather(1, y[:, None]).clamp_min(self.eps)
            score_m = scores / self_s
            idx2 = (score_m > 1.0).float()
            weights = weights * (score_m.pow(self.q) * idx2 + (1.0 - idx2))
        adj = logits + (weights.clamp_min(1e-8).log() * (1.0 - onehot))
        return {"total_loss": F.cross_entropy(adj, y)}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class DBMHead(_CosineHead):
    """Difficulty-aware Balancing Margin (AAAI 2025): маржин = класс-частотный
    (LDAM-style) × инстанс-сложностный (∝ 1-p_true). Косинусная голова."""

    def __init__(self, n_class, embedding_dim, class_counts, max_m=0.5, s=30.0,
                 difficulty_scale=0.5, label_smoothing=0.0, class_balanced=True,
                 target_key="label", fc_weight_path=None):
        super().__init__(n_class, embedding_dim, s, target_key, fc_weight_path)
        self.difficulty_scale = difficulty_scale
        self.label_smoothing = label_smoothing
        self.register_buffer("cls_margin", P.ldam_margins(class_counts, max_m))
        w = P.inverse_freq_weights(class_counts) if class_balanced else torch.ones(n_class)
        self.register_buffer("cls_weight", w)

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        mask = P.valid_rows(labels)
        if not mask.any():
            return {"total_loss": embeddings.sum() * 0.0}
        y = labels[mask]
        cos = P.cosine_logits(embeddings[mask], self.weight)
        with torch.no_grad():
            p_true = F.softmax(self.s * cos, dim=1).gather(1, y[:, None]).squeeze(1)
            inst = self.difficulty_scale * (1.0 - p_true)
        margin = self.cls_margin[y] * (1.0 + inst)
        cos_m = cos.clone()
        cur = cos.gather(1, y[:, None]).squeeze(1) - margin
        cos_m.scatter_(1, y[:, None], cur[:, None])
        loss = F.cross_entropy(self.s * cos_m, y, weight=self.cls_weight.to(cos.dtype),
                               label_smoothing=self.label_smoothing)
        return {"total_loss": loss}


class AAMTripletHead(ClassifierHead):
    """AAM + triplet (semi-hard). Требует PK-батч (позитивы гарантированы сэмплером)."""

    def __init__(self, n_class, embedding_dim, triplet_weight=0.3, m=0.2, s=30.0,
                 label_smoothing=0.0, margin=1.0, target_key="label", fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.target_key = target_key
        self.aam = AAMHead(n_class, embedding_dim, m, s, label_smoothing,
                           target_key=target_key, fc_weight_path=fc_weight_path)
        self.triplet = TripletSemiHardLoss(margin=margin)
        self.triplet_weight = triplet_weight

    @property
    def classifier_weight(self):
        return self.aam.weight

    def forward(self, embeddings, targets):
        labels = self.take_target(targets)
        aam = self.aam(embeddings, targets)["total_loss"]
        trip = self.triplet(embeddings, labels)
        total = aam + self.triplet_weight * trip
        return {"total_loss": total, "aam_loss": aam, "triplet_loss": trip}

    def predict_logits(self, embeddings):
        return self.aam.predict_logits(embeddings)
