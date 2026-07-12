"""Головы-классификаторы (критерии) с единым интерфейсом ``ClassifierHead``.

Все принимают эмбеддинги модели и владеют весами классификатора + функцией потерь:

    LinearHead       — обычный Linear: режимы ce | bce | balanced_softmax (заменяет
                       старые CELoss/BCELoss/LinearClsLoss);
    CosineCEHead     — нормированная косинусная CE (без маржина);
    AAMHead          — Additive Angular Margin (ArcFace);
    SubCenterHead    — Sub-center ArcFace;
    AAMTripletHead   — AAM + triplet.

У каждой: forward(emb, labels)->{'total_loss',...}, predict_logits(emb), classifier_weight,
load_fc_weights(...) (унаследовано) — единый перенос FC-весов между стадиями.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.base import ClassifierHead
from src.losses.triplet_loss import TripletLoss
from src.losses.speaker import AngleProtoLoss, GE2ELoss


# =============================================================================
# Линейная голова: ce / bce / balanced_softmax
# =============================================================================
class LinearHead(ClassifierHead):
    """Линейный классификатор поверх эмбеддингов с выбором режима потерь.

    mode:
      'ce'               — обычный CrossEntropy;
      'bce'              — multilabel one-vs-rest BCE (sigmoid) + class-balanced pos_weight;
      'balanced_softmax' — long-tail softmax со сдвигом на log-приор класса.

    class_counts: частоты классов (для 'bce'/'balanced_softmax'); None → без балансировки.
    fc_weight_path: перенести веса классификатора с предыдущей стадии (любой формат, см. base).
    """

    def __init__(self, n_class, embedding_dim, mode="ce", class_counts=None,
                 label_smoothing=0.0, pos_weight=None, fc_weight_path=None):
        super().__init__()
        assert mode in {"ce", "bce", "balanced_softmax"}, mode
        self.mode = mode
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, n_class)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if class_counts is not None:
            counts = torch.tensor(list(class_counts), dtype=torch.float).clamp_min(1.0)
            log_prior = (counts / counts.sum()).log()
            cb_pos_weight = (counts.sum() - counts) / counts          # neg/pos на класс
        else:
            log_prior = torch.zeros(n_class)
            cb_pos_weight = torch.ones(n_class)
        if pos_weight is not None:                                    # явный приоритет
            cb_pos_weight = torch.as_tensor(pos_weight, dtype=torch.float)
        self.register_buffer("log_prior", log_prior)
        self.register_buffer("pos_weight", cb_pos_weight)

        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    @property
    def classifier_bias(self):
        return self.fc.bias

    def forward(self, embeddings, labels):
        logits = self.fc(embeddings)
        if self.mode == "bce":
            target = F.one_hot(labels.long(), self.n_class).to(logits.dtype)
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight.to(logits.dtype))
        elif self.mode == "balanced_softmax":
            loss = self.ce(logits + self.log_prior, labels)
        else:
            loss = self.ce(logits, labels)
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


# =============================================================================
# Косинусная CE (нормированные веса, без маржина)
# =============================================================================
class CosineCEHead(ClassifierHead):
    def __init__(self, n_class, embedding_dim, s=30.0, label_smoothing=0.0, fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.weight

    def forward(self, embeddings, labels):
        logits = self.predict_logits_train(embeddings)
        return {"total_loss": self.ce(logits, labels)}

    def predict_logits_train(self, embeddings):
        return F.linear(F.normalize(embeddings), F.normalize(self.weight)) * self.s

    def predict_logits(self, embeddings):
        return self.predict_logits_train(embeddings)


# =============================================================================
# LDAM (Label-Distribution-Aware Margin) — long-tail, маржин ∝ 1/n_c^{1/4}
# =============================================================================
class LDAMHead(ClassifierHead):
    """LDAM-Softmax (Cao et al. 2019): косинусная голова + класс-зависимый маржин,
    обратно пропорциональный n_c^{1/4}. Сильнее давит хвостовые классы под long-tail.

    class_counts: частоты классов (обязательно). max_m: макс. маржин (норм., 0.5).
    s: масштаб логитов (30). class_balanced: добавить CB-веса в CE (аналог DRW без расписания).
    """

    def __init__(self, n_class, embedding_dim, class_counts, max_m=0.5, s=30.0,
                 label_smoothing=0.0, class_balanced=False, fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.s = s
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        counts = torch.tensor(list(class_counts), dtype=torch.float).clamp_min(1.0)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))   # ∝ n_c^{-1/4}
        margins = margins * (max_m / margins.max())       # норм.: макс маржин = max_m
        self.register_buffer("margins", margins)          # (C,)
        if class_balanced:
            w = counts.sum() / counts
            w = w / w.mean()                              # норм. вокруг 1
        else:
            w = torch.ones(n_class)
        self.register_buffer("cls_weight", w)

    @property
    def classifier_weight(self):
        return self.weight

    def forward(self, embeddings, labels):
        cos = F.linear(F.normalize(embeddings), F.normalize(self.weight))  # (B,C) ∈ [-1,1]
        onehot = F.one_hot(labels.long(), self.n_class).bool()
        cos_m = torch.where(onehot, cos - self.margins.unsqueeze(0), cos)   # маржин на истинный класс
        loss = F.cross_entropy(self.s * cos_m, labels.long(),
                               weight=self.cls_weight.to(cos_m.dtype),
                               label_smoothing=self.label_smoothing)
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.s * F.linear(F.normalize(embeddings), F.normalize(self.weight))


# =============================================================================
# AAM-Softmax (ArcFace)
# =============================================================================
class AAMHead(ClassifierHead):
    def __init__(self, n_class, embedding_dim, m=0.2, s=30.0, label_smoothing=0.0,
                 fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embedding_dim))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.weight

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return {"total_loss": self.ce(output, labels)}

    def predict_logits(self, embeddings):
        emb = F.normalize(embeddings)
        centers = F.normalize(self.weight)
        return emb @ centers.T * self.s


# =============================================================================
# Sub-center ArcFace
# =============================================================================
class SubCenterHead(ClassifierHead):
    def __init__(self, n_class, embedding_dim, m=0.3, s=30.0, k=2, label_smoothing=0.0,
                 fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.m = m
        self.s = s
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(n_class * k, embedding_dim))
        nn.init.xavier_uniform_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        # для переноса берём по одному (первому) суб-центру на класс → [C, D]
        return self.weight.view(self.n_class, self.k, self.embedding_dim)[:, 0, :]

    def _cosine(self, embeddings):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.view(-1, self.n_class, self.k)
        return cosine.max(dim=2)[0]

    def forward(self, embeddings, labels):
        cosine = self._cosine(embeddings)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.n_class).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return {"total_loss": self.ce(logits, labels)}

    def predict_logits(self, embeddings):
        return self._cosine(embeddings) * self.s


# =============================================================================
# AAM + Triplet
# =============================================================================
class AAMTripletHead(ClassifierHead):
    def __init__(self, n_class, embedding_dim, triplet_weight=0.3, m=0.2, s=30.0,
                 label_smoothing=0.0, fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.aam = AAMHead(n_class, embedding_dim, m, s, label_smoothing,
                           fc_weight_path=fc_weight_path)
        self.triplet = TripletLoss()
        self.triplet_weight = triplet_weight

    @property
    def classifier_weight(self):
        return self.aam.weight

    def forward(self, embeddings, labels):
        aam_loss = self.aam(embeddings, labels)["total_loss"]
        triplet_loss = self.triplet(embeddings, labels)
        total = aam_loss + triplet_loss * self.triplet_weight
        return {"total_loss": total, "aam_loss": aam_loss, "triplet_loss": triplet_loss}

    def predict_logits(self, embeddings):
        return self.aam.predict_logits(embeddings)


# =============================================================================
# Long-tail лоссы (MONICA / survey 2404.15593): Focal/CB, LogitAdjust, Seesaw, VS, DBM
# =============================================================================
def _effective_number_weights(counts, beta):
    """Class-Balanced веса (Cui 2019): (1-beta)/(1-beta^n_c), норм. до среднего 1."""
    eff = 1.0 - torch.pow(beta, counts)
    w = (1.0 - beta) / eff.clamp_min(1e-8)
    return w / w.mean()


class FocalHead(ClassifierHead):
    """Focal loss (Lin 2017): -(1-p_t)^gamma * log p_t. С class_counts+cb_beta — Class-Balanced Focal."""

    def __init__(self, n_class, embedding_dim, gamma=2.0, class_counts=None,
                 cb_beta=None, fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim, self.gamma = n_class, embedding_dim, gamma
        self.fc = nn.Linear(embedding_dim, n_class)
        if class_counts is not None and cb_beta is not None:
            w = _effective_number_weights(torch.tensor(list(class_counts), dtype=torch.float).clamp_min(1.0), cb_beta)
        else:
            w = torch.ones(n_class)
        self.register_buffer("alpha", w)
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, labels):
        labels = labels.long()
        logp = F.log_softmax(self.fc(embeddings), dim=1)
        logp_t = logp.gather(1, labels[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -(self.alpha[labels] * (1.0 - p_t).pow(self.gamma) * logp_t).mean()
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class LogitAdjustHead(ClassifierHead):
    """Logit Adjustment (Menon 2021) / обобщённый Balanced Softmax: CE(z + tau*log prior).
    tau=1 == balanced_softmax. adjust_inference=True → на инференсе вычитаем tau*log prior (LA post-hoc)."""

    def __init__(self, n_class, embedding_dim, class_counts, tau=1.0,
                 label_smoothing=0.0, adjust_inference=False, fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.fc = nn.Linear(embedding_dim, n_class)
        self.tau, self.adjust_inference = tau, adjust_inference
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        counts = torch.tensor(list(class_counts), dtype=torch.float).clamp_min(1.0)
        self.register_buffer("log_prior", (counts / counts.sum()).log())
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, labels):
        return {"total_loss": self.ce(self.fc(embeddings) + self.tau * self.log_prior, labels.long())}

    def predict_logits(self, embeddings):
        z = self.fc(embeddings)
        return z - self.tau * self.log_prior if self.adjust_inference else z


class SeesawHead(ClassifierHead):
    """Seesaw loss (Wang 2021): mitigation (накопл. частоты классов) × compensation (мис-классиф.).
    Реализация как в mmdetection. cum_samples копится по ходу обучения (буфер)."""

    def __init__(self, n_class, embedding_dim, p=0.8, q=2.0, eps=1e-2, fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.fc = nn.Linear(embedding_dim, n_class)
        self.p, self.q, self.eps = p, q, eps
        self.register_buffer("cum_samples", torch.zeros(n_class))
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, labels):
        labels = labels.long()
        logits = self.fc(embeddings)
        self.cum_samples += torch.bincount(labels, minlength=self.n_class).float()
        onehot = F.one_hot(labels, self.n_class).float()
        weights = logits.new_ones(onehot.size())
        # mitigation: для класса j относительно target i, если n_j<n_i → (n_j/n_i)^p
        cum = self.cum_samples.clamp_min(1.0)
        ratio = cum[None, :] / cum[:, None]                    # (C,C)
        idx = (ratio < 1.0).float()
        sample_w = ratio.pow(self.p) * idx + (1.0 - idx)       # (C,C)
        weights = weights * sample_w[labels, :]
        # compensation: если скор класса j выше скора target → штраф
        if self.q > 0:
            scores = F.softmax(logits.detach(), dim=1)
            self_s = scores.gather(1, labels[:, None]).clamp_min(self.eps)
            score_m = scores / self_s
            idx2 = (score_m > 1.0).float()
            weights = weights * (score_m.pow(self.q) * idx2 + (1.0 - idx2))
        adj = logits + (weights.clamp_min(1e-8).log() * (1.0 - onehot))
        return {"total_loss": F.cross_entropy(adj, labels)}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class VSHead(ClassifierHead):
    """VS loss (Kini 2021): z' = Delta*z + iota, Delta_c=(n_c/n_max)^gamma (мультипл.),
    iota_c=tau*log prior (аддит.). Объединяет margin- и logit-adjust подходы."""

    def __init__(self, n_class, embedding_dim, class_counts, gamma=0.3, tau=1.0,
                 label_smoothing=0.0, fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.fc = nn.Linear(embedding_dim, n_class)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        counts = torch.tensor(list(class_counts), dtype=torch.float).clamp_min(1.0)
        self.register_buffer("Delta", (counts / counts.max()).pow(gamma))
        self.register_buffer("iota", tau * (counts / counts.sum()).log())
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.fc.weight

    def forward(self, embeddings, labels):
        return {"total_loss": self.ce(self.Delta * self.fc(embeddings) + self.iota, labels.long())}

    def predict_logits(self, embeddings):
        return self.fc(embeddings)


class DBMHead(ClassifierHead):
    """Difficulty-aware Balancing Margin (AAAI 2025), косинусная голова. Маржин на истинный класс =
    класс-частотный (LDAM-style, ∝ n_c^{-1/4}) + инстанс-сложностный (∝ 1-p_true). Сложные хвостовые
    примеры получают больший маржин. (Реализация в духе статьи; class_balanced ≈ DRW-веса.)"""

    def __init__(self, n_class, embedding_dim, class_counts, max_m=0.5, s=30.0,
                 difficulty_scale=0.5, label_smoothing=0.0, class_balanced=True, fc_weight_path=None):
        super().__init__()
        self.n_class, self.embedding_dim, self.s = n_class, embedding_dim, s
        self.difficulty_scale, self.label_smoothing = difficulty_scale, label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        counts = torch.tensor(list(class_counts), dtype=torch.float).clamp_min(1.0)
        cls_m = 1.0 / torch.sqrt(torch.sqrt(counts))
        cls_m = cls_m * (max_m / cls_m.max())
        self.register_buffer("cls_margin", cls_m)
        w = (counts.sum() / counts) if class_balanced else torch.ones(n_class)
        self.register_buffer("cls_weight", w / w.mean())
        if fc_weight_path is not None:
            self.load_fc_weights(fc_weight_path)

    @property
    def classifier_weight(self):
        return self.weight

    def forward(self, embeddings, labels):
        labels = labels.long()
        cos = F.linear(F.normalize(embeddings), F.normalize(self.weight))  # (B,C)
        with torch.no_grad():
            p_true = F.softmax(self.s * cos, dim=1).gather(1, labels[:, None]).squeeze(1)
            inst = self.difficulty_scale * (1.0 - p_true)                  # сложность 0..scale
        margin = self.cls_margin[labels] * (1.0 + inst)                    # класс × сложность
        cos_m = cos.clone()
        cos_m.scatter_(1, labels[:, None], (cos.gather(1, labels[:, None]).squeeze(1) - margin)[:, None])
        loss = F.cross_entropy(self.s * cos_m, labels,
                               weight=self.cls_weight.to(cos.dtype), label_smoothing=self.label_smoothing)
        return {"total_loss": loss}

    def predict_logits(self, embeddings):
        return self.s * F.linear(F.normalize(embeddings), F.normalize(self.weight))


# =============================================================================
# Speaker-verification метрик-лосс (AngleProto / GE2E) как downstream-голова.
# AAM-классификатор (для логитов/инференса) + SV-лосс на PK-сгруппированном батче.
# Требует PK-батч (data_params=milk_dermo_fold_balanced): N классов × K примеров → (N,M,D).
# =============================================================================
class SpeakerHead(ClassifierHead):
    def __init__(self, n_class, embedding_dim, method="angleproto", sv_weight=0.5,
                 m=0.2, s=30.0, label_smoothing=0.0, init_w=10.0, init_b=-5.0,
                 ge2e_method="softmax", fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.aam = AAMHead(n_class, embedding_dim, m, s, label_smoothing,
                           fc_weight_path=fc_weight_path)
        if method == "angleproto":
            self.sv = AngleProtoLoss(init_w, init_b)
        elif method == "ge2e":
            self.sv = GE2ELoss(init_w, init_b, ge2e_method)
        else:
            raise ValueError(f"method={method!r} (angleproto|ge2e)")
        self.sv_weight = sv_weight

    @property
    def classifier_weight(self):
        return self.aam.weight

    def forward(self, embeddings, labels):
        aam_loss = self.aam(embeddings, labels)["total_loss"]
        sv_loss = embeddings.new_tensor(0.0)
        uniq, counts = labels.unique(return_counts=True)
        valid = uniq[counts >= 2]                       # классы с >=2 примерами в батче
        if valid.numel() >= 2:
            M = int(counts[counts >= 2].min().item())   # единый M для (N,M,D)
            groups = [embeddings[(labels == c).nonzero(as_tuple=True)[0][:M]] for c in valid.tolist()]
            x = torch.stack(groups)                     # (N, M, D)
            sv = self.sv(x)
            if torch.isfinite(sv):
                sv_loss = sv
        total = aam_loss + self.sv_weight * sv_loss
        return {"total_loss": total, "aam_loss": aam_loss, "sv_loss": sv_loss}

    def predict_logits(self, embeddings):
        return self.aam.predict_logits(embeddings)


class MultiTaskHead(ClassifierHead):
    """Multi-task: основная бинарная FocalHead + вспомогательная n_aux-классовая CE (маска -1).

    total = focal_binary + aux_weight * CE_aux(только на размеченных, label2>=0).
    predict_logits -> бинарные логиты (метрики/eval не меняются). Aux-голова структурирует фичи
    (11 классов MILK: MEL/NV/BCC/…) для более чёткой бинарной границы.
    """

    def __init__(self, n_class, embedding_dim, n_aux, gamma=2.0, aux_weight=0.3,
                 class_counts=None, cb_beta=None):
        super().__init__()
        self.n_class, self.embedding_dim = n_class, embedding_dim
        self.binary = FocalHead(n_class, embedding_dim, gamma=gamma,
                                class_counts=class_counts, cb_beta=cb_beta)
        self.aux_fc = nn.Linear(embedding_dim, n_aux)
        self.aux_weight = aux_weight

    @property
    def classifier_weight(self):
        return self.binary.classifier_weight

    def forward(self, embeddings, labels, labels2=None):
        d = self.binary(embeddings, labels)
        bin_loss = d["total_loss"]
        aux = embeddings.new_tensor(0.0)
        if labels2 is not None:
            labels2 = labels2.long()
            mask = labels2 >= 0
            if mask.any():
                aux = F.cross_entropy(self.aux_fc(embeddings[mask]), labels2[mask])
        total = bin_loss + self.aux_weight * aux
        return {"total_loss": total, "binary_loss": bin_loss, "aux11_loss": self.aux_weight * aux}

    def predict_logits(self, embeddings):
        return self.binary.predict_logits(embeddings)


# =============================================================================
# Prototype head (ProtoPNet-style, embedding space) — case-based reasoning
# =============================================================================
class ProtoHead(ClassifierHead):
    """Прототипный классификатор в пространстве эмбеддингов (ProtoPNet, «this looks like that»).

    `k` обучаемых прототипов на класс. Предсказание = «эмбеддинг похож на прототип класса c».
    Используется КОСИНУСНАЯ similarity (устойчива к масштабу фичей замороженного backbone,
    в отличие от L2 у классического ProtoPNet). Логиты — линейная комбинация similarity к
    прототипам (init: свой класс +1, чужие 0).

    Лоссы (ProtoPNet): CE + l_clst·clustering (тянуть к своему прототипу)
    + l_sep·separation (отталкивать от чужих) + l_l1·|off-class связи| (разреженность).

    Идея: структурный регуляризатор — каждый класс (в т.ч. хвостовой MAL_OTH n=9) получает
    ВЫДЕЛЕННУЮ ёмкость из `k` прототипов, а не долю в глобальной линейной границе, которую
    забивают частые классы. Ставится как cRT-стадия поверх замороженного byole093.

    scale: множитель логитов (резкость softmax; fc обучаем, так что это лишь init-масштаб).
    """

    def __init__(self, n_class, embedding_dim, k=10, l_clst=0.5, l_sep=0.5,
                 l_l1=1e-4, scale=10.0, fc_weight_path=None):
        super().__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.k = k
        self.m = k * n_class
        self.l_clst, self.l_sep, self.l_l1, self.scale = l_clst, l_sep, l_l1, scale

        self.prototypes = nn.Parameter(torch.empty(self.m, embedding_dim))
        nn.init.normal_(self.prototypes, std=0.02)

        proto_class = torch.arange(n_class).repeat_interleave(k)          # [m] класс каждого прототипа
        self.register_buffer("proto_class", proto_class)

        onehot = torch.zeros(n_class, self.m)
        onehot[proto_class, torch.arange(self.m)] = 1.0                   # [C, m] связь прототип→свой класс
        self.register_buffer("proto_onehot", onehot)

        self.fc = nn.Linear(self.m, n_class, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(onehot)                                  # init: свой +1, чужие 0
        self.ce = nn.CrossEntropyLoss()

    @property
    def classifier_weight(self):
        return self.fc.weight                                            # [C, m]; перенос между стадиями не используется

    def _sim(self, embeddings):
        e = F.normalize(embeddings, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        return e @ p.t()                                                 # [B, m] косинус в [-1, 1]

    def predict_logits(self, embeddings):
        return self.scale * self.fc(self._sim(embeddings))

    def forward(self, embeddings, labels):
        sim = self._sim(embeddings)                                      # [B, m]
        logits = self.scale * self.fc(sim)
        ce = self.ce(logits, labels)

        own = self.proto_class.unsqueeze(0) == labels.unsqueeze(1)       # [B, m] bool
        NEG = -1e4
        max_own = sim.masked_fill(~own, NEG).max(1).values               # ближайший СВОЙ прототип
        max_oth = sim.masked_fill(own, NEG).max(1).values                # ближайший ЧУЖОЙ прототип
        l_clst = (1.0 - max_own).mean()                                  # тянуть к своему (sim→1)
        l_sep = max_oth.clamp_min(0.0).mean()                            # отталкивать от чужих (sim→≤0)
        l1 = (self.fc.weight * (1.0 - self.proto_onehot)).abs().sum()    # разреженность off-class связей

        total = ce + self.l_clst * l_clst + self.l_sep * l_sep + self.l_l1 * l1
        return {"total_loss": total, "ce": ce.detach(),
                "clst": l_clst.detach(), "sep": l_sep.detach()}
