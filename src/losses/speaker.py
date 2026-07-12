"""Speaker-verification метрик-лоссы (GE2E / Angular Prototypical), адаптированные под
иерархический SSL. Вход (N, M, D): N классов (для нас — диагнозов уровня), M примеров на класс,
D — размерность эмбеддинга. PK-семплер по diag_3 даёт ровно такую PK-структуру (N диагнозов × K).

Источники:
  GE2E       — https://arxiv.org/abs/1710.10467 (адапт. cvqluu/GE2E-Loss)
  AngleProto — https://arxiv.org/abs/2003.11982 (адапт. clovaai/voxceleb_trainer)
w,b — обучаемые (масштаб/сдвиг косинуса), попадают в оптимизатор через criterion.parameters().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleProtoLoss(nn.Module):
    """Angular Prototypical (быстрый, векторизованный). x: (N, M, D), M>=2.
    anchor = среднее M-1 примеров класса, positive = оставшийся; CE по матрице косинусов."""

    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, _label=None):
        assert x.size(1) >= 2, "AngleProto требует M>=2"
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        n = out_anchor.size(0)
        cos = F.cosine_similarity(
            out_positive.unsqueeze(-1).expand(-1, -1, n),
            out_anchor.unsqueeze(-1).expand(-1, -1, n).transpose(0, 2),
        )
        self.w.data.clamp_(1e-6)
        cos = cos * self.w + self.b
        label = torch.arange(n, device=cos.device)
        return self.criterion(cos, label)


class GE2ELoss(nn.Module):
    """Generalized End-to-End (softmax|contrast). x: (N, M, D), M>=2.
    Центроид класса считается без текущего примера (как в статье)."""

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        assert loss_method in ("softmax", "contrast")
        self.loss_method = loss_method

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1:]))
        excl = torch.mean(excl, 0)
        new_centroids = [excl if i == spkr else c for i, c in enumerate(centroids)]
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        rows = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utt in enumerate(speaker):
                nc = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                cs_row.append(torch.clamp(
                    torch.mm(utt.unsqueeze(0), nc.transpose(0, 1))
                    / (torch.norm(utt) * torch.norm(nc, dim=1)), 1e-6))
            rows.append(torch.cat(cs_row, dim=0))
        return torch.stack(rows)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        n, m, _ = dvecs.shape
        return torch.stack([
            torch.stack([-F.log_softmax(cos_sim_matrix[j, i], 0)[j] for i in range(m)])
            for j in range(n)
        ])

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        n, m, _ = dvecs.shape
        L = []
        for j in range(n):
            row = []
            for i in range(m):
                sig = torch.sigmoid(cos_sim_matrix[j, i])
                excl = torch.cat((sig[:j], sig[j + 1:]))
                row.append(1.0 - torch.sigmoid(cos_sim_matrix[j, i, j]) + torch.max(excl))
            L.append(torch.stack(row))
        return torch.stack(L)

    def forward(self, x, _label=None):
        assert x.size(1) >= 2, "GE2E требует M>=2"
        centroids = torch.mean(x, 1)
        cos = self.calc_cosine_sim(x, centroids)
        self.w.data.clamp_(1e-6)
        cos = self.w * cos + self.b
        embed = self.embed_loss_softmax if self.loss_method == "softmax" else self.embed_loss_contrast
        return embed(x, cos).mean()
