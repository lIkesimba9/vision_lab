from torch import nn
import torch
import torch.nn.functional as F
from src.losses.triplet_loss import TripletLoss


def byol_cosine_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()


def byol_multicrop_loss(output):
    """BYOL-член с поддержкой multi-crop: каждая online-предсказанная вьюха тянется к каждой
    глобальной target-вьюхе (для двух глобальных — кросс i!=j, как в симметричном BYOL; локальные
    (output['pl_o'], опц.) — к обеим глобальным). Без 'pl_o' эквивалентно 0.5*(p1·z2 + p2·z1)."""
    p_glob = [output["p1_o"], output["p2_o"]]
    z_glob = [output["z1_t"], output["z2_t"]]
    terms = []
    for i, p in enumerate(p_glob):
        for j, z in enumerate(z_glob):
            if i != j:
                terms.append(byol_cosine_loss(p, z))
    pl = output.get("pl_o")
    if pl is not None:
        for k in range(pl.size(0)):
            for z in z_glob:
                terms.append(byol_cosine_loss(pl[k], z))
    return sum(terms) / len(terms)


class ByolLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, batch):
        p1_o = output["p1_o"]
        p2_o = output["p2_o"]
        z1_t = output["z1_t"]
        z2_t = output["z2_t"]

        loss1 = byol_cosine_loss(p1_o, z2_t)
        loss2 = byol_cosine_loss(p2_o, z1_t)
        byol_loss = 0.5 * (loss1 + loss2)

        return {
            "byol_loss": byol_loss,
            "total_loss": byol_loss
        }


class ByolTripletLoss(nn.Module):
    def __init__(self, triplet_weight=1.0, byol_weight=1.0):
        super().__init__()
        self.byol_loss = ByolLoss()
        self.triplet_loss = TripletLoss()
        self.triplet_weight = triplet_weight
        self.byol_weight = byol_weight

    def forward(self, output, batch):
        byol_values = self.byol_loss(output, batch)

        z1_o = output["z1_o"]
        z2_o = output["z2_o"]

        labels = batch["label"].view(-1).long()

        emb = torch.cat([z1_o, z2_o], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        mask = labels != -1
        emb = emb[mask]
        labels = labels[mask]

        triplet_loss = emb.new_tensor(0.0)

        # triplet loss требует минимум 2 класса и хотя бы одну пару положительных примеров
        if emb.size(0) >= 3 and labels.unique().numel() >= 2:
            _, counts = labels.unique(return_counts=True)
            if (counts >= 2).any():
                emb = F.normalize(emb, dim=1)
                triplet_loss = self.triplet_loss(emb, labels)
                if not torch.isfinite(triplet_loss):
                    triplet_loss = emb.new_tensor(0.0)

        total_loss = self.byol_weight * byol_values["byol_loss"] + self.triplet_weight * triplet_loss

        return {
            "byol_loss": self.byol_weight * byol_values["byol_loss"],
            "triplet_loss": self.triplet_weight * triplet_loss,
            "total_loss": total_loss
        }