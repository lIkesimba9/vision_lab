"""DINOv2-style self-supervised pretraining (DINO + iBOT + KoLeo + Sinkhorn centering).

Mirrors the structure of src/byol.py: a single nn.Module that owns the student/teacher
backbones, the projection heads and the (GPU/kornia) multi-crop augmentations, and whose
forward(batch) returns a dict of losses. The EMA (teacher) momentum and teacher-temperature
schedules are driven from the Lightning trainer (src.trainers.DINOv2Trainer).

Components (all from the DINOv2 paper, https://arxiv.org/abs/2304.07193):
  * DINO     — multi-crop self-distillation, teacher sharpening + centering, CE across crops.
  * iBOT     — masked-image modeling: student sees a block-masked global crop and must match
               the teacher's patch tokens at the masked positions.
  * KoLeo    — spreads embeddings on the unit sphere (−log nearest-neighbour distance).
  * Sinkhorn — optional Sinkhorn–Knopp teacher centering (DINOv2 default) vs EMA centering.

Backbone contract: any timm model built with num_classes=0 (exposes .forward_features,
.num_features and a pooled .forward(x)). We derive patch tokens from .forward_features and
a global vector from the pooled forward.

Note on iBOT + hierarchical backbones: iBOT aligns masks with the OUTPUT token grid. For ViT
that grid equals the patch grid (true iBOT). For Swin the output grid is downsampled (224→7×7),
so we mask the *input* in blocks aligned to that final grid (SimMIM-style input masking). Set
ibot_weight: 0 to run pure DINO+KoLeo (recommended/clean for Swin); use a ViT backbone for
textbook iBOT.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate


# --------------------------------------------------------------------------- heads


class DINOHead(nn.Module):
    """3-layer MLP (GELU) → L2-normalize → weight-normalized linear to `out_dim` prototypes."""

    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256,
                 nlayers=3, norm_last_layer=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class TokenBackbone(nn.Module):
    """Wrap a timm backbone → (global vector [B,C], patch tokens [B,N,C], grid (H,W))."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.num_prefix = getattr(backbone, "num_prefix_tokens", 0)

    def tokens(self, x):
        f = self.backbone.forward_features(x)
        if f.dim() == 4:
            # [B, H, W, C] (swin/timm channels-last) vs [B, C, H, W] (conv)
            if f.shape[-1] == self.num_features:
                B, H, W, C = f.shape
                tok = f.reshape(B, H * W, C)
            else:
                B, C, H, W = f.shape
                tok = f.flatten(2).transpose(1, 2)
            grid = (H, W)
        elif f.dim() == 3:                       # [B, (prefix+)N, C] (vit)
            tok = f[:, self.num_prefix:]
            n = tok.shape[1]
            s = int(round(math.sqrt(n)))
            grid = (s, s)
        else:
            raise ValueError(f"unexpected forward_features ndim {f.dim()}")
        glob = self.backbone(x) if self.num_prefix == 0 else f[:, 0]
        return glob, tok, grid

    def forward(self, x):
        return self.backbone(x)


# --------------------------------------------------------------------------- losses


def koleo_loss(x, eps=1e-8):
    """−mean log(nearest-neighbour distance) on L2-normalized features (spread regularizer)."""
    x = F.normalize(x, dim=-1, p=2)
    sim = x @ x.t()
    sim.fill_diagonal_(-2.0)
    nn_sim = sim.max(dim=1).values
    dist = torch.clamp(2.0 - 2.0 * nn_sim, min=eps)
    return -torch.log(dist).mean()


@torch.no_grad()
def sinkhorn_knopp(logits, n_iters=3, eps=1e-6):
    """Sinkhorn–Knopp normalization of teacher logits → doubly-normalized assignments."""
    Q = torch.exp(logits.float()).t()                # [K, B]
    Q /= Q.sum() + eps
    K_, B = Q.shape
    for _ in range(n_iters):
        Q /= (Q.sum(dim=1, keepdim=True) + eps); Q /= K_
        Q /= (Q.sum(dim=0, keepdim=True) + eps); Q /= B
    Q *= B
    return Q.t()


# --------------------------------------------------------------------------- model


class DINOv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.get("out_dim", 65536)
        self.n_local = config.get("n_local_crops", 6)
        self.student_temp = config.get("student_temp", 0.1)
        self.center_momentum = config.get("center_momentum", 0.9)
        self.center_mode = config.get("center_mode", "sinkhorn")   # 'sinkhorn' | 'ema'
        self.koleo_weight = config.get("koleo_weight", 0.1)
        self.ibot_weight = config.get("ibot_weight", 1.0)
        self.mask_ratio = config.get("mask_ratio", 0.3)
        # schedules set per-step by the trainer:
        self.register_buffer("current_tau", torch.tensor(config.get("base_tau", 0.992)))
        self.register_buffer("teacher_temp", torch.tensor(config.get("teacher_temp", 0.04)))

        # ---- backbones (student trained, teacher = EMA, no grad) ----
        student_bb = instantiate(config.backbone)
        self.student = TokenBackbone(student_bb)
        feat = self.student.num_features

        # weight_norm heads can't be deepcopy'd → build a fresh twin and copy weights.
        def make_head():
            return DINOHead(feat, self.out_dim,
                            hidden_dim=config.get("head_hidden_dim", 2048),
                            bottleneck_dim=config.get("head_bottleneck_dim", 256),
                            norm_last_layer=config.get("norm_last_layer", True))

        self.student_head = make_head()
        self.teacher_head = make_head()
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        if self.ibot_weight > 0:
            self.student_ibot_head = make_head()
            self.teacher_ibot_head = make_head()
            self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())
        else:
            self.student_ibot_head = self.teacher_ibot_head = None

        self.teacher = TokenBackbone(copy.deepcopy(student_bb))
        for p in self._teacher_params():
            p.requires_grad = False

        self.register_buffer("center", torch.zeros(1, self.out_dim))
        self.register_buffer("center_ibot", torch.zeros(1, self.out_dim))

        # ---- multi-crop augmentation (kornia, on GPU) ----
        g = config.get("global_size", 224)
        l = config.get("local_size", 96)
        mean, std = list(config.mean), list(config.std)
        self.global_grid = config.get("global_token_grid", 7)   # mask grid for iBOT
        self.augment_global = self._aug(g, (0.32, 1.0), mean, std, blur_p=1.0, solarize=True)
        self.augment_local = self._aug(l, (0.05, 0.32), mean, std, blur_p=0.5, solarize=False)

    # -- augmentation builder (shared with both views) --
    @staticmethod
    def _aug(size, scale, mean, std, blur_p, solarize):
        import kornia.augmentation as K          # train-time dep; imported lazily
        ops = [
            K.RandomResizedCrop(size=(size, size), scale=tuple(scale), ratio=(0.75, 1.3333), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=180.0, p=0.5),
            K.ColorJiggle(0.4, 0.4, 0.2, 0.1, p=0.8),
            K.RandomGrayscale(p=0.1),
            K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=blur_p),
        ]
        if solarize:
            ops.append(K.RandomSolarize(thresholds=0.1, p=0.2))
        ops.append(K.Normalize(mean=mean, std=std, p=1.0))
        return K.AugmentationSequential(*ops, data_keys=["input"], same_on_batch=False)

    def _teacher_params(self):
        mods = [self.teacher, self.teacher_head]
        if self.teacher_ibot_head is not None:
            mods.append(self.teacher_ibot_head)
        for m in mods:
            yield from m.parameters()

    def _student_ema_pairs(self):
        pairs = [(self.student, self.teacher), (self.student_head, self.teacher_head)]
        if self.student_ibot_head is not None:
            pairs.append((self.student_ibot_head, self.teacher_ibot_head))
        for s, t in pairs:
            for ps, pt in zip(s.parameters(), t.parameters()):
                yield ps, pt

    @torch.no_grad()
    def momentum_update(self):
        tau = float(self.current_tau)
        for ps, pt in self._student_ema_pairs():
            pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)

    @torch.no_grad()
    def extract_embeddings(self, images):
        return self.teacher(images)

    # -- iBOT input-space block masking aligned to a (grid×grid) output grid --
    def _block_mask(self, x):
        B, _, H, W = x.shape
        g = self.global_grid
        m = (torch.rand(B, g, g, device=x.device) < self.mask_ratio)
        m_flat = m.reshape(B, g * g)
        # never mask everything / nothing in a sample (keeps loss well-defined)
        full = m_flat.all(dim=1)
        m_flat[full, 0] = False
        up = m.float().repeat_interleave(H // g, 1).repeat_interleave(W // g, 2)  # [B,H,W]
        x_masked = x * (1.0 - up.unsqueeze(1))
        return x_masked, m_flat                       # mask: [B, g*g] over output tokens

    def _teacher_probs(self, logits):
        if self.center_mode == "sinkhorn":
            return sinkhorn_knopp(logits)
        return F.softmax((logits - self.center) / float(self.teacher_temp), dim=-1)

    def forward(self, batch):
        x = batch["image"]
        # ---- views ----
        g1, g2 = self.augment_global(x), self.augment_global(x)
        locals_ = [self.augment_local(x) for _ in range(self.n_local)]

        # ---- teacher: 2 global crops (no grad) ----
        with torch.no_grad():
            t_g1, t_tok1, _ = self.teacher.tokens(g1)
            t_g2, t_tok2, _ = self.teacher.tokens(g2)
            t_logits = self.teacher_head(torch.cat([t_g1, t_g2]))            # [2B, K]
            t_probs = self._teacher_probs(t_logits).detach()
            tp1, tp2 = t_probs.chunk(2)
            if self.ibot_weight > 0:
                t_itok = torch.cat([t_tok1, t_tok2])                         # [2B, N, C]
                t_ilogits = self.teacher_ibot_head(t_itok)
                t_iprobs = F.softmax(
                    (t_ilogits - self.center_ibot) / float(self.teacher_temp), dim=-1).detach()

        # ---- student: masked globals + local crops ----
        if self.ibot_weight > 0:
            mg1, mask1 = self._block_mask(g1)
            mg2, mask2 = self._block_mask(g2)
        else:
            mg1, mg2 = g1, g2

        s_globs, s_tok_list = [], []
        for v, want_tok in [(mg1, True), (mg2, True)] + [(c, False) for c in locals_]:
            sg, stok, _ = self.student.tokens(v)
            s_globs.append(self.student_head(sg))
            if want_tok:
                s_tok_list.append(stok)
        s_logits = s_globs                                                   # list of [B,K]
        sp = [F.log_softmax(s / self.student_temp, dim=-1) for s in s_logits]

        # ---- DINO loss: every student crop vs each teacher global (skip same view) ----
        dino_terms, n_terms = 0.0, 0
        teacher_views = [tp1, tp2]
        for ti, tprob in enumerate(teacher_views):
            for si, slog in enumerate(sp):
                if si == ti:                       # same global crop → skip
                    continue
                dino_terms = dino_terms - (tprob * slog).sum(dim=-1).mean()
                n_terms += 1
        dino = dino_terms / max(n_terms, 1)

        out = {"dino_loss": dino}

        # ---- iBOT loss: student masked-patch logits vs teacher patch probs at masked pos ----
        if self.ibot_weight > 0:
            ti1 = t_iprobs[:x.shape[0]]
            ti2 = t_iprobs[x.shape[0]:]
            ibot = 0.0
            for stok, tprob, mask in [(s_tok_list[0], ti1, mask1), (s_tok_list[1], ti2, mask2)]:
                s_il = F.log_softmax(self.student_ibot_head(stok) / self.student_temp, dim=-1)
                ce = -(tprob * s_il).sum(dim=-1)              # [B, N]
                m = mask.float()
                ibot = ibot + (ce * m).sum() / m.sum().clamp(min=1.0)
            out["ibot_loss"] = ibot / 2.0

        # ---- KoLeo on student global-crop embeddings ----
        if self.koleo_weight > 0:
            with torch.no_grad() if False else torch.enable_grad():
                g_emb = self.student(torch.cat([mg1, mg2]) if self.ibot_weight > 0
                                     else torch.cat([g1, g2]))
            out["koleo_loss"] = koleo_loss(g_emb)

        total = out["dino_loss"]
        if "ibot_loss" in out:
            total = total + self.ibot_weight * out["ibot_loss"]
        if "koleo_loss" in out:
            total = total + self.koleo_weight * out["koleo_loss"]
        out["total_loss"] = total

        # update EMA centers (teacher) for the EMA centering path
        self._update_center(t_logits, t_ilogits if self.ibot_weight > 0 else None)
        return out

    @torch.no_grad()
    def _update_center(self, t_logits, t_ilogits):
        b = self.center_momentum
        self.center.mul_(b).add_(t_logits.mean(dim=0, keepdim=True), alpha=1 - b)
        if t_ilogits is not None:
            flat = t_ilogits.reshape(-1, self.out_dim).mean(dim=0, keepdim=True)
            self.center_ibot.mul_(b).add_(flat, alpha=1 - b)
