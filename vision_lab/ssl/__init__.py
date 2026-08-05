"""SSL-методы (§4.3): единый контракт SSLMethod + BYOL, DINOv2, SimCLR,
MoCo v3, SimSiam, MAE, SimMIM, GPU-вьюхи."""

from vision_lab.ssl.base import MomentumTeacher, SSLMethod
from vision_lab.ssl.byol import BYOL, BYOLTriplet, byol_cosine_loss, shuffle_within_groups
from vision_lab.ssl.components import (
    MLP,
    DINOHead,
    block_mask,
    koleo_loss,
    patchify,
    sinkhorn_knopp,
)
from vision_lab.ssl.dinov2 import DINOv2
from vision_lab.ssl.gpu_augs import MultiViewAugment, build_ssl_views, build_view_pipeline
from vision_lab.ssl.mae import MAE, random_masking
from vision_lab.ssl.moco import MoCoV3
from vision_lab.ssl.simclr import SimCLR, nt_xent_loss
from vision_lab.ssl.simmim import SimMIM
from vision_lab.ssl.simsiam import SimSiam

__all__ = [
    "SSLMethod",
    "MomentumTeacher",
    "BYOL",
    "BYOLTriplet",
    "byol_cosine_loss",
    "shuffle_within_groups",
    "DINOv2",
    "SimCLR",
    "nt_xent_loss",
    "MoCoV3",
    "SimSiam",
    "MAE",
    "random_masking",
    "SimMIM",
    "MLP",
    "DINOHead",
    "block_mask",
    "koleo_loss",
    "patchify",
    "sinkhorn_knopp",
    "MultiViewAugment",
    "build_ssl_views",
    "build_view_pipeline",
]
