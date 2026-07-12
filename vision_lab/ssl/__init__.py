"""SSL-методы (§4.3): единый контракт SSLMethod + BYOL, DINOv2, GPU-вьюхи."""

from vision_lab.ssl.base import MomentumTeacher, SSLMethod
from vision_lab.ssl.byol import BYOL, BYOLTriplet, byol_cosine_loss, shuffle_within_groups
from vision_lab.ssl.components import MLP, DINOHead, koleo_loss, sinkhorn_knopp
from vision_lab.ssl.dinov2 import DINOv2
from vision_lab.ssl.gpu_augs import MultiViewAugment, build_ssl_views, build_view_pipeline

__all__ = [
    "SSLMethod",
    "MomentumTeacher",
    "BYOL",
    "BYOLTriplet",
    "byol_cosine_loss",
    "shuffle_within_groups",
    "DINOv2",
    "MLP",
    "DINOHead",
    "koleo_loss",
    "sinkhorn_knopp",
    "MultiViewAugment",
    "build_ssl_views",
    "build_view_pipeline",
]
