"""Доменно-нейтральный фундамент: батч-контракт, DDP, чекпоинты, расписания, оптимизация."""

from vision_lab.core.batch import MISSING_LABEL, Batch, target_view
from vision_lab.core.checkpoint import (
    BACKBONE_PREFIXES,
    LoadReport,
    extract_fc_weights,
    load_backbone,
    strip_prefixes,
)
from vision_lab.core.dist import (
    DistributedBatchSamplerMixin,
    all_gather_grad,
    dist_info,
    is_main,
    rank,
    world_size,
)
from vision_lab.core.optim import default_no_decay, param_groups
from vision_lab.core.runtime import configure_threads
from vision_lab.core.schedules import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    LinearWarmupConstant,
    Schedule,
    ScheduleDriver,
    build_warmup_cosine,
)

__all__ = [
    "MISSING_LABEL",
    "Batch",
    "target_view",
    "BACKBONE_PREFIXES",
    "LoadReport",
    "extract_fc_weights",
    "load_backbone",
    "strip_prefixes",
    "DistributedBatchSamplerMixin",
    "all_gather_grad",
    "dist_info",
    "is_main",
    "rank",
    "world_size",
    "default_no_decay",
    "param_groups",
    "configure_threads",
    "ConstantSchedule",
    "CosineSchedule",
    "LinearSchedule",
    "LinearWarmupConstant",
    "Schedule",
    "ScheduleDriver",
    "build_warmup_cosine",
]
