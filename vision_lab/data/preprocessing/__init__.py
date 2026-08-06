"""Доменная предобработка: детерминированные шаги до аугментаций (ТЗ §7.4)."""

from vision_lab.data.preprocessing.color_constancy import ColorConstancy, shades_of_gray
from vision_lab.data.preprocessing.source_align import (
    GLOBAL_KEY,
    SourceAlignment,
    compute_source_stats,
    load_source_stats,
)
from vision_lab.data.preprocessing.vignette import VignetteCrop, vignette_bbox

__all__ = [
    "ColorConstancy",
    "shades_of_gray",
    "GLOBAL_KEY",
    "SourceAlignment",
    "compute_source_stats",
    "load_source_stats",
    "VignetteCrop",
    "vignette_bbox",
]
