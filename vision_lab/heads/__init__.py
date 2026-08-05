"""Головы-классификаторы: единый контракт §4.2 + зоопарк."""

from vision_lab.heads.base import ClassifierHead
from vision_lab.heads.classification import (
    AAMHead,
    AAMTripletHead,
    CosFaceHead,
    CosineCEHead,
    DBMHead,
    FocalHead,
    GCEHead,
    LDAMHead,
    LinearHead,
    LogitAdjustHead,
    MultiLabelHead,
    PolyHead,
    SCEHead,
    SeesawHead,
    SubCenterHead,
    VSHead,
)
from vision_lab.heads.multitask import MultiTaskHead, hierarchical_head

__all__ = [
    "ClassifierHead",
    "LinearHead",
    "MultiLabelHead",
    "PolyHead",
    "CosineCEHead",
    "AAMHead",
    "CosFaceHead",
    "SubCenterHead",
    "LDAMHead",
    "FocalHead",
    "LogitAdjustHead",
    "VSHead",
    "SeesawHead",
    "DBMHead",
    "GCEHead",
    "SCEHead",
    "AAMTripletHead",
    "MultiTaskHead",
    "hierarchical_head",
]
