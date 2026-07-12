"""Головы-классификаторы: единый контракт §4.2 + зоопарк."""

from vision_lab.heads.base import ClassifierHead
from vision_lab.heads.classification import (
    AAMHead,
    AAMTripletHead,
    CosineCEHead,
    DBMHead,
    FocalHead,
    LDAMHead,
    LinearHead,
    LogitAdjustHead,
    SeesawHead,
    SubCenterHead,
    VSHead,
)
from vision_lab.heads.multitask import MultiTaskHead

__all__ = [
    "ClassifierHead",
    "LinearHead",
    "CosineCEHead",
    "AAMHead",
    "SubCenterHead",
    "LDAMHead",
    "FocalHead",
    "LogitAdjustHead",
    "VSHead",
    "SeesawHead",
    "DBMHead",
    "AAMTripletHead",
    "MultiTaskHead",
]
