"""Свободные лоссы: метрические (triplet, supcon) и SSL-лоссы (в ssl/)."""

from vision_lab.losses.metric import SupConLoss, TripletSemiHardLoss, pairwise_distance

__all__ = ["SupConLoss", "TripletSemiHardLoss", "pairwise_distance"]
