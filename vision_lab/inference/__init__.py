"""Единая точка инференса + flip-TTA (§5.5)."""

from vision_lab.inference.predictor import Predictor
from vision_lab.inference.tta import TTA_VIEWS, tta_logits

__all__ = ["Predictor", "TTA_VIEWS", "tta_logits"]
