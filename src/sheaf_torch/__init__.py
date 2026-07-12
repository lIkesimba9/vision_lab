"""Sheaf-ADMM (PyTorch-порт) — координационная голова над картой признаков backbone'а.

См. ../../sheaf_admm/ANALYSIS.md (разбор статьи) и INTEGRATION.md (дизайн порта).
"""

from .model import SheafADMMModule

__all__ = ["SheafADMMModule"]
