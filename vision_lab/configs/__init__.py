"""Эталонные Hydra-конфиги-шаблоны + resolver'ы (§8).

Task-репа композирует эти дефолты и свои оверрайды. Компоненты инстанцируются
через ``_target_``; никаких if-веток по именам конфигов.
"""

from pathlib import Path

from vision_lab.configs.resolvers import register_resolvers

#: Корень эталонных конфигов (для Hydra search path в task-репах).
CONFIG_ROOT = Path(__file__).parent

__all__ = ["register_resolvers", "CONFIG_ROOT"]
