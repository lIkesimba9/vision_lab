"""Кастомные OmegaConf-resolver'ы (ТЗ §8).

Регистрируются один раз в точке входа. Пути — только через ``${paths.*}``,
хардкоды запрещены (переносимость между машинами).
"""

from __future__ import annotations

import os

from omegaconf import OmegaConf

_REGISTERED = False


def register_resolvers() -> None:
    """Идемпотентная регистрация: ``${len:}``, ``${env:}``, ``${eval_int:}``."""
    global _REGISTERED
    if _REGISTERED:
        return
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("env", lambda key, default="": os.environ.get(key, default))
    OmegaConf.register_new_resolver("eval_int", lambda expr: int(eval(expr, {"__builtins__": {}})))  # noqa: S307
    _REGISTERED = True
