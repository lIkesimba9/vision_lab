"""OmegaConf/Hydra resolver registration, in one idempotent place.

The `len` and `config_names` resolvers used to be registered as import-time side
effects inside `train.py`, which is why `predict.py` (and a dozen other modules)
did `import train  # noqa` purely to trigger them — brittle, and it re-runs the
whole training entrypoint's imports. Call `register_all_resolvers()` instead;
it is safe to call any number of times, from any process.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from src.paths import register_resolvers as _register_derma_root


def _format_config_names(overrides_str: str) -> str:
    """`${config_names:...}` — build a run tag from Hydra override strings."""
    if not overrides_str:
        return "no_override"
    return "_".join(x.split("=")[1] for x in overrides_str.split(",") if "=" in x)


def register_all_resolvers() -> None:
    """Register every custom resolver the configs rely on. Idempotent."""
    _register_derma_root()  # ${derma_root:} — already uses replace=True
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("config_names", _format_config_names, replace=True)
