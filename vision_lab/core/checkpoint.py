"""Чекпоинт-контракт (ТЗ §4.4): чекпоинт ЛЮБОЙ стадии грузится в чистый timm-бэкбон.

Возможен он потому, что имена атрибутов в библиотеке фиксированы:
обёртки бэкбонов держат timm-модель в атрибуте ``net``, трейнеры — компоненты
в ``backbone``/``method``. Реестр :data:`BACKBONE_PREFIXES` — ЕДИНСТВЕННОЕ
место, где перечислены префиксы стадий; новые стадии добавляют сюда строку
(и тест) вместо копипасты срезалок по скриптам.

Политика загрузки: ``torch.load(weights_only=True)`` по умолчанию; небезопасная
загрузка — только явным флагом для доверенных Lightning-чекпоинтов.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

log = logging.getLogger(__name__)

#: Реестр префиксов всех стадий. Порядок = приоритет (первое совпадение выигрывает):
#: для BYOL-чекпоинта берём online-бэкбон (проверено прототипом),
#: для DINO/I-JEPA — teacher (EMA лучше student).
BACKBONE_PREFIXES: tuple[str, ...] = (
    "method.backbone.net.",                # SSLTrainer/BYOL: online-энкодер (приоритет)
    "method.teacher.module.net.",          # SSLTrainer/DINO, I-JEPA: EMA-teacher (приоритет)
    "method.student.net.",                 # SSLTrainer/DINO: student (фолбэк)
    "method.teacher.module.backbone.net.", # SSLTrainer/BYOL: EMA-teacher (фолбэк)
    "backbone.net.",                       # ClassificationTrainer
    "net.",                                # голая обёртка (EmbeddingBackbone.state_dict())
)


@dataclass(frozen=True)
class LoadReport:
    """Результат загрузки бэкбона — для логов и тестов."""

    prefix: str | None  # какой префикс сработал (None = state_dict уже «голый»)
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def strip_prefixes(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...] = BACKBONE_PREFIXES,
) -> tuple[dict[str, torch.Tensor], str | None]:
    """Находит первый подходящий префикс и срезает его.

    Возвращает (веса бэкбона, сработавший префикс | None). Если ни один префикс
    не найден — считаем, что это уже «голый» timm state_dict, отдаём как есть.
    """
    prefix = next((p for p in prefixes if any(k.startswith(p) for k in state_dict)), None)
    if prefix is None:
        return dict(state_dict), None
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}, prefix


def load_state_dict_file(
    path: str | Path, *, weights_only: bool = True
) -> dict[str, torch.Tensor]:
    """Читает .pt/.pth/.ckpt и разворачивает Lightning-обёртку ``{"state_dict": ...}``."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=weights_only)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Ожидался state_dict (dict), получено {type(ckpt)} из {path}")
    return ckpt


def load_backbone(
    net: torch.nn.Module,
    source: str | Path | dict[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...] = BACKBONE_PREFIXES,
    weights_only: bool = True,
    strict: bool = False,
) -> LoadReport:
    """Грузит чекпоинт любой стадии в чистый бэкбон (timm-модель).

    ``source`` — путь к чекпоинту или уже загруженный state_dict.
    ``weights_only=False`` — только для доверенных Lightning-чекпоинтов
    (гиперпараметры в них могут содержать произвольные объекты).
    """
    sd = load_state_dict_file(source, weights_only=weights_only) if not isinstance(source, dict) else source
    stripped, prefix = strip_prefixes(sd, prefixes)
    missing, unexpected = net.load_state_dict(stripped, strict=strict)
    report = LoadReport(prefix=prefix, missing=tuple(missing), unexpected=tuple(unexpected))
    log.info(
        "load_backbone: prefix=%r missing=%d unexpected=%d",
        report.prefix, len(report.missing), len(report.unexpected),
    )
    return report


# ---------------------------------------------------------------------------
# Перенос FC-весов головы между стадиями (порт из прототипа losses/base.py)
# ---------------------------------------------------------------------------

def extract_fc_weights(
    source: str | Path | dict | torch.Tensor,
    n_class: int,
    embedding_dim: int,
    *,
    weights_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Достаёт [C, D]-матрицу весов классификатора (+ опц. bias) из разных источников.

    ``source``: путь к .pt/.pth (сырой тензор, state_dict или Lightning-чекпоинт),
    dict (state_dict) либо готовый Tensor [C, D]. Возвращает (weight, bias | None).
    """
    if isinstance(source, (str, Path)):
        source = torch.load(str(source), map_location="cpu", weights_only=weights_only)

    if isinstance(source, torch.Tensor):
        return _check_fc_shape(source, n_class, embedding_dim), None

    if not isinstance(source, dict):
        raise TypeError(f"Неподдерживаемый источник FC-весов: {type(source)}")

    state = source.get("state_dict", source)
    want = (n_class, embedding_dim)

    # 1) приоритет — точные имена
    for wkey in ("classifier_weight", "fc.weight", "weight",
                 "head.fc.weight", "criterion.fc.weight", "criterion.weight"):
        if wkey in state and tuple(state[wkey].shape) == want:
            bias = state.get(wkey.rsplit("weight", 1)[0] + "bias")
            return state[wkey], bias

    # 2) иначе — любой [C, D]-тензор, чей ключ оканчивается на 'weight'
    for k, v in state.items():
        if k.endswith("weight") and torch.is_tensor(v) and tuple(v.shape) == want:
            return v, state.get(k[: -len("weight")] + "bias")

    raise KeyError(
        f"FC-веса формы {want} не найдены. "
        f"Ключи: {list(state)[:8]}{'...' if len(state) > 8 else ''}"
    )


def _check_fc_shape(t: torch.Tensor, n_class: int, embedding_dim: int) -> torch.Tensor:
    if tuple(t.shape) != (n_class, embedding_dim):
        raise ValueError(
            f"Ожидалась форма FC-весов {(n_class, embedding_dim)}, получено {tuple(t.shape)}"
        )
    return t
