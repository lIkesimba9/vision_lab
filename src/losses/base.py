"""Единый интерфейс головы-классификатора (она же критерий/лосс).

Раньше существовало два несовместимых мира:
  * "логиты в модели"   — модель отдавала {"logits"}, лосс принимал logits (CELoss/BCELoss);
  * "логиты в лоссе"     — модель отдавала эмбеддинги, веса классификатора жили внутри лосса
                           (AAMSoftmax/SubCenter/...), логиты доставались через get_logits().

Теперь единый контракт. Backbone (любой timm-модели) отдаёт ТОЛЬКО эмбеддинги, а голова
владеет весами классификатора И функцией потерь:

    head.forward(embeddings, labels) -> {"total_loss": Tensor, <доп. компоненты>: Tensor}
    head.predict_logits(embeddings)  -> Tensor [B, C]   # БЕЗ маржина, для метрик/инференса
    head.classifier_weight           -> Tensor [C, D]   # для переноса между стадиями
    head.load_fc_weights(source)     -> None            # ЕДИНАЯ загрузка FC-весов прошлой стадии

`load_fc_weights` решает проблему "не все лоссы умеют грузить полносвязные веса": теперь ЛЮБАЯ
голова грузит [C, D]-матрицу (и bias, если есть) из чекпоинта прошлой стадии одинаково.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _extract_fc_tensor(source, n_class: int, embedding_dim: int):
    """Достаёт [C, D]-матрицу весов классификатора (+ опц. bias) из разных источников.

    source может быть:
      * путь (str/Path) к .pt/.pth: сырой тензор, dict со state_dict или Lightning-чекпоинт;
      * dict (state_dict);
      * torch.Tensor [C, D].

    Возвращает (weight[C, D], bias[C] | None). Ищет ключи, оканчивающиеся на
    'weight'/'fc.weight' с подходящей формой; bias — соответствующий 'bias'/'fc.bias'.
    """
    if isinstance(source, (str,)) or hasattr(source, "__fspath__"):
        source = torch.load(str(source), map_location="cpu", weights_only=False)

    if isinstance(source, torch.Tensor):
        return _check_shape(source, n_class, embedding_dim), None

    if not isinstance(source, dict):
        raise TypeError(f"Неподдерживаемый источник FC-весов: {type(source)}")

    # Lightning-чекпоинт
    state = source.get("state_dict", source)

    want = (n_class, embedding_dim)
    weight, bias = None, None

    # 1) приоритет — точные имена
    for wkey in ("classifier_weight", "fc.weight", "weight",
                 "criterion.fc.weight", "criterion.weight"):
        if wkey in state and tuple(state[wkey].shape) == want:
            weight = state[wkey]
            bkey = wkey.rsplit("weight", 1)[0] + "bias"
            bias = state.get(bkey)
            break

    # 2) иначе — любой [C, D]-тензор, чей ключ оканчивается на 'weight'
    if weight is None:
        for k, v in state.items():
            if k.endswith("weight") and torch.is_tensor(v) and tuple(v.shape) == want:
                weight = v
                bias = state.get(k[: -len("weight")] + "bias")
                break

    if weight is None:
        raise KeyError(
            f"FC-веса формы {want} не найдены в источнике. "
            f"Ключи: {list(state)[:8]}{'...' if len(state) > 8 else ''}"
        )
    return weight, bias


def _check_shape(t, n_class, embedding_dim):
    if tuple(t.shape) != (n_class, embedding_dim):
        raise ValueError(
            f"Ожидалась форма FC-весов {(n_class, embedding_dim)}, получено {tuple(t.shape)}"
        )
    return t


class ClassifierHead(nn.Module):
    """База для всех голов-классификаторов с единым интерфейсом.

    Подклассы обязаны:
      * хранить обучаемую [C, D]-матрицу и вернуть её из ``classifier_weight``;
      * реализовать ``forward(embeddings, labels) -> dict`` (ключ 'total_loss' обязателен);
      * реализовать ``predict_logits(embeddings) -> [B, C]`` (без маржина).

    Опциональный ``fc_weight_path`` в __init__ подклассов → вызвать ``self.load_fc_weights``
    после инициализации весов для переноса классификатора с предыдущей стадии.
    """

    n_class: int
    embedding_dim: int

    # --- обязательный контракт подкласса -------------------------------------
    @property
    def classifier_weight(self) -> torch.Tensor:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def predict_logits(self, embeddings: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    # обратная совместимость со старым именем
    @torch.no_grad()
    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.predict_logits(embeddings)

    # --- единая загрузка/сохранение FC-весов ---------------------------------
    @torch.no_grad()
    def load_fc_weights(self, source, load_bias: bool = True) -> None:
        """Грузит [C, D]-веса классификатора (+ опц. bias) из ``source`` в эту голову."""
        weight, bias = _extract_fc_tensor(source, self.n_class, self.embedding_dim)
        self.classifier_weight.copy_(weight.to(self.classifier_weight))
        if load_bias and bias is not None:
            b = getattr(self, "classifier_bias", None)
            if b is not None:
                b.copy_(bias.to(b))
        print(f"[{type(self).__name__}] загружены FC-веса {tuple(weight.shape)}"
              + ("" if bias is None or not load_bias else " (+bias)"))

    @torch.no_grad()
    def save_fc_weights(self, path) -> None:
        """Сохраняет {'weight': [C, D], 'bias': [C]?} для переноса на следующую стадию."""
        out = {"weight": self.classifier_weight.detach().cpu()}
        b = getattr(self, "classifier_bias", None)
        if b is not None:
            out["bias"] = b.detach().cpu()
        torch.save(out, str(path))
