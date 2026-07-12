"""Единый контракт головы-классификатора (ТЗ §4.2).

Голова владеет весами классификатора И функцией потерь. Backbone отдаёт только
эмбеддинги, поэтому один ``ClassificationTrainer`` обслуживает
CE/BCE/AAM/SubCenter/Focal/LDAM/... без разветвлений, инференс единообразен для
любого чекпоинта, а FC-веса переносимы между стадиями отдельно от бэкбона.

Ключевое отличие от прототипа: ``forward(embeddings, targets)`` где
``targets`` — всегда ``Mapping[str, Tensor]`` (не голый тензор и не
``labels2=None``). Голова сама берёт нужный ключ через ``target_key`` — это
убирает ветку ``if "label11" in batch`` из трейнера и делает multi-task
композицией голов.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ClassifierHead(nn.Module):
    """База всех голов-классификаторов.

    Подклассы обязаны:

    * хранить обучаемую ``[C, D]``-матрицу и вернуть её из ``classifier_weight``;
    * реализовать ``forward(embeddings, targets) -> dict`` (ключ ``total_loss``);
    * реализовать ``predict_logits(embeddings) -> [B, C]`` (без маржина).

    ``target_key`` — какой ключ батча голова читает как свою метку (по умолчанию
    ``"label"``); multi-task-подголовы задают свой (``"label_diag3"`` и т.п.).
    """

    n_class: int
    embedding_dim: int
    target_key: str = "label"

    # --- обязательный контракт подкласса -------------------------------------
    @property
    def classifier_weight(self) -> torch.Tensor:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def predict_logits(self, embeddings: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, embeddings: torch.Tensor,
                targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    # --- удобный доступ к своей метке ----------------------------------------
    def take_target(self, targets: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Достаёт метку головы из словаря таргетов по ``target_key``."""
        if self.target_key not in targets:
            raise KeyError(
                f"{type(self).__name__}: в таргетах нет ключа {self.target_key!r} "
                f"(есть: {list(targets)})"
            )
        return targets[self.target_key].long()

    # --- перенос FC-весов между стадиями -------------------------------------
    @torch.no_grad()
    def load_fc_weights(self, source, load_bias: bool = True) -> None:
        """Грузит ``[C, D]``-веса классификатора (+ опц. bias) из ``source``."""
        # ленивый импорт: разрывает цикл heads.base -> core.checkpoint -> core.module -> heads.base
        from vision_lab.core.checkpoint import extract_fc_weights

        weight, bias = extract_fc_weights(source, self.n_class, self.embedding_dim)
        self.classifier_weight.copy_(weight.to(self.classifier_weight))
        if load_bias and bias is not None:
            b = getattr(self, "classifier_bias", None)
            if b is not None:
                b.copy_(bias.to(b))

    @torch.no_grad()
    def save_fc_weights(self, path) -> None:
        """Сохраняет ``{'weight': [C, D], 'bias': [C]?}`` для следующей стадии."""
        out = {"weight": self.classifier_weight.detach().cpu()}
        b = getattr(self, "classifier_bias", None)
        if b is not None:
            out["bias"] = b.detach().cpu()
        torch.save(out, str(path))
