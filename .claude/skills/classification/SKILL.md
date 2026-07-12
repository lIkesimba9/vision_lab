---
name: classification
description: Как обучать классификацию в vision-lab — контракт ClassifierHead, зоопарк голов, multi-task/иерархия, baseline-first, перенос SSL-бэкбона.
---

# Классификация в vision-lab

## Контракт головы (`vision_lab/heads/base.py`)
`ClassifierHead`:
- `forward(embeddings, targets: Mapping[str, Tensor]) -> {"total_loss", ...}` —
  голова берёт свою метку по `target_key` из словаря таргетов, маскирует `-1`;
- `predict_logits(embeddings) -> (B, C)` (без маржина, для метрик/инференса);
- `classifier_weight -> (C, D)`; `load_fc_weights`/`save_fc_weights` — перенос FC.

## Baseline-first (§5.1)
Эталон каждого пайплайна — `LinearHead(mode="ce")` + рецепт `heavy_v1`. Зоопарк
доступен, но **каждый эксперимент сравнивается с этим бейзлайном**, не вместо.
Урок прототипа: простой CE обыграл весь long-tail зоопарк.

## Зоопарк голов (`vision_lab/heads/classification.py`)
- softmax: `LinearHead` (ce/bce/balanced_softmax);
- angular: `CosineCEHead`, `AAMHead`, `SubCenterHead`;
- long-tail: `FocalHead`, `LDAMHead`, `LogitAdjustHead`, `SeesawHead`, `VSHead`, `DBMHead`;
- метрические: `AAMTripletHead` (требует PK-сэмплер — позитивы в батче).

Новая голова: наследуй `ClassifierHead`, композируй чистые функции из
`heads/primitives.py` (cosine_logits, additive_angular_margin, ldam_margins,
class_balanced_weights, logit_prior, masked_cross_entropy). Без глубокого наследования.

## Multi-task / иерархия
`MultiTaskHead(heads={"main": ..., "aux": ...}, primary="main", weights={...})` —
композиция под-голов, каждая со своим `target_key` (`label`, `label_aux`).
Батч несёт доп. ключи `label_<task>` (§7.2); `-1` маскируется автоматически.

## Перенос SSL-бэкбона
`EmbeddingBackbone(model_name=..., ckpt_path="<ssl_ckpt>")` — префикс стадии
срезается автоматически (`BACKBONE_PREFIXES`). `backbone_lr`: None=finetune,
0=заморозка, малое=доменная адаптация.

Конфиг-шаблон: `configs/experiment/classification_ce.yaml`.
