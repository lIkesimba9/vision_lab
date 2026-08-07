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
- softmax: `LinearHead` (ce/bce/balanced_softmax; weighted CE через
  `weighting="inverse"|"cb"`), `PolyHead` (Poly-1);
- angular: `CosineCEHead`, `AAMHead`, `CosFaceHead`, `SubCenterHead`;
- long-tail: `FocalHead`, `LDAMHead`, `LogitAdjustHead`, `SeesawHead`, `VSHead`, `DBMHead`;
- noise-robust: `GCEHead`, `SCEHead` (шумные метки);
- метрические: `AAMTripletHead` (требует PK-сэмплер — позитивы в батче).

Новая голова: наследуй `ClassifierHead`, композируй чистые функции из
`heads/primitives.py` (cosine_logits, additive_angular_margin, ldam_margins,
class_balanced_weights, logit_prior, masked_cross_entropy). Без глубокого наследования.

## Типы пайплайнов (§5.1)
- **Бинарная** = multiclass с `n_class=2` (или `MultiLabelHead(n_class=1)` —
  один логит, плоский {0,1,-1}-таргет).
- **Multi-label**: `MultiLabelHead` (bce/asl), таргет — мульти-хот `(B, C)` из
  {0,1,-1}, `-1` маскируется поэлементно. Данные:
  `ManifestDataset(label_kind="multilabel")` (колонка list<string>). Трейнер:
  `ClassificationTrainer(task="multilabel")` — Multilabel-метрики в val.
  Шаблон: `configs/experiment/classification_multilabel.yaml`.
- **Multi-task**: см. ниже.
- **Иерархическая**: `hierarchical_head(taxonomy, embedding_dim, ...)` —
  MultiTaskHead с под-головой на уровень; `ManifestDataset(taxonomy=...)`
  кладёт в батч `label_<level>` на каждый уровень (`-1` тоньше метки).

## Multi-task / иерархия
`MultiTaskHead(heads={"main": ..., "aux": ...}, primary="main", weights={...})` —
композиция под-голов, каждая со своим `target_key` (`label`, `label_aux`).
Батч несёт доп. ключи `label_<task>` (§7.2); `-1` маскируется автоматически.
Val-метрики трейнера считаются по `head.target_key` (у MultiTaskHead — ключ
primary-головы).

## Перенос SSL-бэкбона
`EmbeddingBackbone(model_name=..., ckpt_path="<ssl_ckpt>")` — префикс стадии
срезается автоматически (`BACKBONE_PREFIXES`). `backbone_lr`: None=finetune,
0=заморозка, малое=доменная адаптация.

Конфиг-шаблон: `configs/experiment/classification_ce.yaml`.
