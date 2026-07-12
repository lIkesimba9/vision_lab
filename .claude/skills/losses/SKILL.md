---
name: losses
description: Лоссы и головы vision-lab — где что живёт, чистые примитивы, метрик-лоссы, маскирование -1, baseline-first.
---

# Лоссы в vision-lab

Голова **владеет** весами классификатора И лоссом (§4.2) — поэтому один трейнер
на любую голову. Свободные метрик-лоссы — в `losses/`.

## Где что
- `heads/classification.py` — головы с лоссом внутри (CE/AAM/LDAM/Focal/…).
- `heads/primitives.py` — **чистые функции**, переиспользуй их в новых головах:
  `cosine_logits`, `additive_angular_margin`, `subcenter_reduce`, `ldam_margins`,
  `subtract_class_margin`, `class_balanced_weights`, `inverse_freq_weights`,
  `logit_prior`, `masked_cross_entropy`, `valid_rows`, `has_positive_pairs`.
- `losses/metric.py` — `TripletSemiHardLoss`, `SupConLoss` (свободные, вход
  embeddings+labels; 0 без позитивов; маскируют `-1`).

## Маскирование -1
Любая голова/лосс исключает строки с `target == -1` (`valid_rows` /
`masked_cross_entropy`). Это едино для semi-supervised, multi-task и partially
labeled — не пиши свою ветку.

## Метрик-лоссы: без miner-абстракции (v1)
Структуру батча (гарантию позитивов) даёт **PK-сэмплер** (`data/samplers.py`) —
это и есть «майнер». Если понадобятся mined-варианты — бери
pytorch-metric-learning как зависимость, не переписывай.

## Новая голова
Наследуй `ClassifierHead`, реализуй `forward(emb, targets)`/`predict_logits`/
`classifier_weight`, композируй примитивы. Тест: сверь формулу с референсом,
проверь backward и маскирование `-1` (см. `tests/test_heads.py`).

## Baseline-first (§5.1)
CE — священный бейзлайн. Зоопарк доступен, но сравнивается с ним. Урок
прототипа: CE обыграл весь long-tail зоопарк.
