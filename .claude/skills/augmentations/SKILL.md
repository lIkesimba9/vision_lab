---
name: augmentations
description: Двухуровневый конвейер аугментаций vision-lab — CPU-рецепты классификации (albumentations) и GPU-вьюхи SSL (kornia) с инвариантами порядка.
---

# Аугментации в vision-lab (§6)

Урок прототипа: **рецепт аугментаций влияет на качество сильнее выбора лосса и
бэкбона** (~0.10 F1). Это полноценный объект проектирования.

```
диск ─► CPU (DataLoader worker) ─► GPU (внутри training_step/SSL-forward)
        OpenCV + albumentations     kornia.AugmentationSequential
```

## CPU: классификация (`vision_lab/data/transforms/classification.py`)
`build_classification_transform(recipe, image_size, train)`:
- рецепты: `eval`, `light`, `medium`, `heavy_v1` (порт «hugeron»);
- одна вьюха на шаг, ImageNet-нормализация;
- **канонический стек — albumentations** (порт между стеками не поддерживается, §6.2).

## GPU: SSL-вьюхи (`vision_lab/ssl/gpu_augs.py`)
`build_ssl_views(recipe)`: `byol_v1` (2 асимметричных глобальных),
`dino_v1` (2 глобальных + n локальных multi-crop). Вся стохастика + Normalize —
на GPU: нужны N *независимых* вьюх одного тензора (CPU-augs дали бы одинаковую).

## Инварианты порядка — в КОДЕ, не в YAML (§6.3)
`build_view_pipeline` вызывает `_assert_order`:
- `Normalize` — строго последним;
- value-range ops (шум, соляризация, blur) — ДО Normalize (иначе device-assert).
Правкой YAML контракт не нарушить. Builder делает коэрцию Hydra ListConfig.

## Таблица рецептов
| рецепт | стек | назначение |
|---|---|---|
| eval | albumentations | resize + normalize (val/inference) |
| light/medium/heavy_v1 | albumentations | классификация, одна вьюха |
| byol_v1 | kornia | 2 асимметричных глобальных вьюхи |
| dino_v1 | kornia | 2 глобальных + n локальных (multi-crop) |

Именуй и версионируй рецепты (`heavy_v1`); имя фиксируется в логе эксперимента.
Производительность: `configure_threads()` в точке входа (cv2/BLAS vs воркеры, §6.4).
