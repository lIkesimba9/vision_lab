---
name: data
description: Слой данных vision-lab — сэмпл-центричный манифест, декодеры, таксономия, PK-сэмплеры с DDP, доменная предобработка.
---

# Данные в vision-lab (§7)

**Пиксели и разметка раздельны, связь — через манифест.** Никаких `ImageFolder`.

## Манифест (`vision_lab/data/manifest.py`)
`ManifestDataset(manifest, root, split, where, label_column, classes, ...)`:
- parquet, строка = сэмпл; N входов (`input_<modality>_path`) и N таргетов;
- обязательные колонки: `sample_id`, `source`, `split`, `input_<modality>_path`;
- отсутствие метки → `-1` (`MISSING_LABEL`); `unknown="error"` (тихие фолбэки
  запрещены §7.4);
- `where` — pandas-query (`"label.notna()"` для semi-supervised);
- `taxonomy=` добавляет `levels` (предки метки выводятся кодом, не в манифесте).

Батч (плоский dict): `image`, `label`, `levels`, `sample_id`, `source`; резерв
неймспейсов `image_<mod>`, `label_<task>`, `target_<name>` (§7.2).

## Декодеры (`decoders.py`)
Реестр по формату (`source_format_<modality>` / расширение): png/jpeg/tiff/npy
(8- и 16-бит). `register_decoder(fmt, fn)` — точка расширения. DICOM — только
офлайн (`preprocessing/dicom.py`), не на лету.

## Таксономия (`taxonomy.py`)
`Taxonomy.from_yaml(...)` — иерархия отдельным версионируемым YAML; в манифесте
самая специфичная метка, предки — кодом. `levels_vector(label)`; неизвестная
метка = ошибка.

## PK-сэмплеры (`samplers.py`, §5.4)
`PKCoverageBatchSampler` (полное покрытие + позитивы), `PositivePairsBatchSampler`,
`PKBatchSampler` (alpha-температура). Все DDP-safe: единый seed на эпоху, срез
`batches[rank::world]`, равное число батчей. Трейнер: `use_distributed_sampler=false`.

## Предобработка (`preprocessing/`, §7.4)
`ColorConstancy` (shades-of-gray), `SourceAlignment` (пер-source в референс;
source ТОЛЬКО из колонки, отсутствие статистик = ошибка). Одинаково train/inference.
`compute_source_stats(dataset)` — расчёт статистик (одна реализация, не копируй).
