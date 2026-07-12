# vision-lab — карта пакета для агентов

Библиотека обучения CV-моделей: классификация + SSL-предобучение (Lightning +
Hydra + timm + kornia). **Только код фреймворка**; данные/эксперименты живут в
отдельных тонких task-репах, ставящих `vision-lab==x.y.z` с пином.

## Соглашения
- **Язык**: код и идентификаторы — английский; доки/комментарии — **русский** (весь репо).
- Пакетный менеджер — **uv**. Тесты: `uv run pytest -q`. Линт: `uv run ruff check`.
- Компоненты принимают **явные kwargs**, никогда config-объект (тестируемость из REPL).
- Трейнеры получают уже инстанцированные компоненты; оптимизатор — `_partial_`.
- Отсутствие метки в батче — всегда `-1` (`MISSING_LABEL`), никогда None/пропуск ключа.
- Запрещён бесформенный `squeeze()` — только `squeeze(dim)`.
- `torch.load(weights_only=True)` по умолчанию.
- Инварианты (порядок augs, диапазоны) — assert'ами в builder'ах, не в комментариях YAML.

## Карта подпакетов
- `core/` — доменно-нейтральный фундамент:
  - `batch.py` — контракт батча (плоский dict, `target_view`);
  - `module.py` — **два трейнера**: `ClassificationTrainer`, `SSLTrainer`;
  - `callbacks.py` — `KNNProbeCallback`, `topk_per_metric_checkpoints`, `FreezeParams`;
  - `schedules.py` — `CosineSchedule`/… + `ScheduleDriver` (пишет float-атрибуты);
  - `checkpoint.py` — реестр `BACKBONE_PREFIXES`, `load_backbone`, перенос FC;
  - `optim.py` — `param_groups` (именованные, no-decay); `dist.py` — `all_gather_grad`, DDP-mixin.
- `data/` — `manifest.py` (parquet, §7.2), `decoders.py`, `taxonomy.py`,
  `samplers.py` (PK-семейство), `preprocessing/` (color constancy, source-align), `transforms/` (albumentations).
- `models/backbones.py` — `Embedding`/`Spatial`/`TokenBackbone` над timm (атрибут `net`).
- `heads/` — `ClassifierHead` (контракт `forward(emb, targets: Mapping)`), `primitives.py`
  (чистые функции), `classification.py` (зоопарк), `multitask.py`.
- `losses/metric.py` — `TripletSemiHardLoss`, `SupConLoss`.
- `ssl/` — `base.py` (`SSLMethod`, `MomentumTeacher`), `byol.py`, `dinov2.py`,
  `gpu_augs.py` (kornia-builder с инвариантами §6.3), `components.py`.
- `inference/` — `Predictor` (load → transform → predict), `tta.py` (flip-TTA).
- `eval/knn_probe.py` — kNN/linear-probe macro-F1.
- `experimental/` — **НЕ боевое** (Proto, Speaker); не импортируется боевыми.
- `configs/` — эталонные Hydra-шаблоны (`experiment/`, `model/`, `head/`, …).

## Ключевые контракты
- **Backbone → голый тензор**: `EmbeddingBackbone(x) -> (B,D)`; `TokenBackbone(x) -> (pooled, tokens, grid)`.
- **Head**: `forward(embeddings, targets) -> {"total_loss", ...}`; `predict_logits`; `classifier_weight [C,D]`.
- **SSLMethod**: `forward(batch) -> dict лоссов`; `extract_embeddings`; `momentum_update()` (зовёт трейнер в `on_before_zero_grad`).
- **Чекпоинт любой стадии → чистый бэкбон**: `load_backbone(net, ckpt)` через `BACKBONE_PREFIXES`.

## DDP (§11.1)
`use_distributed_sampler=False`, `sync_batchnorm=True`, `bf16-mixed`; кастомные
сэмплеры шардят сами (единый seed на эпоху, `batches[rank::world]`). EMA — ровно
раз на optimizer step. Смоук: `tests/ddp_smoke.py` (реальный 2-GPU).
