---
name: ssl-pretraining
description: Как добавить/сконфигурировать SSL-предобучение (BYOL, DINOv2) в vision-lab — контракт SSLMethod, MomentumTeacher, расписания, probe-отбор чекпоинтов, DDP.
---

# SSL-предобучение в vision-lab

## Контракт SSLMethod (`vision_lab/ssl/base.py`)
`SSLMethod(nn.Module)` владеет энкодерами, головами и GPU-вьюхами:
- `forward(batch) -> dict[str, Tensor]` с ключом `total_loss`; вьюхи генерируются
  ВНУТРИ forward на GPU (N независимых kornia-вьюх); gather по рангам — явным
  вызовом `all_gather_grad` (только для contrastive; BYOL/DINO НЕ гейтерят);
- `extract_embeddings(images) -> (B, D)` для probe — **докстринг обязан сказать,
  какой энкодер** (BYOL — online, DINO — teacher);
- `momentum_update()` — трейнер зовёт в `on_before_zero_grad` (ровно раз на
  optimizer step, корректно при grad accumulation).

## Как добавить новый SSL-метод
1. Новый файл `ssl/<method>.py`, класс наследует `SSLMethod`.
2. EMA-учитель — **всегда** `MomentumTeacher(student)` (submodule ⇒ resume
   бесплатно; `factory=` для weight_norm-голов, которые не deepcopy'ятся). Не
   пиши свой EMA-цикл.
3. Расписания (tau, temp) — объяви как float-атрибут; навесь `ScheduleDriver`
   (`{"method.current_tau": CosineSchedule(...)}`). Не храни как буфер.
4. Маскирование по сетке токенов: `TokenBackbone.grid_size(hw)` ДО forward
   (iBOT/MAE/I-JEPA).
5. Готово: probe, top-K чекпоинты, DDP достаются бесплатно (новый трейнер не нужен).

Семейства и их особенности — таблица в докстринге `ssl/base.py`
(teacher? gather? masking? probe-энкодер?).

## Отбор чекпоинтов (§5.6)
SSL-лосс НЕ коррелирует с качеством. Навесь `KNNProbeCallback` (kNN/linear-probe
macro-F1 на val) + `topk_per_metric_checkpoints({"val/knn_f1":"max",
"val/linprobe_f1":"max", "val/sel_f1":"max"})` — пики метрик на разных эпохах.

## Перенос в классификацию (§4.4)
`EmbeddingBackbone(model_name=..., ckpt_path="<ssl_ckpt>")` — `load_backbone`
срежет префикс стадии через `BACKBONE_PREFIXES` автоматически.

## Конфиги
`configs/experiment/ssl_byol.yaml`, `ssl_dinov2.yaml` — самодостаточные шаблоны.
DDP: `use_distributed_sampler=false`, `sync_batchnorm=true`, `bf16-mixed` (§11.1).
