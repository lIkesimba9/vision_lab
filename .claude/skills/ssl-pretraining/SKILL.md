---
name: ssl-pretraining
description: Как добавить/сконфигурировать SSL-предобучение (BYOL, DINOv2, SimCLR, MoCo v3, SimSiam, MAE, SimMIM) в vision-lab — контракт SSLMethod, MomentumTeacher, расписания, probe-отбор чекпоинтов, DDP.
---

# SSL-предобучение в vision-lab

## Реализованные методы (`vision_lab/ssl/`)
- self-distillation: `BYOL` (+`BYOLTriplet`, positive-shuffle, multi-crop),
  `DINOv2` (DINO+iBOT+KoLeo+Sinkhorn), `SimSiam` (stop-grad, без EMA);
- contrastive: `SimCLR` (NT-Xent, gather негативов по рангам),
  `MoCoV3` (EMA-ключи, негативы батча);
- masked image modeling: `MAE` (только timm ViT: энкодер по видимым патчам),
  `SimMIM` (любой TokenBackbone: маска входа + L1 на маске).

## Контракт SSLMethod (`vision_lab/ssl/base.py`)
`SSLMethod(nn.Module)` владеет энкодерами, головами и GPU-вьюхами:
- `forward(batch) -> dict[str, Tensor]` с ключом `total_loss`; вьюхи генерируются
  ВНУТРИ forward на GPU (N независимых kornia-вьюх); gather по рангам — явным
  вызовом `all_gather_grad` (только для contrastive: SimCLR/MoCo; BYOL/DINO НЕ гейтерят);
- `extract_embeddings(images) -> (B, D)` для probe — **докстринг обязан сказать,
  какой энкодер** (BYOL/SimCLR/MoCo/SimSiam — online, DINO — teacher, MAE/SimMIM —
  энкодер без маски);
- `momentum_update()` — трейнер зовёт в `on_before_zero_grad` (ровно раз на
  optimizer step, корректно при grad accumulation); методы без учителя
  (SimCLR/SimSiam/MAE/SimMIM) — no-op из базы.

## Как добавить новый SSL-метод
1. Новый файл `ssl/<method>.py`, класс наследует `SSLMethod`.
2. EMA-учитель — **всегда** `MomentumTeacher(student)` (submodule ⇒ resume
   бесплатно; `factory=` для weight_norm-голов, которые не deepcopy'ятся). Не
   пиши свой EMA-цикл.
3. Расписания (tau, temp) — объяви как float-атрибут; навесь `ScheduleDriver`
   (`{"method.current_tau": CosineSchedule(...)}`). Не храни как буфер.
4. Маскирование по сетке токенов: `TokenBackbone.grid_size(hw)` ДО forward;
   готовые примитивы — `components.block_mask` (вход, iBOT/SimMIM),
   `mae.random_masking` (токены, MAE), `components.patchify` (пиксельный таргет).
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
Самодостаточные шаблоны в `configs/experiment/`: `ssl_byol`, `ssl_dinov2`,
`ssl_simclr`, `ssl_moco_v3`, `ssl_simsiam`, `ssl_mae`, `ssl_simmim`.
Рецепты вьюх (`gpu_augs.build_ssl_views`): `byol_v1` (асимметричные),
`dino_v1` (+multi-crop), `simclr_v1` (симметричные), `mim_v1` (только
crop+flip — для MAE/SimMIM маскирование заменяет аугментации).
DDP: `use_distributed_sampler=false`, `sync_batchnorm=true`, `bf16-mixed` (§11.1).
