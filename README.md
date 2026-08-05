# vision-lab

Переиспользуемая библиотека для обучения моделей компьютерного зрения:
**классификация изображений + self-supervised предобучение**. Построена на
PyTorch Lightning + Hydra + timm + kornia.

Библиотека содержит **только код фреймворка** (трейнеры, датасеты, сэмплеры,
аугментации, головы/лоссы, SSL-методы, инференс, эталонные конфиги). Конкретные
задачи (данные, эксперименты, результаты) живут в отдельных тонких
task-репозиториях, которые ставят `vision-lab` с пином версии.

## Установка (uv)

```bash
uv sync                      # разработка (dev-группа: pytest, ruff)
uv sync --extra dicom        # + офлайн-конвертация DICOM
uv run pytest -q             # тесты
```

## Что внутри

| Слой | Модуль | Ключевое |
|---|---|---|
| Каркас | `core/` | 2 трейнера (`ClassificationTrainer`, `SSLTrainer`), расписания, чекпоинт-контракт, DDP |
| Данные | `data/` | сэмпл-центричный parquet-манифест, декодеры, таксономия, PK-сэмплеры, color constancy |
| Бэкбоны | `models/backbones.py` | `Embedding`/`Spatial`/`TokenBackbone` над любой timm-моделью |
| Головы | `heads/` | единый `ClassifierHead` + зоопарк (CE/AAM/CosFace/LDAM/Focal/…), multi-label, multi-task, иерархия |
| SSL | `ssl/` | BYOL, DINOv2 (DINO+iBOT+KoLeo+Sinkhorn), SimCLR, MoCo v3, SimSiam, MAE, SimMIM |
| Оценка | `eval/`, `inference/` | kNN/linear-probe, единый инференс + flip-TTA |

## Два режима использования

**Python-модуль** — компоненты создаются явными kwargs, Hydra не обязательна:

```python
from functools import partial
import torch, lightning.pytorch as pl
from vision_lab.models.backbones import EmbeddingBackbone
from vision_lab.heads import LinearHead
from vision_lab.core import ClassificationTrainer

backbone = EmbeddingBackbone("convnextv2_tiny", pretrained=True)
head = LinearHead(n_class=3, embedding_dim=backbone.out_dim, mode="ce")
module = ClassificationTrainer(backbone, head,
                               optimizer=partial(torch.optim.AdamW, lr=1e-3),
                               num_classes=3)
# trainer.fit(module, train_loader, val_loader)
```

**CLI (Hydra)** — тот же состав собирается из YAML тонким раннером
`vision_lab/train.py` (задачной логики в нём нет — только instantiate + fit):

```bash
# из task-репы: свои конфиги + эталонные шаблоны библиотеки
vision-lab-train --config-dir ./configs --config-name my_experiment
# то же без установки console-script
python -m vision_lab.train --config-dir ./configs --config-name my_experiment \
    module.optimizer.lr=3e-4          # hydra-оверрайды работают как обычно
```

Конфиг задаёт `module` / `data` (DataLoader поверх `ManifestDataset`) /
`trainer` / опционально `schedules`, `callbacks`, `seed`, `resume_from` — см.
докстринг `vision_lab/train.py`. Эталонные шаблоны `module`-части — в
`vision_lab/configs/experiment/`; секцию `data` добавляет task-репа.

SSL-претрейн → перенос бэкбона в классификацию — см. `.claude/skills/` и
эталонные конфиги в `vision_lab/configs/experiment/`.

## Принципы

- **Baseline-first**: эталон каждого пайплайна — CE + сильные аугментации;
  зоопарк лоссов сравнивается с ним, а не запускается вместо.
- **Единые контракты** (backbone→тензор / head / SSLMethod): один трейнер на
  любую голову, единый инференс, перенос весов между стадиями.
- **DDP из коробки**: single-GPU и multi-GPU без изменения кода задачи.

Полная карта — в [CLAUDE.md](CLAUDE.md). ТЗ — в `vision-lab-spec.md`.
