"""Тонкая CLI-точка входа (§8): Hydra-раннер поверх тех же компонентов.

Два равноправных режима использования библиотеки:

* **Python-модуль** — компоненты создаются явными kwargs (REPL/ноутбук/свой
  скрипт), трейнеры принимают готовые объекты; Hydra не нужна вовсе;
* **CLI** — ``vision-lab-train --config-name experiment/classification_ce``
  (или ``python -m vision_lab.train ...``): тот же состав компонентов собирает
  Hydra из YAML. Task-репа добавляет свои конфиги через
  ``--config-dir ./configs``.

Раннер намеренно тонкий (анти-цель §1: не CLI-монолит): никакой логики задач и
if-веток по именам конфигов — только instantiate + fit. Состав конфига:

    module:      LightningModule (ClassificationTrainer | SSLTrainer)
    data:        train_dataloader / val_dataloader — обычно
                 torch DataLoader поверх ManifestDataset (задаётся в task-репе;
                 эталонные experiment-шаблоны библиотеки данных НЕ содержат)
    trainer:     lightning.pytorch.Trainer (шаблон configs/trainer/default.yaml)
    schedules:   {атрибут: Schedule} -> ScheduleDriver          (опционально)
    callbacks:   список _target_-коллбеков                      (опционально)
    seed:        int                                            (опционально)
    resume_from: путь к Lightning-чекпоинту                     (опционально)
"""

from __future__ import annotations

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from vision_lab.configs import CONFIG_ROOT, register_resolvers
from vision_lab.core import ScheduleDriver
from vision_lab.core.dist import DistributedBatchSamplerMixin
from vision_lab.core.runtime import configure_threads

register_resolvers()


def check_distributed_sampler(trainer: pl.Trainer, train_loader) -> None:
    """Ловит молчаливое дублирование датасета между рангами на DDP.

    ``trainer/default.yaml`` задаёт ``use_distributed_sampler=false`` — это
    верно для PK-семейства сэмплеров, которые шардят сами через
    :class:`~vision_lab.core.dist.DistributedBatchSamplerMixin`. Но с обычным
    ``DataLoader(shuffle=True)`` тот же флаг означает, что КАЖДЫЙ ранг пройдёт
    весь датасет целиком: эпоха тихо становится в ``world_size`` раз длиннее,
    градиенты усредняются по дублям, а метрики выглядят правдоподобно. Ошибка не
    падает и в логах никак не видна — поэтому проверяем явно.
    """
    if trainer.num_devices <= 1 and trainer.num_nodes <= 1:
        return
    if trainer._accelerator_connector.use_distributed_sampler:
        return
    sampler = getattr(train_loader, "batch_sampler", None)
    if isinstance(sampler, DistributedBatchSamplerMixin):
        return
    raise ValueError(
        "use_distributed_sampler=false при обычном DataLoader и "
        f"{trainer.num_devices} устройствах: каждый ранг пройдёт весь датасет, "
        "эпоха станет в world_size раз длиннее, а градиенты будут усреднены по "
        "дублям. Либо поставьте trainer.use_distributed_sampler=true, либо "
        "передайте batch_sampler из vision_lab.data.samplers (PK-семейство "
        "шардит само). Дефолт false в trainer/default.yaml рассчитан на второй "
        "случай."
    )


def run(cfg: DictConfig) -> pl.Trainer:
    """Собирает компоненты из конфига и запускает fit; возвращает Trainer.

    Отделён от :func:`main`, чтобы Python-режим мог вызвать то же самое с
    конфигом из compose API или собранным вручную (OmegaConf.create).
    """
    configure_threads()
    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    if "data" not in cfg or "train_dataloader" not in cfg.data:
        raise ValueError(
            "Конфиг обязан задать data.train_dataloader (DataLoader поверх "
            "ManifestDataset). Эталонные experiment-шаблоны библиотеки содержат "
            "только module/transform — секцию data добавляет task-репа."
        )

    module = instantiate(cfg.module, _convert_="all")
    train_loader = instantiate(cfg.data.train_dataloader, _convert_="all")
    val_loader = (instantiate(cfg.data.val_dataloader, _convert_="all")
                  if "val_dataloader" in cfg.data else None)

    callbacks = [instantiate(cb, _convert_="all") for cb in cfg.get("callbacks") or []]
    if "schedules" in cfg:
        callbacks.append(ScheduleDriver(
            {key: instantiate(s) for key, s in cfg.schedules.items()}))

    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is not None:
        trainer = instantiate(trainer_cfg, callbacks=callbacks, _convert_="all")
    else:
        trainer = pl.Trainer(callbacks=callbacks)

    check_distributed_sampler(trainer, train_loader)
    trainer.fit(module, train_loader, val_loader, ckpt_path=cfg.get("resume_from"))
    return trainer


@hydra.main(config_path=str(CONFIG_ROOT), config_name=None, version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
