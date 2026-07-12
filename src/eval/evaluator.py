"""Загрузка чекпоинта (модель + голова), инференс и сборка таблиц метрик.

Единый стандарт после рефакторинга:
    model(images) -> {"embeddings"};  head.predict_logits(embeddings) -> логиты.
Голова (criterion) владеет весами классификатора, поэтому для инференса нужны ОБА:
веса model.* и criterion.* из Lightning-чекпоинта.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

import hydra

from src.eval.metrics import build_metrics_table


def load_model_and_head(ckpt_path, cfg, device="cuda"):
    """Строит ClassificationModel + голову из cfg и грузит веса из Lightning-чекпоинта."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    head = hydra.utils.instantiate(cfg.loss)

    model_sd = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    head_sd = {k[len("criterion."):]: v for k, v in state.items() if k.startswith("criterion.")}

    m_miss, m_unexp = model.load_state_dict(model_sd, strict=False)
    h_miss, h_unexp = head.load_state_dict(head_sd, strict=False)
    print(f"[load] model: missing={len(m_miss)} unexpected={len(m_unexp)} | "
          f"head: missing={len(h_miss)} unexpected={len(h_unexp)}")

    return model.to(device).eval(), head.to(device).eval()


@torch.no_grad()
def run_inference(model, head, dataset, device="cuda", batch_size=16, num_workers=8):
    """Прогон датасета → (y_true [N], logits [N, C], probs [N, C])."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_logits, all_labels = [], []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)
    for batch in tqdm.tqdm(loader, leave=False):
        images = batch["image"].to(device)
        with autocast:
            embeddings = model(images)["embeddings"]
            logits = head.predict_logits(embeddings)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return labels, logits, probs


def evaluate_datasets(model, head, datasets: dict, class_names: dict, device="cuda",
                      batch_size=16, num_workers=8, save_logits_dir=None, exp_name="exp",
                      sens_target=0.8, pauc_min_tpr=0.8):
    """Считает таблицу метрик по каждому датасету. Опц. сохраняет логиты (.npz) для ROC.

    Возвращает {dataset_name: DataFrame(per-class + macro)}.
    """
    tables = {}
    for name, dataset in datasets.items():
        print(f"[eval] {exp_name} :: {name}  (N={len(dataset)})")
        y_true, logits, probs = run_inference(
            model, head, dataset, device, batch_size, num_workers)

        if save_logits_dir is not None:
            out = Path(save_logits_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out / f"{exp_name}__{name}.npz",
                logits=logits, probs=probs, labels=y_true,
                class_names=np.array([class_names.get(i, str(i))
                                      for i in range(probs.shape[1])]),
            )

        tables[name] = build_metrics_table(
            y_true, probs, class_names, sens_target, pauc_min_tpr)
    return tables


def save_tables(tables: dict, output_dir, exp_name: str):
    """Сохраняет per-dataset таблицы (csv/md/xlsx) и сводку по датасетам."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        stem = out / f"{exp_name}__{name}"
        df.to_csv(f"{stem}.csv")
        with open(f"{stem}.md", "w") as f:
            f.write(f"### {exp_name} — {name}\n\n{df.to_markdown()}\n")

    # сводка: строка macro каждого датасета
    summary = pd.DataFrame({name: df.loc["macro"] for name, df in tables.items()}).T
    summary.to_csv(out / f"{exp_name}__summary.csv")
    with open(out / f"{exp_name}__summary.md", "w") as f:
        f.write(f"### {exp_name} — macro по датасетам\n\n{summary.round(4).to_markdown()}\n")
    try:
        with pd.ExcelWriter(out / f"{exp_name}.xlsx") as xls:
            summary.round(4).to_excel(xls, sheet_name="summary")
            for name, df in tables.items():
                df.to_excel(xls, sheet_name=name[:31])
    except Exception as e:  # openpyxl может отсутствовать
        print(f"[warn] xlsx не сохранён: {e}")
    return summary
