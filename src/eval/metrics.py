"""Метрики классификации в стиле таблицы ISIC milk10k.

По каждому классу (one-vs-rest) и общая (macro) строка:
    AUC, pAUC80 (partial AUC выше 80% TPR), Spec@Sens80 (специфичность при чувств.≥0.80),
    AP (Average Precision), Accuracy, Sensitivity, Specificity, F1, PPV, NPV, Support.

Threshold-free метрики (AUC, pAUC80, Spec@Sens80, AP) считаются по вероятностям one-vs-rest.
Точечные метрики (Sensitivity/Specificity/F1/PPV/NPV/Accuracy) — по argmax-предсказаниям
из мультиклассовой confusion-матрицы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)

# порядок колонок итоговой таблицы
METRIC_COLUMNS = [
    "AUC", "pAUC80", "Spec@Sens80", "AP", "Accuracy",
    "Sensitivity", "Specificity", "F1", "PPV", "NPV", "Support",
]


def partial_auc_above_tpr(y_true_bin: np.ndarray, y_score: np.ndarray,
                          min_tpr: float = 0.8) -> float:
    """Partial AUC в области TPR ∈ [min_tpr, 1], нормированный в [0, 1] (McClish).

    Считается как площадь под ROC в этой полосе, делённая на площадь полосы
    (1 - min_tpr). Эквивалент метрики ISIC 2024 pAUC при min_tpr=0.8.
    """
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return float("nan")
    width = 1.0 - min_tpr
    if width <= 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    # вставляем точку ровно на TPR = min_tpr (линейная интерполяция fpr по tpr)
    if min_tpr not in tpr:
        fpr_at = np.interp(min_tpr, tpr, fpr)
        idx = np.searchsorted(tpr, min_tpr)
        tpr = np.insert(tpr, idx, min_tpr)
        fpr = np.insert(fpr, idx, fpr_at)
    mask = tpr >= min_tpr
    # интегрируем специфичность (1 - FPR) по TPR в полосе и нормируем на её ширину →
    # средняя специфичность при чувствительности ≥ min_tpr, значение в [0, 1]
    spec = 1.0 - fpr
    area = np.trapezoid(spec[mask], tpr[mask])
    return float(area / width)


def specificity_at_sensitivity(y_true_bin: np.ndarray, y_score: np.ndarray,
                               target_sens: float = 0.8) -> float:
    """Специфичность (1 - FPR) в рабочей точке, где чувствительность (TPR) ≥ target_sens."""
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    idx = np.where(tpr >= target_sens)[0]
    if len(idx) == 0:
        return 0.0
    return float(1.0 - fpr[idx[0]])


def per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int,
                      sens_target: float = 0.8, pauc_min_tpr: float = 0.8) -> list[dict]:
    """Список метрик по каждому классу (one-vs-rest)."""
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    rows = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        y_bin = (y_true == c).astype(int)
        score = y_prob[:, c]
        both_present = 0 < y_bin.sum() < len(y_bin)

        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0

        rows.append({
            "AUC": roc_auc_score(y_bin, score) if both_present else float("nan"),
            "pAUC80": partial_auc_above_tpr(y_bin, score, pauc_min_tpr),
            "Spec@Sens80": specificity_at_sensitivity(y_bin, score, sens_target),
            "AP": average_precision_score(y_bin, score) if both_present else float("nan"),
            "Accuracy": (tp + tn) / cm.sum() if cm.sum() else 0.0,
            "Sensitivity": sens,
            "Specificity": spec,
            "F1": f1,
            "PPV": ppv,
            "NPV": npv,
            "Support": int(y_bin.sum()),
        })
    return rows


def build_metrics_table(y_true: np.ndarray, y_prob: np.ndarray, class_names: dict,
                        sens_target: float = 0.8, pauc_min_tpr: float = 0.8) -> pd.DataFrame:
    """Таблица: строка на класс + строка 'macro'/'overall'. Колонки — METRIC_COLUMNS."""
    num_classes = y_prob.shape[1]
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rows = per_class_metrics(y_true, y_prob, num_classes, sens_target, pauc_min_tpr)
    index = [class_names.get(c, str(c)) for c in range(num_classes)]

    df = pd.DataFrame(rows, index=index)[METRIC_COLUMNS]

    # строка macro: среднее по классам (Support — сумма; Accuracy — глобальная)
    macro = df.drop(columns=["Support", "Accuracy"]).mean(axis=0, skipna=True)
    macro["Accuracy"] = (y_prob.argmax(axis=1) == y_true).mean()
    macro["Support"] = int(df["Support"].sum())
    df.loc["macro"] = macro[METRIC_COLUMNS]

    return df.round(4)
