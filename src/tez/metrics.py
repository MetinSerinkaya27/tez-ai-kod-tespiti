"""Tum modeller icin ortak degerlendirme metrikleri.

Pozitif sinif AI (label=1). Esik dogrulama kumesinde secilir, test kumesine
oldugu gibi uygulanir; testte esik ayarlanmaz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Olasiliklardan siniflandirma metriklerini hesaplar."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_class = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if two_class else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if two_class else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, y_prob) -> float:
    """F1'i en yukseltan esigi dondurur (yalnizca dogrulama kumesinde kullan)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.concatenate([y_prob, [0.5]]))
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def save_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
