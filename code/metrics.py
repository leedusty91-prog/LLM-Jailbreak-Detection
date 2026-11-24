"""Binary detection metrics used in the paper tables.

Accuracy alone is a poor summary when a jailbreak type is rare. Ethical Appeal
has on the order of ten positives in the 168-row hand-labelled file, so a
classifier that never predicts that type still scores above 90% accuracy.
Every reporter below returns Accuracy, F1, AUC, TPR, and FNR. F1, TPR, and FNR
use label 1 as the positive class. AUC is ``None`` when the evaluated slice
contains a single class.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(list(y_true)).astype(int)
    y_pred = np.asarray(list(y_pred)).astype(int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    positive = tp + fn
    tpr = tp / positive if positive else 0.0
    fnr = fn / positive if positive else 0.0
    accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if len(y_true) else 0.0
    auc = _auc(y_true, y_score)
    return {
        "Accuracy": accuracy,
        "F1": f1,
        "AUC": auc,
        "TPR": float(tpr),
        "FNR": float(fnr),
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def category_row(jailbreak_type: str, metrics: dict, **extra) -> dict:
    row = {
        "jailbreak_type": jailbreak_type,
        "accuracy": metrics["Accuracy"],
        "f1": metrics["F1"],
        "auc": metrics["AUC"],
        "tpr": metrics["TPR"],
        "fnr": metrics["FNR"],
        "n": metrics["n"],
        "n_positive": metrics["n_positive"],
    }
    row.update(extra)
    return row


def trainer_compute_metrics(pred) -> dict:
    """``transformers.Trainer`` metric function, including AUC.

    ``pred.predictions`` are logits. Class 1's logit is the ranking score.
    """
    labels = np.asarray(pred.label_ids)
    logits = np.asarray(pred.predictions)
    if logits.ndim == 1:
        scores = logits
        predictions = (logits >= 0).astype(int)
    else:
        predictions = logits.argmax(axis=-1)
        scores = logits[:, 1] if logits.shape[1] >= 2 else predictions
    metrics = classification_metrics(labels, predictions, scores)
    auc = metrics["AUC"]
    return {
        "accuracy": metrics["Accuracy"],
        "f1": metrics["F1"],
        "auc": float("nan") if auc is None else auc,
        "tpr": metrics["TPR"],
        "fnr": metrics["FNR"],
    }


def _auc(y_true: np.ndarray, y_score) -> float | None:
    if y_score is None or len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    scores = np.asarray(list(y_score), dtype=float)
    if len(scores) != len(y_true):
        raise ValueError("y_score must have the same length as y_true")
    if np.isnan(scores).any():
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def is_missing_number(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False
