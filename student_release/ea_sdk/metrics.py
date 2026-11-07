import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report)

LABELS = [-1, 0, 1]


def evaluate(y_true, y_pred):
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or len(truth) != len(pred):
        raise ValueError("Expected equal-length one-dimensional arrays")
    mask = ~pd.isna(truth)
    truth, pred = truth[mask], pred[mask]
    if not len(truth):
        raise ValueError("No labeled rows available for scoring")
    if not np.isin(truth, LABELS).all() or not np.isin(pred, LABELS).all():
        raise ValueError("Labels must be -1, 0, or 1")
    report = classification_report(truth, pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    return {
        "labels": LABELS,
        "n_scored": int(len(truth)),
        "macro_f1": float(f1_score(truth, pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "mcc": float(matthews_corrcoef(truth, pred)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=LABELS).tolist(),
        "per_class": {str(k): report[str(k)] for k in LABELS},
    }
