"""Metrics that matter for fraud: PR-AUC first, accuracy only as a sanity check."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate(y_true, proba, amount=None, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if amount is not None:
        amount = np.asarray(amount, dtype=float)
        fraud_total = amount[y_true == 1].sum()
        out["fraud_value_caught_pct"] = float(100 * amount[(y_true == 1) & (pred == 1)].sum() / fraud_total) if fraud_total else 0.0
        out["false_alarm_value"] = float(amount[(y_true == 0) & (pred == 1)].sum())
    return out
