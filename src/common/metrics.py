from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def threshold_from_percentile(scores: np.ndarray, percentile: float) -> float:
    """Compute threshold as the given percentile of scores."""
    return float(np.percentile(scores, percentile))


def evaluate_binary_classification(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    preds = (scores > threshold).astype(int)

    try:
        roc = roc_auc_score(y_true, scores)
    except Exception:
        roc = 0.0
    try:
        ap = average_precision_score(y_true, scores)
    except Exception:
        ap = 0.0

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sensitivity + specificity) / 2.0

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denom) if denom > 0 else 0.0
    dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'roc_auc': roc,
        'average_precision': ap,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'balanced_accuracy': bal_acc,
        'mcc': mcc,
        'dsc': dsc,
        'fpr': fpr,
        'fnr': fnr,
        'predictions': preds,
    }


