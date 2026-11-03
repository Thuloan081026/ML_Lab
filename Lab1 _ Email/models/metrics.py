"""
Các hàm đo lường hiệu năng mô hình phân loại (From Scratch).
"""

import numpy as np

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính Confusion Matrix, Accuracy, Precision, Recall và F1-Score."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray):
    """Tính FPR, TPR và ROC-AUC qua tích phân hình thang."""
    order = np.argsort(y_scores)[::-1]
    y_sorted = y_true[order]
    p = np.sum(y_true == 1)
    n = np.sum(y_true == 0)

    fps, tps = 0, 0
    fpr, tpr = [0.0], [0.0]
    for label in y_sorted:
        if label == 1:
            tps += 1
        else:
            fps += 1
        fpr.append(fps / n if n > 0 else 0.0)
        tpr.append(tps / p if p > 0 else 0.0)
    fpr.append(1.0)
    tpr.append(1.0)

    fpr_arr = np.array(fpr)
    tpr_arr = np.array(tpr)
    trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc_val = float(trap_fn(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc_val
