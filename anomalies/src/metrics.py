from sklearn.metrics import precision_recall_curve, f1_score, balanced_accuracy_score, roc_curve, fbeta_score, precision_score, recall_score, recall_score
import numpy as np
import torch

def compute_sample_weight(y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    sample_weight = np.ones_like(y_true)
    sample_weight[y_true == 1] = 1 / np.mean(y_true)
    sample_weight[y_true == 0] = 1 / (1 - np.mean(y_true))
    return sample_weight

def optimal_threshold(y_true, y_score, beta=1):
    sample_weight = compute_sample_weight(y_true)
    thresholds = np.unique(y_score)
    best_threshold = None
    best_score = -np.inf
    for threshold in thresholds:
        y_pred = y_score > threshold
        score = fbeta_score(y_true, y_pred, sample_weight=sample_weight, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold

def instance_metric(y_true, y_pred, beta=1):
    sample_weight = compute_sample_weight(y_true)
    return fbeta_score(y_true, y_pred, sample_weight=sample_weight, beta=beta)

def false_negative_rate(y_true, y_pred):
    sample_weight = compute_sample_weight(y_true)
    return 1 - recall_score(y_true, y_pred, sample_weight=sample_weight)

def precision_recall(y_true, y_score):
    return precision(y_true, y_score), recall(y_true, y_score)

def fpr_at_95_tpr(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=compute_sample_weight(y_true))
    return fpr[np.searchsorted(tpr, 0.95)]
