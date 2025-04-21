import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from typing import Dict

def metrics(
    y_true,
    y_pred,
    n_classes: int = 0,
) -> Dict[str, float]:
    accuracy_at_1 = top_k_accuracy_score(y_true, y_pred, k=1, labels=np.arange(0, n_classes))
    accuracy_at_2 = top_k_accuracy_score(y_true, y_pred, k=2, labels=np.arange(0, n_classes))
    accuracy_at_3 = top_k_accuracy_score(y_true, y_pred, k=3, labels=np.arange(0, n_classes))
    accuracy_at_4 = top_k_accuracy_score(y_true, y_pred, k=4, labels=np.arange(0, n_classes))
    accuracy_at_5 = top_k_accuracy_score(y_true, y_pred, k=5, labels=np.arange(0, n_classes))

    return {
        "accuracy@1": accuracy_at_1,
        "accuracy@2": accuracy_at_2,
        "accuracy@3": accuracy_at_3,
        "accuracy@4": accuracy_at_4,
        "accuracy@5": accuracy_at_5,
    }