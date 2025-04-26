import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score
from typing import Dict

def metrics(
    y_true,
    y_pred,
    n_classes: int = 0,
    k=1
) -> Dict[str, float]:
    acc = []
    for i in range(1,k+1):
        accuracy = top_k_accuracy_score(y_true, y_pred, k=i, labels=np.arange(0, n_classes))
        acc.append([i, accuracy])
    acc = dict(acc)

    return acc