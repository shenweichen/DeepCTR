"""Evaluation metrics for the benchmark suite.

Thin wrappers over scikit-learn so every track reports metrics the same way:
binary classification -> AUC + LogLoss, regression -> MSE + RMSE.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score


def binary_metrics(y_true, y_pred):
    """AUC and LogLoss for a single binary task.

    AUC is undefined when ``y_true`` has a single class (can happen on tiny
    test splits); we return NaN in that case instead of raising.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    out = {}
    try:
        out["AUC"] = round(float(roc_auc_score(y_true, y_pred)), 4)
    except ValueError:
        out["AUC"] = float("nan")
    # labels=[0, 1] keeps log_loss well-defined even if a split is single-class.
    out["LogLoss"] = round(float(log_loss(y_true, y_pred, labels=[0, 1])), 4)
    return out


def regression_metrics(y_true, y_pred):
    """MSE and RMSE for a single regression task."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    return {"MSE": round(mse, 4), "RMSE": round(float(np.sqrt(mse)), 4)}


def compute_metrics(task, y_true, y_pred):
    """Dispatch to the right metric set for ``task`` ('binary' or 'regression')."""
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    return binary_metrics(y_true, y_pred)


# Direction of the headline metric, used by the leaderboard sorter.
HIGHER_IS_BETTER = {
    "AUC": True,
    "LogLoss": False,
    "MSE": False,
    "RMSE": False,
    "mean_AUC": True,
}
