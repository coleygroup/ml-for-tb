import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def max_f1_score(y_true, y_score):
    return max(
        [f1_score(y_true, (np.array(y_score) > t).astype(int)) for t in np.arange(0.0, 1.01, 0.02)]
    )


def binary_classification_metrics(y_true, y_pred, y_score, return_type="dataframe"):
    """
    Metrics for the binary classification setting.

    Parameters
    ----------
    y_true: List[int] or List[float]
        Ground truth labels (0. or 1. for each sample)
    y_pred: List[int] or List[float]
        Binary predictions for each sample (0. or 1.)
    y_score: List[float]
        Scores between [0., 1.] for each sample given by model.
    return_type: str
        Type of return value. Either 'dataframe' or 'dict'.

    Returns
    -------
    metrics: pd.DataFrame
        Dataframe with each metric as a separate row.
    """

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Calculate frequencies of each label in predictions
    _, counts = np.unique(y_pred, return_counts=True)

    # Check if any positive samples present
    n_pos = 0 if len(counts) == 1 else counts[1]

    metrics = []
    metrics.append(["n_pos", n_pos])
    metrics.append(["accuracy", round(accuracy_score(y_true, y_pred), 5) * 100])
    metrics.append(["balanced_accuracy", round(balanced_accuracy_score(y_true, y_pred), 5) * 100])
    metrics.append(["auroc", round(roc_auc_score(y_true, y_score), 5) * 100])
    metrics.append(["ap", round(average_precision_score(y_true, y_score), 5) * 100])
    metrics.append(["precision", round(precision_score(y_true, y_pred), 5) * 100])
    metrics.append(["recall", round(recall_score(y_true, y_pred), 5) * 100])
    metrics.append(["f1_score", round(f1_score(y_true, y_pred), 5) * 100])

    if return_type == "dataframe":
        return pd.DataFrame(metrics, columns=["Metric", "Value"])
    elif return_type == "dict":
        return dict(metrics)
    else:
        raise ValueError("return_type must be either 'dataframe' or 'dict'.")


def regression_metrics(y_true, y_pred, return_type="dataframe"):
    """
    Metrics for the regression setting.

    Parameters
    ----------
    y_true: List[float]
        Ground truth label column values
    y_pred: List[float]
        Label column predictions for each sample
    y_score: None
        Not used, but used for interfacing.

    Returns
    -------
    metrics: pd.DataFrame
        Dataframe with each metric as a separate row.
    """

    metrics = []
    metrics.append(["mse", round(mean_squared_error(y_true, y_pred), 5)])
    metrics.append(["rmse", round(mean_squared_error(y_true, y_pred, squared=False), 5)])
    metrics.append(["r2", round(r2_score(y_true, y_pred), 5)])
    metrics.append(["mae", round(mean_absolute_error(y_true, y_pred), 5)])

    if return_type == "dataframe":
        return pd.DataFrame(metrics, columns=["Metric", "Value"])
    elif return_type == "dict":
        return dict(metrics)
    else:
        raise ValueError("return_type must be either 'dataframe' or 'dict'.")
