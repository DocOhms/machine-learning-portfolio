"""Shared forecasting metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def bias(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def evaluate_predictions(y_true, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in predictions.items():
        frame = pd.DataFrame({"y": y_true, "pred": pred}).dropna()
        if frame.empty:
            continue
        yt, yp = frame["y"].values, frame["pred"].values
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": r2_score(yt, yp) if len(yt) > 1 else np.nan,
            "SMAPE": smape(yt, yp),
            "Bias": bias(yt, yp),
            "Pearson": pearson_corr(yt, yp),
            "N": len(yt),
        })
    return pd.DataFrame(rows).sort_values("RMSE")
