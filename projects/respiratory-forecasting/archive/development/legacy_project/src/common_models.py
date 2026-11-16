"""Shared model training and inverse-MSE ensemble logic."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from common_config import MODEL_PARAMS


def build_models():
    return {
        "RandomForest": RandomForestRegressor(**MODEL_PARAMS["RandomForest"]),
        "GradientBoosting": GradientBoostingRegressor(**MODEL_PARAMS["GradientBoosting"]),
        "MLP": MLPRegressor(**MODEL_PARAMS["MLP"]),
    }


def inverse_mse_weights(
    model_errors: dict[str, float],
    eps: float = 1e-12,
) -> dict[str, float]:
    clean_errors: dict[str, float] = {}
    for name, value in model_errors.items():
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Validation MSE for {name} is not finite: {value}")
        clean_errors[name] = max(value, eps)

    inverse = {name: 1.0 / value for name, value in clean_errors.items()}
    total = sum(inverse.values())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Unable to normalize inverse-MSE ensemble weights.")
    return {name: value / total for name, value in inverse.items()}


def compute_validation_weights(X_fit, y_fit, X_val, y_val):
    """Fit on earlier training weeks and weight models on later training weeks."""
    val_errors: dict[str, float] = {}
    val_predictions: dict[str, np.ndarray] = {}

    for name, model in build_models().items():
        model.fit(X_fit, y_fit)
        prediction = model.predict(X_val)
        val_predictions[name] = prediction
        val_errors[name] = float(mean_squared_error(y_val, prediction))

    weights = inverse_mse_weights(val_errors)
    hybrid = np.zeros(len(X_val), dtype=float)
    for name, prediction in val_predictions.items():
        hybrid += weights[name] * prediction
    val_predictions["Hybrid_InvMSE"] = hybrid
    return weights, val_errors, val_predictions


def fit_predict_final_models(X_train, y_train, X_test, weights: dict[str, float]):
    """Refit each model on the complete training period and predict the test set."""
    predictions: dict[str, np.ndarray] = {}
    final_models = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        final_models[name] = model
        predictions[name] = model.predict(X_test)

    hybrid = np.zeros(len(X_test), dtype=float)
    for name, prediction in predictions.items():
        if name not in weights:
            raise KeyError(f"Missing ensemble weight for model: {name}")
        hybrid += weights[name] * prediction
    predictions["Hybrid_InvMSE"] = hybrid
    return predictions, final_models


def train_predict_models(
    X_train,
    y_train,
    X_test,
    *,
    X_val=None,
    y_val=None,
):
    """Compatibility wrapper for diagnostics.

    New experiment code should pass an explicit chronological validation set.
    """
    if X_val is None or y_val is None:
        raise ValueError(
            "An explicit chronological validation set is required. "
            "Use compute_validation_weights() and fit_predict_final_models(), "
            "or pass X_val and y_val to this wrapper."
        )

    weights, val_errors, _ = compute_validation_weights(X_train, y_train, X_val, y_val)
    predictions, final_models = fit_predict_final_models(X_train, y_train, X_test, weights)
    return predictions, final_models, weights, val_errors
