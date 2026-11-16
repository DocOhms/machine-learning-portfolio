"""Leakage-safe feature engineering, temporal splitting, and preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common_config


@dataclass
class FittedPreprocessor:
    """Artifacts needed to transform validation or test observations."""

    scaler: StandardScaler
    feature_cols: list[str]
    use_clustering: bool
    kmeans: KMeans | None
    cluster_feature_cols: list[str]
    effective_n_clusters: int
    silhouette: float


def add_time_series_features(
    df: pd.DataFrame,
    target: str,
    group_cols: list[str],
    time_col: str = "epi_order",
    week_col: str = "WEEK",
    lagged_covariates: dict[str, Iterable[int]] | None = None,
) -> pd.DataFrame:
    """Add target-history and optional lagged-covariate features.

    Every rolling statistic is formed from values shifted by one time step, so
    the row for week t uses only target observations from weeks before t.
    Optional covariates are also explicitly lagged before they can be selected.
    """
    required = set(group_cols + [time_col, target])
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns required for feature engineering: {sorted(missing)}")

    out = df.sort_values(group_cols + [time_col]).copy()
    target_group = out.groupby(group_cols, dropna=False, sort=False)[target]

    for lag in common_config.LAGS:
        out[f"lag_{lag}"] = target_group.shift(lag)

    out[f"rolling_mean_{common_config.ROLLING_WINDOW}"] = target_group.transform(
        lambda series: series.shift(1)
        .rolling(common_config.ROLLING_WINDOW, min_periods=common_config.ROLLING_WINDOW)
        .mean()
    )
    out[f"rolling_std_{common_config.ROLLING_WINDOW}"] = target_group.transform(
        lambda series: series.shift(1)
        .rolling(common_config.ROLLING_WINDOW, min_periods=common_config.ROLLING_WINDOW)
        .std()
    )
    out["seasonal_naive"] = target_group.shift(common_config.SEASONAL_LAG)

    if week_col in out.columns:
        week = pd.to_numeric(out[week_col], errors="coerce")
        out["week_sin"] = np.sin(2.0 * np.pi * week / 52.0)
        out["week_cos"] = np.cos(2.0 * np.pi * week / 52.0)

    for column, lags in (lagged_covariates or {}).items():
        if column not in out.columns:
            continue
        covariate_group = out.groupby(group_cols, dropna=False, sort=False)[column]
        for lag in sorted(set(int(value) for value in lags)):
            if lag < 1:
                raise ValueError(f"Covariate lag must be at least 1; got {lag} for {column}.")
            out[f"{column}_lag_{lag}"] = covariate_group.shift(lag)

    return out


def core_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        *[f"lag_{lag}" for lag in common_config.LAGS],
        f"rolling_mean_{common_config.ROLLING_WINDOW}",
        f"rolling_std_{common_config.ROLLING_WINDOW}",
        "week_sin",
        "week_cos",
    ]
    return [column for column in candidates if column in df.columns]


def choose_feature_columns(df: pd.DataFrame, target: str, dataset_type: str) -> list[str]:
    """Return an explicit, reproducible forecasting feature set.

    The function deliberately avoids adding every numeric column. This prevents
    accidental inclusion of same-week targets, smoothed outcomes, identifiers,
    or newly introduced numeric fields.
    """
    features = core_feature_columns(df)
    dataset_type = dataset_type.lower()

    if dataset_type == "nssp":
        extras = [
            f"{column}_lag_1"
            for column in common_config.NSSP_COMPONENT_COLUMNS
            if f"{column}_lag_1" in df.columns
        ]
    elif dataset_type == "ilinet":
        extras = [
            f"{column}_lag_1"
            for column in common_config.ILINET_REPORTING_COLUMNS
            if f"{column}_lag_1" in df.columns
        ]
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    selected = features + extras
    if target in selected:
        raise RuntimeError("Target leakage detected: target was selected as a feature.")
    return selected


def _validate_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise KeyError(f"Requested feature columns are missing: {missing}")
    if not feature_cols:
        raise ValueError("At least one feature column is required.")


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "epi_order",
    test_size: float | None = None,
):
    """Split by unique time values, preserving all locations in each period."""
    test_size = common_config.TEST_SIZE if test_size is None else test_size
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    unique_times = np.array(sorted(df[time_col].dropna().unique()))
    if len(unique_times) < 2:
        raise ValueError("At least two unique time values are required for a train/test split.")

    test_n = max(1, int(np.ceil(len(unique_times) * test_size)))
    split_idx = max(1, len(unique_times) - test_n)
    cutoff_time = unique_times[split_idx]

    train = df[df[time_col] < cutoff_time].copy()
    test = df[df[time_col] >= cutoff_time].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    return train, test, cutoff_time


def chronological_validation_split(
    train: pd.DataFrame,
    time_col: str = "epi_order",
    validation_fraction: float | None = None,
):
    """Use the latest training weeks as the ensemble-weight validation period."""
    validation_fraction = (
        common_config.VALIDATION_FRACTION_FOR_ENSEMBLE_WEIGHTS
        if validation_fraction is None
        else validation_fraction
    )
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    unique_times = np.array(sorted(train[time_col].dropna().unique()))
    if len(unique_times) < 2:
        raise ValueError("At least two unique training time values are required for validation.")

    val_n = max(1, int(np.ceil(len(unique_times) * validation_fraction)))
    split_idx = max(1, len(unique_times) - val_n)
    validation_cutoff = unique_times[split_idx]

    fit = train[train[time_col] < validation_cutoff].copy()
    validation = train[train[time_col] >= validation_cutoff].copy()
    if fit.empty or validation.empty:
        raise ValueError("Chronological validation split produced an empty subset.")
    return fit, validation, validation_cutoff


def explicit_epi_split(
    df: pd.DataFrame,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
):
    if train_end >= test_start:
        raise ValueError("Explicit train and test periods must not overlap.")
    train = df[(df["epi_order"] >= train_start) & (df["epi_order"] <= train_end)].copy()
    test = df[(df["epi_order"] >= test_start) & (df["epi_order"] <= test_end)].copy()
    if train.empty or test.empty:
        raise ValueError("Explicit epidemiological split produced an empty train or test set.")
    return train, test


def _append_cluster_indicators(
    X: pd.DataFrame,
    labels: np.ndarray,
    cluster_feature_cols: list[str],
) -> pd.DataFrame:
    out = X.copy()
    for cluster_index, column in enumerate(cluster_feature_cols):
        out[column] = (labels == cluster_index).astype(float)
    return out


def fit_preprocessor(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_clusters: int | None = None,
    use_clustering: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, FittedPreprocessor, list[str]]:
    """Fit scaling and optional K-Means using only the supplied training subset."""
    _validate_feature_columns(train, feature_cols)
    aligned = train.dropna(subset=feature_cols).copy()
    if aligned.empty:
        raise ValueError("No complete training rows remain after feature filtering.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(aligned[feature_cols])
    X = pd.DataFrame(scaled, columns=feature_cols, index=aligned.index)

    kmeans: KMeans | None = None
    cluster_cols: list[str] = []
    effective_k = 0
    silhouette = np.nan

    if use_clustering:
        requested_k = common_config.N_CLUSTERS if n_clusters is None else int(n_clusters)
        if requested_k < 2:
            raise ValueError(
                "n_clusters must be at least 2 when clustering is enabled. "
                "Use use_clustering=False for the no-clustering ablation."
            )
        unique_rows = np.unique(scaled, axis=0).shape[0]
        effective_k = min(requested_k, len(aligned), unique_rows)
        if effective_k < 2:
            raise ValueError("Fewer than two unique training observations are available for K-Means.")

        kmeans = KMeans(
            n_clusters=effective_k,
            random_state=common_config.RANDOM_STATE,
            n_init=common_config.KMEANS_N_INIT,
        )
        labels = kmeans.fit_predict(scaled)
        cluster_cols = [f"cluster_{index}" for index in range(effective_k)]
        X = _append_cluster_indicators(X, labels, cluster_cols)

        if len(aligned) > effective_k and np.unique(labels).size > 1:
            sample_size = min(common_config.SILHOUETTE_SAMPLE_SIZE, len(aligned))
            silhouette = float(
                silhouette_score(
                    scaled,
                    labels,
                    sample_size=sample_size if sample_size < len(aligned) else None,
                    random_state=common_config.RANDOM_STATE,
                )
            )

    preprocessor = FittedPreprocessor(
        scaler=scaler,
        feature_cols=list(feature_cols),
        use_clustering=use_clustering,
        kmeans=kmeans,
        cluster_feature_cols=cluster_cols,
        effective_n_clusters=effective_k,
        silhouette=silhouette,
    )
    final_feature_cols = list(feature_cols) + cluster_cols
    return X, aligned, preprocessor, final_feature_cols


def transform_preprocessor(
    frame: pd.DataFrame,
    preprocessor: FittedPreprocessor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform observations without refitting scaling or clustering."""
    _validate_feature_columns(frame, preprocessor.feature_cols)
    aligned = frame.dropna(subset=preprocessor.feature_cols).copy()
    if aligned.empty:
        raise ValueError("No complete rows remain after feature filtering.")

    scaled = preprocessor.scaler.transform(aligned[preprocessor.feature_cols])
    X = pd.DataFrame(scaled, columns=preprocessor.feature_cols, index=aligned.index)

    if preprocessor.use_clustering:
        if preprocessor.kmeans is None:
            raise RuntimeError("Clustering is enabled but no fitted K-Means model is available.")
        labels = preprocessor.kmeans.predict(scaled)
        X = _append_cluster_indicators(X, labels, preprocessor.cluster_feature_cols)

    return X, aligned


def fit_transform_preprocessor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_clusters: int | None = None,
    use_clustering: bool = True,
):
    """Compatibility wrapper used by standalone diagnostics."""
    X_train, train_aligned, preprocessor, final_features = fit_preprocessor(
        train,
        feature_cols,
        n_clusters=n_clusters,
        use_clustering=use_clustering,
    )
    X_test, test_aligned = transform_preprocessor(test, preprocessor)
    return (
        X_train,
        X_test,
        train_aligned,
        test_aligned,
        preprocessor.scaler,
        preprocessor.kmeans,
        final_features,
    )


def add_baseline_predictions(test: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    out["Persistence"] = out["lag_1"]
    out["MovingAverage3"] = out[f"rolling_mean_{common_config.ROLLING_WINDOW}"]
    out["SeasonalNaive"] = (
        out["seasonal_naive"] if "seasonal_naive" in out.columns else np.nan
    )
    return out
