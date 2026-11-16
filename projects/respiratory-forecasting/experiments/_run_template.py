# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
# Per-run fit/validation/test boundaries are implemented below. This does not erase development history: all three final test windows were inspected during auxiliary-lag development. See docs/CALCULATIONS.md.
"""Internal runner shared by all respiratory forecasting experiments."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import common_config
from common_features import (
    add_baseline_predictions,
    add_time_series_features,
    choose_feature_columns,
    chronological_split,
    chronological_validation_split,
    explicit_epi_split,
    fit_preprocessor,
    transform_preprocessor,
)
from common_io import ensure_dirs
from common_metrics import evaluate_predictions
from common_models import compute_validation_weights, fit_predict_final_models
from common_plots import (
    plot_actual_vs_predictions,
    plot_metric_bar,
    plot_residuals,
    plot_state_choropleth,
)


def _validate_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} columns are missing: {missing}")


# Calculation provenance:
#   _validate_columns -> local function in this file
#   add_baseline_predictions -> common_features.add_baseline_predictions (src/common_features.py)
#   add_time_series_features -> common_features.add_time_series_features (src/common_features.py)
#   choose_feature_columns -> common_features.choose_feature_columns (src/common_features.py)
#   chronological_split -> common_features.chronological_split (src/common_features.py)
#   chronological_validation_split -> common_features.chronological_validation_split (src/common_features.py)
#   compute_validation_weights -> common_models.compute_validation_weights (src/common_models.py)
#   ensure_dirs -> common_io.ensure_dirs (src/common_io.py)
#   evaluate_predictions -> common_metrics.evaluate_predictions (src/common_metrics.py)
#   explicit_epi_split -> common_features.explicit_epi_split (src/common_features.py)
#   fit_predict_final_models -> common_models.fit_predict_final_models (src/common_models.py)
#   fit_preprocessor -> common_features.fit_preprocessor (src/common_features.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame({'Feature': final_features}).to_csv -> pandas.DataFrame({'Feature': final_features}).to_csv (third-party package)
#   pd.DataFrame({'Model': list(weights.keys()), 'Weight': list(weights.values()), 'Validation_MSE': [val_errors[name] for name in weights]}).to_csv -> pandas.DataFrame({'Model': list(weights.keys()), 'Weight': list(weights.values()), 'Validation_MSE': [val_errors[name] for name in weights]}).to_csv (third-party package)
#   plot_actual_vs_predictions -> common_plots.plot_actual_vs_predictions (src/common_plots.py)
#   plot_metric_bar -> common_plots.plot_metric_bar (src/common_plots.py)
#   plot_residuals -> common_plots.plot_residuals (src/common_plots.py)
#   plot_state_choropleth -> common_plots.plot_state_choropleth (src/common_plots.py)
#   transform_preprocessor -> common_features.transform_preprocessor (src/common_features.py)
def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    target: str,
    dataset_type: str,
    group_cols: list[str],
    split_mode: str = "chronological",
    explicit_split: dict | None = None,
    *,
    n_clusters: int | None = None,
    use_clustering: bool = True,
    lagged_covariates: dict[str, Iterable[int]] | None = None,
    feature_cols_override: list[str] | None = None,
    cohort_feature_cols_override: list[str] | None = None,
    align_to_default_feature_cohort: bool = False,
    make_figures: bool = True,
):
    """Run one leakage-safe experiment and save reproducible outputs.

    The test set is never used for preprocessing, cluster fitting, model
    weighting, or model selection. Ensemble weights are calculated from the
    latest chronological portion of the training period. Scaling and K-Means
    are fitted separately on the fit subset for validation, then refitted on
    the complete training period before final test evaluation.
    """
    n_clusters = common_config.N_CLUSTERS if n_clusters is None else int(n_clusters)

    out_root = PROJECT_ROOT / "results" / experiment_name
    pred_dir = out_root / "predictions"
    metric_dir = out_root / "metrics"
    fig_dir = out_root / "figures"
    ensure_dirs(pred_dir, metric_dir, fig_dir)

    engineered = add_time_series_features(
        df,
        target=target,
        group_cols=group_cols,
        lagged_covariates=lagged_covariates,
    )
    default_feature_cols = choose_feature_columns(
        engineered,
        target=target,
        dataset_type=dataset_type,
    )
    feature_cols = list(feature_cols_override or default_feature_cols)
    _validate_columns(engineered, feature_cols, "Model feature")

    if cohort_feature_cols_override is not None:
        cohort_feature_cols = list(cohort_feature_cols_override)
    elif align_to_default_feature_cohort:
        cohort_feature_cols = list(default_feature_cols)
    else:
        cohort_feature_cols = list(feature_cols)
    _validate_columns(engineered, cohort_feature_cols, "Cohort feature")

    modeling_df = engineered.dropna(subset=[target] + cohort_feature_cols).copy()
    if modeling_df.empty:
        raise ValueError("No modeling observations remain after complete-case filtering.")

    if split_mode == "explicit_epi":
        if explicit_split is None:
            raise ValueError("explicit_split settings are required for split_mode='explicit_epi'.")
        train, test = explicit_epi_split(modeling_df, **explicit_split)
        test_cutoff = explicit_split["test_start"]
        split_description = str(explicit_split)
    elif split_mode == "chronological":
        train, test, test_cutoff = chronological_split(modeling_df)
        split_description = f"test begins at epi_order {test_cutoff}"
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    fit_frame, validation_frame, validation_cutoff = chronological_validation_split(train)

    # Fit-only preprocessing for leakage-free validation and ensemble weighting.
    X_fit, fit_aligned, validation_preprocessor, _ = fit_preprocessor(
        fit_frame,
        feature_cols,
        n_clusters=n_clusters,
        use_clustering=use_clustering,
    )
    X_validation, validation_aligned = transform_preprocessor(
        validation_frame,
        validation_preprocessor,
    )
    y_fit = fit_aligned[target]
    y_validation = validation_aligned[target]

    weights, val_errors, validation_predictions = compute_validation_weights(
        X_fit,
        y_fit,
        X_validation,
        y_validation,
    )
    validation_metrics = evaluate_predictions(y_validation, validation_predictions)

    # Refit preprocessing and models on the full training period for final testing.
    X_train, train_aligned, final_preprocessor, final_features = fit_preprocessor(
        train,
        feature_cols,
        n_clusters=n_clusters,
        use_clustering=use_clustering,
    )
    X_test, test_aligned = transform_preprocessor(test, final_preprocessor)
    y_train = train_aligned[target]
    y_test = test_aligned[target]

    model_predictions, models = fit_predict_final_models(
        X_train,
        y_train,
        X_test,
        weights,
    )

    pred_df = add_baseline_predictions(test_aligned)
    for name, prediction in model_predictions.items():
        pred_df[name] = prediction

    prediction_cols = [
        "Persistence",
        "MovingAverage3",
        "SeasonalNaive",
        *model_predictions.keys(),
    ]
    metrics = evaluate_predictions(
        y_test,
        {
            column: pred_df[column].to_numpy()
            for column in prediction_cols
            if column in pred_df.columns
        },
    )

    pred_path = pred_dir / f"{experiment_name}_predictions.csv"
    metrics_path = metric_dir / f"{experiment_name}_metrics.csv"
    validation_metrics_path = metric_dir / f"{experiment_name}_validation_metrics.csv"
    weights_path = metric_dir / f"{experiment_name}_ensemble_weights.csv"
    feature_path = metric_dir / f"{experiment_name}_features_used.csv"
    metadata_path = metric_dir / f"{experiment_name}_run_metadata.csv"

    pred_df.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    validation_metrics.to_csv(validation_metrics_path, index=False)
    pd.DataFrame(
        {
            "Model": list(weights.keys()),
            "Weight": list(weights.values()),
            "Validation_MSE": [val_errors[name] for name in weights],
        }
    ).to_csv(weights_path, index=False)
    pd.DataFrame({"Feature": final_features}).to_csv(feature_path, index=False)

    metadata = pd.DataFrame(
        [
            {
                "Experiment": experiment_name,
                "Dataset_Type": dataset_type,
                "Split_Mode": split_mode,
                "Split_Description": split_description,
                "Test_Cutoff": test_cutoff,
                "Validation_Cutoff": validation_cutoff,
                "Requested_N_Clusters": n_clusters if use_clustering else 0,
                "Validation_Effective_N_Clusters": validation_preprocessor.effective_n_clusters,
                "Final_Effective_N_Clusters": final_preprocessor.effective_n_clusters,
                "Use_Clustering": use_clustering,
                "Validation_Train_Silhouette": validation_preprocessor.silhouette,
                "Full_Train_Silhouette": final_preprocessor.silhouette,
                "Fit_Rows": len(fit_aligned),
                "Validation_Rows": len(validation_aligned),
                "Train_Rows": len(train_aligned),
                "Test_Rows": len(test_aligned),
                "Feature_Count": len(final_features),
                "Cohort_Feature_Count": len(cohort_feature_cols),
            }
        ]
    )
    metadata.to_csv(metadata_path, index=False)

    if make_figures:
        readable_title = experiment_name.replace("_", " ").title().replace("Nssp", "NSSP").replace("Ilinet", "ILINet")
        plot_actual_vs_predictions(
            pred_df,
            target,
            fig_dir / f"{experiment_name}_actual_vs_predicted.png",
            readable_title,
        )
        plot_metric_bar(
            metrics,
            fig_dir / f"{experiment_name}_rmse_comparison.png",
            metric="RMSE",
            title=f"{readable_title} — RMSE Comparison",
        )
        plot_residuals(
            pred_df=pred_df,
            output_path=fig_dir / f"{experiment_name}_residuals.png",
            title=f"{readable_title} — Residual Analysis",
            event_line="2020-03-11" if "pandemic" in experiment_name else None,
            event_label="WHO Pandemic Declaration",
            target_col=target,
        )

        geo_col = next(
            (
                column
                for column in ["REGION", "geography", "geography_name", "state"]
                if column in pred_df.columns
            ),
            None,
        )
        if geo_col:
            state_level = pred_df.groupby(geo_col, as_index=False).agg(
                **{
                    "Observed": (target, "mean"),
                    "Hybrid_InvMSE": ("Hybrid_InvMSE", "mean"),
                }
            )
            state_level["Residual"] = state_level["Hybrid_InvMSE"] - state_level["Observed"]
            state_level.to_csv(
                metric_dir / f"{experiment_name}_state_test_period_summary.csv",
                index=False,
            )
            plot_state_choropleth(
                state_df=state_level,
                value_col="Hybrid_InvMSE",
                output_path=fig_dir / f"{experiment_name}_state_hotspots.png",
                title=f"{readable_title} — Mean Test-Period Predicted Burden",
                geo_col=geo_col,
            )

    return {
        "predictions": pred_df,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "weights": weights,
        "validation_errors": val_errors,
        "models": models,
        "final_features": final_features,
        "default_feature_cols": default_feature_cols,
        "cohort_feature_cols": cohort_feature_cols,
        "metadata": metadata,
        "paths": {
            "predictions": pred_path,
            "metrics": metrics_path,
            "validation_metrics": validation_metrics_path,
            "weights": weights_path,
            "features": feature_path,
            "metadata": metadata_path,
        },
    }
