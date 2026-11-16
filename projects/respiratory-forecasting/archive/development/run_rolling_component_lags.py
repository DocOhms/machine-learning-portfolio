from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Public archival adapter: only input/output path setup differs from the source.
# Calculation helpers below are the matching legacy_project/src copies, not the
# current primary pipeline. See README.md and SOURCE_MANIFEST.json for lineage.
parser = argparse.ArgumentParser(description="Archived development calculation; explicit paths required. See README.md.")
parser.add_argument("--legacy-project", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
args = parser.parse_args()
PROJECT = args.legacy_project.resolve()
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "src"))

import common_config
from common_features import (
    add_time_series_features,
    core_feature_columns,
    fit_preprocessor,
    transform_preprocessor,
)
from common_io import load_nssp
from common_metrics import evaluate_predictions
from common_models import compute_validation_weights


TARGET = "percent_visits_combined"
GROUPS = ["geography", "county"]
TEST_CUTOFF = 202419
N_FOLDS = 3
VALIDATION_WEEKS = 10

DETAIL_PATH = OUTPUT / "nssp_rolling_component_lags_fold_metrics.csv"
WEIGHT_PATH = OUTPUT / "nssp_rolling_component_lags_weights.csv"
FOLD_PATH = OUTPUT / "nssp_rolling_component_lags_folds.csv"
SUMMARY_PATH = OUTPUT / "nssp_rolling_component_lags_summary.csv"


def save_rows(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    started = time.time()
    print("Loading NSSP and constructing component lags 1-3...", flush=True)
    raw = load_nssp(PROJECT / "data" / "raw" / "NSSP.csv", target=TARGET)
    lagged_covariates = {
        column: [1, 2, 3] for column in common_config.NSSP_COMPONENT_COLUMNS
    }
    engineered = add_time_series_features(
        raw,
        target=TARGET,
        group_cols=GROUPS,
        lagged_covariates=lagged_covariates,
    )

    core = core_feature_columns(engineered)
    component_features = {
        lag: [
            f"{column}_lag_{value}"
            for column in common_config.NSSP_COMPONENT_COLUMNS
            for value in range(1, lag + 1)
        ]
        for lag in [1, 2, 3]
    }
    configurations = {
        "Core_Only": core,
        "Components_Lag1": core + component_features[1],
        "Components_Lag1_2": core + component_features[2],
        "Components_Lag1_2_3": core + component_features[3],
    }

    # Require every candidate feature once so all configurations use the same rows.
    union_features = configurations["Components_Lag1_2_3"]
    cohort = engineered.dropna(subset=[TARGET] + union_features).copy()
    pretest = cohort[cohort["epi_order"] < TEST_CUTOFF].copy()
    times = np.array(sorted(pretest["epi_order"].unique()))
    required_weeks = N_FOLDS * VALIDATION_WEEKS
    if len(times) <= required_weeks:
        raise ValueError("Not enough pre-test weeks for the requested rolling folds.")

    fold_rows: list[dict] = []
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    validation_start = len(times) - required_weeks
    for fold in range(1, N_FOLDS + 1):
        start = validation_start + (fold - 1) * VALIDATION_WEEKS
        stop = start + VALIDATION_WEEKS
        train_times = times[:start]
        validation_times = times[start:stop]
        train_frame = pretest[pretest["epi_order"].isin(train_times)]
        validation_frame = pretest[pretest["epi_order"].isin(validation_times)]
        fold_rows.append(
            {
                "Fold": fold,
                "Train_Start": int(train_times[0]),
                "Train_End": int(train_times[-1]),
                "Validation_Start": int(validation_times[0]),
                "Validation_End": int(validation_times[-1]),
                "Train_Weeks": len(train_times),
                "Validation_Weeks": len(validation_times),
                "Train_Rows": len(train_frame),
                "Validation_Rows": len(validation_frame),
            }
        )
        folds.append((fold, train_times, validation_times))
    save_rows(FOLD_PATH, fold_rows)
    print(pd.DataFrame(fold_rows).to_string(index=False), flush=True)

    metric_rows: list[dict] = []
    weight_rows: list[dict] = []
    total = len(configurations) * len(folds)
    completed = 0

    for configuration, features in configurations.items():
        for fold, train_times, validation_times in folds:
            run_started = time.time()
            train_frame = pretest[pretest["epi_order"].isin(train_times)]
            validation_frame = pretest[pretest["epi_order"].isin(validation_times)]
            print(
                f"[{completed + 1}/{total}] {configuration}, fold {fold}: fitting...",
                flush=True,
            )
            X_train, train_aligned, preprocessor, final_features = fit_preprocessor(
                train_frame,
                features,
                n_clusters=common_config.N_CLUSTERS,
                use_clustering=True,
            )
            X_validation, validation_aligned = transform_preprocessor(
                validation_frame, preprocessor
            )
            weights, errors, predictions = compute_validation_weights(
                X_train,
                train_aligned[TARGET],
                X_validation,
                validation_aligned[TARGET],
            )
            metrics = evaluate_predictions(validation_aligned[TARGET], predictions)
            elapsed = time.time() - run_started
            for row in metrics.to_dict("records"):
                metric_rows.append(
                    {
                        "Configuration": configuration,
                        "Fold": fold,
                        **row,
                        "Raw_Feature_Count": len(features),
                        "Final_Feature_Count": len(final_features),
                        "Effective_Clusters": preprocessor.effective_n_clusters,
                        "Train_Rows": len(train_aligned),
                        "Validation_Rows": len(validation_aligned),
                        "Validation_Start": int(validation_times[0]),
                        "Validation_End": int(validation_times[-1]),
                        "Elapsed_Minutes": elapsed / 60.0,
                    }
                )
            for model, weight in weights.items():
                weight_rows.append(
                    {
                        "Configuration": configuration,
                        "Fold": fold,
                        "Model": model,
                        "Weight": weight,
                        "Validation_MSE": errors[model],
                    }
                )
            save_rows(DETAIL_PATH, metric_rows)
            save_rows(WEIGHT_PATH, weight_rows)
            completed += 1
            hybrid = metrics[metrics["Model"] == "Hybrid_InvMSE"].iloc[0]
            print(
                f"[{completed}/{total}] completed in {elapsed / 60:.1f} min: "
                f"RMSE={hybrid['RMSE']:.6f}, R2={hybrid['R2']:.6f}",
                flush=True,
            )

    detail = pd.DataFrame(metric_rows)
    hybrid = detail[detail["Model"] == "Hybrid_InvMSE"].copy()
    summary = (
        hybrid.groupby("Configuration", as_index=False)
        .agg(
            Folds=("Fold", "nunique"),
            Mean_RMSE=("RMSE", "mean"),
            SD_RMSE=("RMSE", "std"),
            Mean_MAE=("MAE", "mean"),
            SD_MAE=("MAE", "std"),
            Mean_R2=("R2", "mean"),
            SD_R2=("R2", "std"),
            Mean_SMAPE=("SMAPE", "mean"),
            Mean_Bias=("Bias", "mean"),
            Mean_Pearson=("Pearson", "mean"),
        )
        .sort_values("Mean_RMSE")
        .reset_index(drop=True)
    )
    core_rmse = hybrid[hybrid["Configuration"] == "Core_Only"].set_index("Fold")[
        "RMSE"
    ]
    win_counts = {}
    for configuration, frame in hybrid.groupby("Configuration"):
        candidate = frame.set_index("Fold")["RMSE"]
        win_counts[configuration] = int((candidate < core_rmse).sum())
    summary["Folds_Beating_Core_RMSE"] = summary["Configuration"].map(win_counts)
    summary["Rank_By_Mean_RMSE"] = np.arange(1, len(summary) + 1)
    summary["Selected_By_Rolling_Validation"] = summary["Rank_By_Mean_RMSE"] == 1
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nROLLING-VALIDATION SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nTotal elapsed: {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
