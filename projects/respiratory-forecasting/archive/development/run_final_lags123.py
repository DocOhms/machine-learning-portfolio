from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
    add_baseline_predictions,
    add_time_series_features,
    chronological_split,
    chronological_validation_split,
    core_feature_columns,
    fit_preprocessor,
    transform_preprocessor,
)
from common_io import load_nssp
from common_metrics import evaluate_predictions
from common_models import compute_validation_weights, fit_predict_final_models


TARGET = "percent_visits_combined"
GROUPS = ["geography", "county"]
EXPECTED_TEST_CUTOFF = 202419
EXPECTED_VALIDATION_CUTOFF = 202402
EXPECTED_TEST_ROWS = 47534


def main() -> None:
    started = time.time()
    print("Loading NSSP and engineering locked component lags [1,2,3]...", flush=True)
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
    component_features = [
        f"{column}_lag_{lag}"
        for column in common_config.NSSP_COMPONENT_COLUMNS
        for lag in [1, 2, 3]
    ]
    feature_cols = core_feature_columns(engineered) + component_features
    modeling = engineered.dropna(subset=[TARGET] + feature_cols).copy()

    train, test, test_cutoff = chronological_split(modeling)
    fit, validation, validation_cutoff = chronological_validation_split(train)
    if int(test_cutoff) != EXPECTED_TEST_CUTOFF:
        raise AssertionError(f"Unexpected test cutoff: {test_cutoff}")
    if int(validation_cutoff) != EXPECTED_VALIDATION_CUTOFF:
        raise AssertionError(f"Unexpected validation cutoff: {validation_cutoff}")
    if len(test) != EXPECTED_TEST_ROWS:
        raise AssertionError(f"Unexpected test row count: {len(test)}")
    print(
        f"Verified cohort: fit={len(fit):,}, validation={len(validation):,}, "
        f"train={len(train):,}, test={len(test):,}; "
        f"cutoffs validation={validation_cutoff}, test={test_cutoff}",
        flush=True,
    )

    print("Calculating chronological validation weights...", flush=True)
    phase_started = time.time()
    X_fit, fit_aligned, validation_preprocessor, _ = fit_preprocessor(
        fit,
        feature_cols,
        n_clusters=common_config.N_CLUSTERS,
        use_clustering=True,
    )
    X_validation, validation_aligned = transform_preprocessor(
        validation, validation_preprocessor
    )
    weights, validation_errors, validation_predictions = compute_validation_weights(
        X_fit,
        fit_aligned[TARGET],
        X_validation,
        validation_aligned[TARGET],
    )
    validation_metrics = evaluate_predictions(
        validation_aligned[TARGET], validation_predictions
    )
    print(
        f"Validation weighting completed in {(time.time() - phase_started) / 60:.1f} min",
        flush=True,
    )

    print("Refitting preprocessing and models on the complete pre-test history...", flush=True)
    phase_started = time.time()
    X_train, train_aligned, final_preprocessor, final_features = fit_preprocessor(
        train,
        feature_cols,
        n_clusters=common_config.N_CLUSTERS,
        use_clustering=True,
    )
    X_test, test_aligned = transform_preprocessor(test, final_preprocessor)
    if len(test_aligned) != EXPECTED_TEST_ROWS:
        raise AssertionError(f"Transformed test row count changed: {len(test_aligned)}")
    predictions, _ = fit_predict_final_models(
        X_train,
        train_aligned[TARGET],
        X_test,
        weights,
    )
    prediction_frame = add_baseline_predictions(test_aligned)
    for name, values in predictions.items():
        prediction_frame[name] = values
    prediction_names = [
        "Persistence",
        "MovingAverage3",
        "SeasonalNaive",
        *predictions.keys(),
    ]
    test_metrics = evaluate_predictions(
        test_aligned[TARGET],
        {
            name: prediction_frame[name].to_numpy()
            for name in prediction_names
            if name in prediction_frame
        },
    )
    print(
        f"Final refit/test completed in {(time.time() - phase_started) / 60:.1f} min",
        flush=True,
    )

    prefix = "nssp_locked_component_lags123"
    test_metrics.to_csv(OUTPUT / f"{prefix}_test_metrics.csv", index=False)
    validation_metrics.to_csv(
        OUTPUT / f"{prefix}_validation_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "Model": list(weights),
            "Weight": [weights[name] for name in weights],
            "Validation_MSE": [validation_errors[name] for name in weights],
        }
    ).to_csv(OUTPUT / f"{prefix}_weights.csv", index=False)
    pd.DataFrame({"Feature": final_features}).to_csv(
        OUTPUT / f"{prefix}_features.csv", index=False
    )
    prediction_columns = [
        column
        for column in [
            "date",
            "week_end",
            "epi_order",
            "YEAR",
            "WEEK",
            "geography",
            "county",
            TARGET,
            *prediction_names,
        ]
        if column in prediction_frame
    ]
    prediction_frame[prediction_columns].to_csv(
        OUTPUT / f"{prefix}_predictions.csv", index=False
    )

    metadata = pd.DataFrame(
        [
            {
                "Configuration": "Components_Lag1_2_3",
                "Selection_Basis": "Lowest mean RMSE across three rolling-origin validation windows",
                "Modeling_Rows": len(modeling),
                "Fit_Rows": len(fit_aligned),
                "Validation_Rows": len(validation_aligned),
                "Train_Rows": len(train_aligned),
                "Test_Rows": len(test_aligned),
                "Validation_Cutoff": validation_cutoff,
                "Test_Cutoff": test_cutoff,
                "Raw_Feature_Count": len(feature_cols),
                "Final_Feature_Count": len(final_features),
                "Requested_Clusters": common_config.N_CLUSTERS,
                "Effective_Clusters": final_preprocessor.effective_n_clusters,
                "Elapsed_Minutes": (time.time() - started) / 60.0,
            }
        ]
    )
    metadata.to_csv(OUTPUT / f"{prefix}_metadata.csv", index=False)

    lag1_path = OUTPUT / "nssp_t_plus_1_lag1_comparison.csv"
    lag1 = pd.read_csv(lag1_path)
    lag1_test = lag1[
        (lag1["Configuration"] == "Core_Plus_Component_Lag1")
        & (lag1["Split"] == "Test")
    ].iloc[0]
    locked_test = test_metrics[test_metrics["Model"] == "Hybrid_InvMSE"].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "Configuration": "Components_Lag1",
                "N": int(lag1_test["N"]),
                "MAE": lag1_test["MAE"],
                "RMSE": lag1_test["RMSE"],
                "R2": lag1_test["R2"],
                "SMAPE": lag1_test["SMAPE"],
                "Bias": lag1_test["Bias"],
                "Pearson": lag1_test["Pearson"],
            },
            {
                "Configuration": "Components_Lag1_2_3",
                "N": int(locked_test["N"]),
                "MAE": locked_test["MAE"],
                "RMSE": locked_test["RMSE"],
                "R2": locked_test["R2"],
                "SMAPE": locked_test["SMAPE"],
                "Bias": locked_test["Bias"],
                "Pearson": locked_test["Pearson"],
            },
        ]
    )
    comparison["R2_Delta_vs_Lag1"] = comparison["R2"] - lag1_test["R2"]
    comparison["RMSE_Delta_vs_Lag1"] = comparison["RMSE"] - lag1_test["RMSE"]
    comparison["MAE_Delta_vs_Lag1"] = comparison["MAE"] - lag1_test["MAE"]
    comparison.to_csv(OUTPUT / f"{prefix}_comparison_vs_lag1.csv", index=False)

    print("\nLOCKED FINAL TEST METRICS", flush=True)
    print(test_metrics.to_string(index=False), flush=True)
    print("\nDIRECT HYBRID COMPARISON", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"\nTotal elapsed: {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
