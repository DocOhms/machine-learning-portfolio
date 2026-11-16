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
    add_baseline_predictions,
    add_time_series_features,
    chronological_split,
    chronological_validation_split,
    core_feature_columns,
    explicit_epi_split,
    fit_preprocessor,
    transform_preprocessor,
)
from common_io import load_ilinet
from common_metrics import evaluate_predictions
from common_models import compute_validation_weights, fit_predict_final_models


TARGET = "%UNWEIGHTED ILI"
GROUPS = ["REGION TYPE", "REGION"]
PROVIDER = "NUM. OF PROVIDERS"
VALIDATION_WEEKS = 10
N_ROLLING_FOLDS = 3
PREFIX = "ilinet_exp2_exp3_provider_lags"


EXPERIMENTS = {
    "Experiment_2_Contemporary": {
        "file": "ILINet.csv",
        "split": "chronological",
    },
    "Experiment_3_Pandemic": {
        "file": "ILINet_2016_2021.csv",
        "split": "explicit",
        "explicit": {
            "train_start": 201640,
            "train_end": 201939,
            "test_start": 201940,
            "test_end": 202039,
        },
    },
}


def hybrid_row(metrics: pd.DataFrame) -> pd.Series:
    return metrics[metrics["Model"] == "Hybrid_InvMSE"].iloc[0]


def main() -> None:
    started = time.time()
    rolling_rows: list[dict] = []
    fold_rows: list[dict] = []
    final_metric_rows: list[dict] = []
    final_weight_rows: list[dict] = []
    metadata_rows: list[dict] = []

    for experiment, spec in EXPERIMENTS.items():
        print(f"\n=== {experiment} ===", flush=True)
        raw = load_ilinet(PROJECT / "data" / "raw" / spec["file"], target=TARGET)
        engineered = add_time_series_features(
            raw,
            target=TARGET,
            group_cols=GROUPS,
            lagged_covariates={PROVIDER: [1, 2, 3]},
        )
        core = core_feature_columns(engineered)
        configurations = {
            "Provider_Lag1": core + [f"{PROVIDER}_lag_1"],
            "Provider_Lags1_2_3": core
            + [f"{PROVIDER}_lag_{lag}" for lag in [1, 2, 3]],
        }
        union_features = configurations["Provider_Lags1_2_3"]
        cohort = engineered.dropna(subset=[TARGET] + union_features).copy()

        if spec["split"] == "chronological":
            train, test, test_cutoff = chronological_split(cohort)
            split_description = f"test begins {int(test_cutoff)}"
        else:
            train, test = explicit_epi_split(cohort, **spec["explicit"])
            test_cutoff = spec["explicit"]["test_start"]
            split_description = str(spec["explicit"])

        train_times = np.array(sorted(train["epi_order"].unique()))
        required = N_ROLLING_FOLDS * VALIDATION_WEEKS
        if len(train_times) <= required:
            raise ValueError(f"Not enough training weeks for {experiment}")
        rolling_start = len(train_times) - required
        folds: list[tuple[int, np.ndarray, np.ndarray]] = []
        for fold in range(1, N_ROLLING_FOLDS + 1):
            start = rolling_start + (fold - 1) * VALIDATION_WEEKS
            stop = start + VALIDATION_WEEKS
            fold_train_times = train_times[:start]
            fold_validation_times = train_times[start:stop]
            fold_train = train[train["epi_order"].isin(fold_train_times)]
            fold_validation = train[train["epi_order"].isin(fold_validation_times)]
            folds.append((fold, fold_train_times, fold_validation_times))
            fold_rows.append(
                {
                    "Experiment": experiment,
                    "Fold": fold,
                    "Train_Start": int(fold_train_times[0]),
                    "Train_End": int(fold_train_times[-1]),
                    "Validation_Start": int(fold_validation_times[0]),
                    "Validation_End": int(fold_validation_times[-1]),
                    "Train_Weeks": len(fold_train_times),
                    "Validation_Weeks": len(fold_validation_times),
                    "Train_Rows": len(fold_train),
                    "Validation_Rows": len(fold_validation),
                }
            )

        for configuration, features in configurations.items():
            print(f"{configuration}: rolling validation", flush=True)
            for fold, fold_train_times, fold_validation_times in folds:
                fold_train = train[train["epi_order"].isin(fold_train_times)]
                fold_validation = train[
                    train["epi_order"].isin(fold_validation_times)
                ]
                X_fold_train, fold_train_aligned, preprocessor, _ = fit_preprocessor(
                    fold_train,
                    features,
                    n_clusters=common_config.N_CLUSTERS,
                    use_clustering=True,
                )
                X_fold_validation, fold_validation_aligned = transform_preprocessor(
                    fold_validation, preprocessor
                )
                _, _, predictions = compute_validation_weights(
                    X_fold_train,
                    fold_train_aligned[TARGET],
                    X_fold_validation,
                    fold_validation_aligned[TARGET],
                )
                metrics = evaluate_predictions(
                    fold_validation_aligned[TARGET], predictions
                )
                row = hybrid_row(metrics)
                rolling_rows.append(
                    {
                        "Experiment": experiment,
                        "Configuration": configuration,
                        "Fold": fold,
                        "N": int(row["N"]),
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "R2": row["R2"],
                        "SMAPE": row["SMAPE"],
                        "Bias": row["Bias"],
                        "Pearson": row["Pearson"],
                    }
                )
                print(
                    f"  fold {fold}: RMSE={row['RMSE']:.6f}, R2={row['R2']:.6f}",
                    flush=True,
                )

            print(f"{configuration}: final weighting/refit/test", flush=True)
            fit, validation, validation_cutoff = chronological_validation_split(train)
            X_fit, fit_aligned, validation_preprocessor, _ = fit_preprocessor(
                fit,
                features,
                n_clusters=common_config.N_CLUSTERS,
                use_clustering=True,
            )
            X_validation, validation_aligned = transform_preprocessor(
                validation, validation_preprocessor
            )
            weights, errors, _ = compute_validation_weights(
                X_fit,
                fit_aligned[TARGET],
                X_validation,
                validation_aligned[TARGET],
            )
            X_train, train_aligned, final_preprocessor, final_features = fit_preprocessor(
                train,
                features,
                n_clusters=common_config.N_CLUSTERS,
                use_clustering=True,
            )
            X_test, test_aligned = transform_preprocessor(test, final_preprocessor)
            predictions, _ = fit_predict_final_models(
                X_train, train_aligned[TARGET], X_test, weights
            )
            pred_frame = add_baseline_predictions(test_aligned)
            for name, values in predictions.items():
                pred_frame[name] = values
            names = [
                "Persistence",
                "MovingAverage3",
                "SeasonalNaive",
                *predictions.keys(),
            ]
            metrics = evaluate_predictions(
                test_aligned[TARGET],
                {name: pred_frame[name].to_numpy() for name in names},
            )
            for row in metrics.to_dict("records"):
                final_metric_rows.append(
                    {
                        "Experiment": experiment,
                        "Configuration": configuration,
                        **row,
                    }
                )
            for model, weight in weights.items():
                final_weight_rows.append(
                    {
                        "Experiment": experiment,
                        "Configuration": configuration,
                        "Model": model,
                        "Weight": weight,
                        "Validation_MSE": errors[model],
                    }
                )
            prediction_columns = [
                column
                for column in [
                    "date",
                    "epi_order",
                    "YEAR",
                    "WEEK",
                    "REGION TYPE",
                    "REGION",
                    TARGET,
                    *names,
                ]
                if column in pred_frame
            ]
            safe_experiment = experiment.lower()
            safe_configuration = configuration.lower()
            pred_frame[prediction_columns].to_csv(
                OUTPUT
                / f"{PREFIX}_{safe_experiment}_{safe_configuration}_predictions.csv",
                index=False,
            )
            metadata_rows.append(
                {
                    "Experiment": experiment,
                    "Configuration": configuration,
                    "Data_File": spec["file"],
                    "Split": split_description,
                    "Modeling_Rows": len(cohort),
                    "Fit_Rows": len(fit_aligned),
                    "Validation_Rows": len(validation_aligned),
                    "Train_Rows": len(train_aligned),
                    "Test_Rows": len(test_aligned),
                    "Validation_Cutoff": validation_cutoff,
                    "Test_Cutoff": test_cutoff,
                    "Raw_Feature_Count": len(features),
                    "Final_Feature_Count": len(final_features),
                    "Effective_Clusters": final_preprocessor.effective_n_clusters,
                }
            )
            final_hybrid = hybrid_row(metrics)
            print(
                f"  TEST RMSE={final_hybrid['RMSE']:.6f}, "
                f"R2={final_hybrid['R2']:.6f}",
                flush=True,
            )

    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(OUTPUT / f"{PREFIX}_rolling_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / f"{PREFIX}_folds.csv", index=False)
    final_metrics = pd.DataFrame(final_metric_rows)
    final_metrics.to_csv(OUTPUT / f"{PREFIX}_final_test_metrics.csv", index=False)
    pd.DataFrame(final_weight_rows).to_csv(
        OUTPUT / f"{PREFIX}_final_weights.csv", index=False
    )
    pd.DataFrame(metadata_rows).to_csv(OUTPUT / f"{PREFIX}_metadata.csv", index=False)

    rolling_summary = (
        rolling.groupby(["Experiment", "Configuration"], as_index=False)
        .agg(
            Folds=("Fold", "nunique"),
            Mean_RMSE=("RMSE", "mean"),
            SD_RMSE=("RMSE", "std"),
            Mean_MAE=("MAE", "mean"),
            Mean_R2=("R2", "mean"),
            SD_R2=("R2", "std"),
        )
        .sort_values(["Experiment", "Mean_RMSE"])
    )
    rolling_summary.to_csv(OUTPUT / f"{PREFIX}_rolling_summary.csv", index=False)
    final_hybrid = final_metrics[final_metrics["Model"] == "Hybrid_InvMSE"].copy()
    final_hybrid.to_csv(OUTPUT / f"{PREFIX}_final_hybrid_comparison.csv", index=False)

    print("\nROLLING SUMMARY", flush=True)
    print(rolling_summary.to_string(index=False), flush=True)
    print("\nFINAL TEST HYBRID COMPARISON", flush=True)
    print(final_hybrid.to_string(index=False), flush=True)
    print(f"\nTotal elapsed: {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
