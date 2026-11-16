# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Experiment 3 — Historical ILINet pandemic stress-test."""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "experiments"))

from common_config import DATA_RAW, ILINET_REPORTING_COLUMNS
from common_io import load_ilinet
from common_metrics import evaluate_predictions
from _run_template import run_experiment

DATA_FILE = DATA_RAW / "ILINet_2016_2021.csv"
TARGET = "%UNWEIGHTED ILI"
EVENT_DATE = pd.Timestamp("2020-03-11")
LAGGED_COVARIATES = {column: [1] for column in ILINET_REPORTING_COLUMNS}


def _national_weekly_series(pred_df: pd.DataFrame) -> pd.DataFrame:
    frame = pred_df.copy()
    if "REGION TYPE" in frame.columns:
        national_mask = frame["REGION TYPE"].astype(str).str.contains(
            "national", case=False, na=False
        )
        if national_mask.any():
            frame = frame[national_mask].copy()

    return (
        frame.dropna(subset=["date", TARGET, "Hybrid_InvMSE"])
        .groupby("date", as_index=False)[[TARGET, "Hybrid_InvMSE"]]
        .mean(numeric_only=True)
        .sort_values("date")
    )


# Calculation provenance:
#   Path -> pathlib.Path (Python standard library)
#   _national_weekly_series -> local function in this file
#   evaluate_predictions -> common_metrics.evaluate_predictions (src/common_metrics.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(rows).to_csv -> pandas.DataFrame(rows).to_csv (third-party package)
#   pd.to_datetime -> pandas.to_datetime (third-party package)
def save_stress_test_outputs(result: dict) -> None:
    pred_df = result["predictions"].copy()
    metrics_dir = Path(result["paths"]["metrics"]).parent

    weekly = _national_weekly_series(pred_df)
    if weekly.empty:
        raise ValueError("No observations are available for pandemic peak analysis.")

    actual_peak = weekly.loc[weekly[TARGET].idxmax()]
    predicted_peak = weekly.loc[weekly["Hybrid_InvMSE"].idxmax()]
    signed_timing_error = int((predicted_peak["date"] - actual_peak["date"]).days)

    peak_metrics = pd.DataFrame(
        [
            {
                "Actual_Peak_Date": actual_peak["date"],
                "Predicted_Peak_Date": predicted_peak["date"],
                "Signed_Timing_Error_Days": signed_timing_error,
                "Absolute_Timing_Error_Days": abs(signed_timing_error),
                "Actual_Peak_ILI": actual_peak[TARGET],
                "Predicted_Peak_ILI": predicted_peak["Hybrid_InvMSE"],
                "Signed_Magnitude_Error": predicted_peak["Hybrid_InvMSE"] - actual_peak[TARGET],
                "Absolute_Magnitude_Error": abs(
                    predicted_peak["Hybrid_InvMSE"] - actual_peak[TARGET]
                ),
            }
        ]
    )
    peak_metrics.to_csv(
        metrics_dir / "experiment_3_ilinet_pandemic_stress_test_peak_metrics.csv",
        index=False,
    )

    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    period_frames = {
        "Before_2020_03_11": pred_df[pred_df["date"] < EVENT_DATE],
        "On_or_After_2020_03_11": pred_df[pred_df["date"] >= EVENT_DATE],
    }
    rows = []
    for label, frame in period_frames.items():
        if frame.empty:
            continue
        period_metrics = evaluate_predictions(
            frame[TARGET],
            {"Hybrid_InvMSE": frame["Hybrid_InvMSE"].to_numpy()},
        )
        if not period_metrics.empty:
            row = period_metrics.iloc[0].to_dict()
            row["Period"] = label
            rows.append(row)

    pd.DataFrame(rows).to_csv(
        metrics_dir / "experiment_3_ilinet_pandemic_stress_test_pre_post_metrics.csv",
        index=False,
    )
    weekly.to_csv(
        metrics_dir / "experiment_3_ilinet_pandemic_stress_test_national_weekly.csv",
        index=False,
    )


# Calculation provenance:
#   load_ilinet -> common_io.load_ilinet (src/common_io.py)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
#   save_stress_test_outputs -> local function in this file
def main():
    df = load_ilinet(DATA_FILE, target=TARGET)
    result = run_experiment(
        df=df,
        experiment_name="experiment_3_ilinet_pandemic_stress_test",
        target=TARGET,
        dataset_type="ilinet",
        group_cols=["REGION TYPE", "REGION"],
        split_mode="explicit_epi",
        explicit_split={
            "train_start": 201640,
            "train_end": 201939,
            "test_start": 201940,
            "test_end": 202039,
        },
        lagged_covariates=LAGGED_COVARIATES,
    )
    save_stress_test_outputs(result)


if __name__ == "__main__":
    main()
