# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""NSSP component-feature leakage diagnostic.

This diagnostic compares, on the exact same observations and temporal splits:
1. Strict forecasting features: lagged component percentages only.
2. Core temporal features only.
3. Diagnostic-only same-week component percentages.

The same-week configuration is intentionally leakage-prone and must not be used
as the primary forecasting experiment. It is included only to quantify the
performance inflation caused by contemporaneous component outcomes.
"""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "experiments"))

from common_config import DATA_RAW, NSSP_COMPONENT_COLUMNS
from common_io import load_nssp
from _run_template import run_experiment

DATA_FILE = DATA_RAW / "NSSP.csv"
TARGET = "percent_visits_combined"
CORE_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "week_sin",
    "week_cos",
]
LAGGED_COMPONENT_FEATURES = [
    f"{column}_lag_1" for column in NSSP_COMPONENT_COLUMNS
]
LAGGED_COVARIATES = {column: [1] for column in NSSP_COMPONENT_COLUMNS}


def _hybrid_row(result: dict) -> pd.Series:
    frame = result["metrics"]
    rows = frame[frame["Model"] == "Hybrid_InvMSE"]
    if rows.empty:
        raise ValueError("Hybrid metrics were not produced.")
    return rows.iloc[0]


# Calculation provenance:
#   _hybrid_row -> local function in this file
#   load_nssp -> common_io.load_nssp (src/common_io.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
def main():
    df = load_nssp(DATA_FILE, target=TARGET)
    union_cohort = CORE_FEATURES + LAGGED_COMPONENT_FEATURES + NSSP_COMPONENT_COLUMNS

    configurations = {
        "Strict_Lagged_Components": CORE_FEATURES + LAGGED_COMPONENT_FEATURES,
        "Core_Temporal_Only": CORE_FEATURES,
        "Diagnostic_Current_Week_Components": CORE_FEATURES + NSSP_COMPONENT_COLUMNS,
    }

    rows = []
    for label, features in configurations.items():
        print(f"\n>>> Running {label}")
        result = run_experiment(
            df=df.copy(),
            experiment_name=f"diagnostic_nssp_{label.lower()}",
            target=TARGET,
            dataset_type="nssp",
            group_cols=["geography", "county"],
            split_mode="chronological",
            lagged_covariates=LAGGED_COVARIATES,
            feature_cols_override=features,
            cohort_feature_cols_override=union_cohort,
            make_figures=False,
        )
        row = _hybrid_row(result).to_dict()
        row["Configuration"] = label
        rows.append(row)

    summary = pd.DataFrame(rows)
    current_week_r2 = summary.loc[
        summary["Configuration"] == "Diagnostic_Current_Week_Components", "R2"
    ].iloc[0]
    strict_r2 = summary.loc[
        summary["Configuration"] == "Strict_Lagged_Components", "R2"
    ].iloc[0]
    summary["R2_Difference_From_Strict"] = summary["R2"] - strict_r2

    output_path = PROJECT_ROOT / "results" / "nssp_component_leakage_diagnostic.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("\n" + summary.to_string(index=False))
    print(
        "\nSame-week component R² inflation relative to strict lagged components: "
        f"{current_week_r2 - strict_r2:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
