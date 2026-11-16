# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Ablation for Experiment 1: engineered temporal core features only."""
from pathlib import Path
import sys

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
LAGGED_COVARIATES = {column: [1] for column in NSSP_COMPONENT_COLUMNS}


# Calculation provenance:
#   load_nssp -> common_io.load_nssp (src/common_io.py)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
def main():
    df = load_nssp(DATA_FILE, target=TARGET)
    run_experiment(
        df=df,
        experiment_name="experiment_1_nssp_ablation_core_features_only",
        target=TARGET,
        dataset_type="nssp",
        group_cols=["geography", "county"],
        split_mode="chronological",
        lagged_covariates=LAGGED_COVARIATES,
        feature_cols_override=CORE_FEATURES,
        align_to_default_feature_cohort=True,
    )


if __name__ == "__main__":
    main()
