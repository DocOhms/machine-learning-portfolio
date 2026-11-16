# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Ablation for Experiment 1: remove K-Means and all cluster indicators."""
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
LAGGED_COVARIATES = {column: [1] for column in NSSP_COMPONENT_COLUMNS}


# Calculation provenance:
#   load_nssp -> common_io.load_nssp (src/common_io.py)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
def main():
    df = load_nssp(DATA_FILE, target=TARGET)
    run_experiment(
        df=df,
        experiment_name="experiment_1_nssp_ablation_no_clustering",
        target=TARGET,
        dataset_type="nssp",
        group_cols=["geography", "county"],
        split_mode="chronological",
        lagged_covariates=LAGGED_COVARIATES,
        use_clustering=False,
    )


if __name__ == "__main__":
    main()
