# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Experiment 2 — Contemporary ILINet same-framework validation."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "experiments"))

from common_config import DATA_RAW, ILINET_REPORTING_COLUMNS
from common_io import load_ilinet
from _run_template import run_experiment

DATA_FILE = DATA_RAW / "ILINet.csv"
TARGET = "%UNWEIGHTED ILI"
LAGGED_COVARIATES = {column: [1] for column in ILINET_REPORTING_COLUMNS}


# Calculation provenance:
#   load_ilinet -> common_io.load_ilinet (src/common_io.py)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
def main():
    df = load_ilinet(DATA_FILE, target=TARGET)
    run_experiment(
        df=df,
        experiment_name="experiment_2_ilinet_contemporary",
        target=TARGET,
        dataset_type="ilinet",
        group_cols=["REGION TYPE", "REGION"],
        split_mode="chronological",
        lagged_covariates=LAGGED_COVARIATES,
    )


if __name__ == "__main__":
    main()
