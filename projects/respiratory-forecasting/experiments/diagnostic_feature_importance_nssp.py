# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Feature-importance diagnostic for the corrected NSSP primary pipeline."""
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
LAGGED_COVARIATES = {column: [1] for column in NSSP_COMPONENT_COLUMNS}


# Calculation provenance:
#   load_nssp -> common_io.load_nssp (src/common_io.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(rows).to_csv -> pandas.DataFrame(rows).to_csv (third-party package)
#   pd.Series -> pandas.Series (third-party package)
#   pd.Series(model.feature_importances_, index=result['final_features'], name='Importance').sort_values -> pandas.Series(model.feature_importances_, index=result['final_features'], name='Importance').sort_values (third-party package)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
def main():
    df = load_nssp(DATA_FILE, target=TARGET)
    result = run_experiment(
        df=df,
        experiment_name="diagnostic_feature_importance_nssp",
        target=TARGET,
        dataset_type="nssp",
        group_cols=["geography", "county"],
        split_mode="chronological",
        lagged_covariates=LAGGED_COVARIATES,
        make_figures=False,
    )

    rows = []
    for model_name in ["RandomForest", "GradientBoosting"]:
        model = result["models"].get(model_name)
        if model is None or not hasattr(model, "feature_importances_"):
            continue
        importance = pd.Series(
            model.feature_importances_,
            index=result["final_features"],
            name="Importance",
        ).sort_values(ascending=False)
        print(f"\n{model_name} feature importance:")
        print(importance.head(20).to_string())
        for feature, value in importance.items():
            rows.append(
                {"Model": model_name, "Feature": feature, "Importance": value}
            )

    output_path = (
        PROJECT_ROOT
        / "results"
        / "diagnostic_feature_importance_nssp"
        / "metrics"
        / "diagnostic_feature_importance_nssp.csv"
    )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
