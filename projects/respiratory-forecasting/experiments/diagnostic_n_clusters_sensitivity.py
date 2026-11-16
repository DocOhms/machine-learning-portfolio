# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""K-Means cluster-count sensitivity study for Experiment 1."""
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
N_CLUSTERS_VALUES = [2, 3, 4, 6, 8]
LAGGED_COVARIATES = {column: [1] for column in NSSP_COMPONENT_COLUMNS}


def _model_row(metrics: pd.DataFrame, model: str) -> pd.Series:
    rows = metrics[metrics["Model"] == model]
    if rows.empty:
        raise ValueError(f"Metrics do not contain model: {model}")
    return rows.iloc[0]


# Calculation provenance:
#   _model_row -> local function in this file
#   load_nssp -> common_io.load_nssp (src/common_io.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(rows).sort_values -> pandas.DataFrame(rows).sort_values (third-party package)
#   run_experiment -> _run_template.run_experiment (experiments/_run_template.py)
def main():
    print("=" * 72)
    print("NSSP K-Means n_clusters Sensitivity Study")
    print("Selection criterion: lowest chronological-validation Hybrid RMSE")
    print("=" * 72)

    df = load_nssp(DATA_FILE, target=TARGET)
    rows = []

    for k in N_CLUSTERS_VALUES:
        experiment_name = f"experiment_1_nssp_nclusters_{k}"
        print(f"\n>>> Running n_clusters={k}")
        result = run_experiment(
            df=df.copy(),
            experiment_name=experiment_name,
            target=TARGET,
            dataset_type="nssp",
            group_cols=["geography", "county"],
            split_mode="chronological",
            lagged_covariates=LAGGED_COVARIATES,
            n_clusters=k,
            use_clustering=True,
            make_figures=False,
        )

        validation_hybrid = _model_row(result["validation_metrics"], "Hybrid_InvMSE")
        test_hybrid = _model_row(result["metrics"], "Hybrid_InvMSE")
        metadata = result["metadata"].iloc[0]
        rows.append(
            {
                "Requested_N_Clusters": k,
                "Effective_N_Clusters": int(metadata["Final_Effective_N_Clusters"]),
                "Full_Train_Silhouette": metadata["Full_Train_Silhouette"],
                "Validation_MAE": validation_hybrid["MAE"],
                "Validation_RMSE": validation_hybrid["RMSE"],
                "Validation_R2": validation_hybrid["R2"],
                "Test_MAE": test_hybrid["MAE"],
                "Test_RMSE": test_hybrid["RMSE"],
                "Test_R2": test_hybrid["R2"],
                "Test_SMAPE": test_hybrid["SMAPE"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("Requested_N_Clusters")
    best = summary.loc[summary["Validation_RMSE"].idxmin()]
    summary["Selected_By_Validation"] = (
        summary["Requested_N_Clusters"] == best["Requested_N_Clusters"]
    )

    output_path = PROJECT_ROOT / "results" / "nssp_nclusters_sensitivity_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("\n" + summary.to_string(index=False))
    print(
        f"\nValidation-selected cluster count: {int(best['Requested_N_Clusters'])} "
        f"(Validation RMSE={best['Validation_RMSE']:.4f})"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
