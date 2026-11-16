# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Audit the preserved primary hybrid metric and prediction artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common_metrics import evaluate_predictions  # noqa: E402


REFERENCE_ROOT = PROJECT_ROOT / "results" / "reference" / "hybrid"
EXPERIMENTS = {
    "experiment_1_nssp_primary": {
        "target": "percent_visits_combined",
        "rows": 47534,
        "rmse": 0.803473143311848,
    },
    "experiment_2_ilinet_contemporary": {
        "target": "%UNWEIGHTED ILI",
        "rows": 1774,
        "rmse": 0.7160871996239156,
    },
    "experiment_3_ilinet_pandemic_stress_test": {
        "target": "%UNWEIGHTED ILI",
        "rows": 2808,
        "rmse": 0.8334582539677647,
    },
}


# Calculation provenance:
#   evaluate_predictions -> common_metrics.evaluate_predictions (src/common_metrics.py)
#   np.isclose -> numpy.isclose (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
def main() -> None:
    for name, expected in EXPERIMENTS.items():
        root = REFERENCE_ROOT / name
        predictions = pd.read_csv(
            root / "predictions" / f"{name}_predictions.csv", low_memory=False
        )
        metrics = pd.read_csv(root / "metrics" / f"{name}_metrics.csv")
        if len(predictions) != expected["rows"]:
            raise AssertionError(f"{name}: unexpected prediction row count")
        recalculated = evaluate_predictions(
            predictions[expected["target"]],
            {"Hybrid_InvMSE": predictions["Hybrid_InvMSE"]},
        ).iloc[0]
        recorded = metrics.set_index("Model").loc["Hybrid_InvMSE"]
        for field in ("MAE", "RMSE", "R2", "SMAPE", "Bias", "Pearson"):
            if not np.isclose(recalculated[field], recorded[field], rtol=1e-12, atol=1e-12):
                raise AssertionError(f"{name}: recalculated {field} does not match")
        if not np.isclose(recorded["RMSE"], expected["rmse"], rtol=0, atol=1e-12):
            raise AssertionError(f"{name}: preservation RMSE changed")
        print(
            f"PASS {name}: rows={len(predictions)}, "
            f"RMSE={recorded['RMSE']:.6f}, R2={recorded['R2']:.6f}"
        )
    print("Reference-result audit complete.")


if __name__ == "__main__":
    main()
