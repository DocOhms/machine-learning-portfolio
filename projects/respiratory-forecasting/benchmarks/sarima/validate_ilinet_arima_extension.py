# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Independent audit for the Experiment 2/3 ARIMA extension outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sarima_pilot import EXPERIMENTS, OUTPUT_ROOT, metrics


ROOT = OUTPUT_ROOT / "arima_extension_exp2_exp3"
EXPECTED = {
    "exp2": {"candidate": "ARIMA_011", "rows": 1774, "groups": 54},
    "exp3": {"candidate": "ARIMA_110", "rows": 2808, "groups": 54},
}


# Calculation provenance:
#   np.isclose -> numpy.isclose (third-party package)
def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance))


# Calculation provenance:
#   close -> local function in this file
#   metrics -> sarima_pilot.metrics (benchmarks/sarima/sarima_pilot.py)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.read_csv(ROOT / 'ARIMA_EXTENSION_DECISION_DATA.csv').set_index -> pandas.read_csv(ROOT / 'ARIMA_EXTENSION_DECISION_DATA.csv').set_index (third-party package)
#   pd.read_csv(ROOT / 'EXACT_ROW_AUDIT.csv').set_index -> pandas.read_csv(ROOT / 'EXACT_ROW_AUDIT.csv').set_index (third-party package)
def main() -> None:
    checks: list[str] = []
    hash_audit = pd.read_csv(ROOT / "LOCKED_INPUT_HASH_AUDIT.csv")
    assert hash_audit["Hash_Match"].astype(bool).all()
    assert hash_audit["Size_Match"].astype(bool).all()
    checks.append("PASS: all locked raw-data, prediction, and metadata hashes and sizes match")

    exact = pd.read_csv(ROOT / "EXACT_ROW_AUDIT.csv").set_index("Experiment")
    combined = pd.read_csv(ROOT / "COMBINED_SAME_ROW_MODEL_COMPARISON.csv")
    decisions = pd.read_csv(ROOT / "ARIMA_EXTENSION_DECISION_DATA.csv").set_index("Experiment")

    for exp_key, expected in EXPECTED.items():
        exp = EXPERIMENTS[exp_key]
        exp_root = ROOT / exp_key
        selection = pd.read_csv(exp_root / "validation_nonseasonal_selection.csv")
        selected = selection.loc[selection["Selected"].astype(bool), "Candidate"].tolist()
        assert selected == [expected["candidate"]]
        ranked = selection.sort_values(["RMSE", "MAE", "Candidate"]).iloc[0]["Candidate"]
        assert ranked == expected["candidate"]
        checks.append(f"PASS: {exp_key} selection is validation-only {expected['candidate']}")

        predictions = pd.read_csv(exp_root / "final_test_predictions.csv")
        diagnostics = pd.read_csv(exp_root / "final_test_group_diagnostics.csv")
        summary = pd.read_csv(exp_root / "final_test_summary.csv").iloc[0]
        keys = [*exp.groups, "epi_order"]
        assert len(predictions) == expected["rows"]
        assert not predictions.duplicated(keys).any()
        assert predictions["prediction"].notna().all()
        assert len(diagnostics) == expected["groups"]
        assert diagnostics["converged"].astype(bool).all()
        assert not diagnostics["error"].fillna("").astype(str).str.len().gt(0).any()
        assert int(summary["N"]) == expected["rows"]
        assert close(summary["Coverage"], 1.0)
        assert close(summary["Convergence_Rate"], 1.0)
        assert not bool(summary["Final_Test_Used_For_Selection"])

        recalculated = metrics(
            predictions[exp.target].to_numpy(float), predictions["prediction"].to_numpy(float)
        )
        for metric in ("RMSE", "MAE", "R2"):
            assert close(recalculated[metric], summary[metric])
        checks.append(
            f"PASS: {exp_key} has {expected['rows']} unique exact rows, full predictions, "
            f"54/54 convergence, no errors, and independently reproduced metrics"
        )

        audit_row = exact.loc[exp_key]
        assert int(audit_row["Locked_Rows"]) == expected["rows"]
        assert int(audit_row["ARIMA_Rows"]) == expected["rows"]
        assert int(audit_row["SARIMA_Rows"]) == expected["rows"]
        assert bool(audit_row["Exact_Key_Set_Match"])
        assert bool(audit_row["Actual_Value_Match"])
        checks.append(f"PASS: {exp_key} locked, ARIMA, and SARIMA keys and actual values match exactly")

        comparison = combined[combined["Experiment"].eq(exp_key)].set_index("Model")
        for model in (
            "Persistence",
            "Moving average (3 weeks)",
            "Seasonal naive (52 weeks)",
            "Hybrid inverse-MSE ensemble",
            "Selected ARIMA",
            "Selected SARIMA",
        ):
            assert int(comparison.loc[model, "N"]) == expected["rows"]
            assert close(comparison.loc[model, "Coverage"], 1.0)
        assert close(decisions.loc[exp_key, "ARIMA_RMSE"], comparison.loc["Selected ARIMA", "RMSE"])
        assert close(decisions.loc[exp_key, "SARIMA_RMSE"], comparison.loc["Selected SARIMA", "RMSE"])
        assert close(
            decisions.loc[exp_key, "Hybrid_RMSE"], comparison.loc["Hybrid inverse-MSE ensemble", "RMSE"]
        )
        checks.append(f"PASS: {exp_key} same-row comparison and decision data agree")

    exp2 = decisions.loc["exp2"]
    exp3 = decisions.loc["exp3"]
    assert exp2["Hybrid_RMSE"] < exp2["SARIMA_RMSE"] < exp2["ARIMA_RMSE"]
    assert exp3["Hybrid_RMSE"] < exp3["SARIMA_RMSE"] < exp3["ARIMA_RMSE"]
    checks.append("RESULT: hybrid < annual SARIMA < nonseasonal ARIMA RMSE in both ILINet experiments")
    checks.append("RESULT: ILINet ARIMA EXTENSION AUDIT PASSED")

    report = "# ILINet ARIMA extension audit\n\n" + "\n".join(f"- {item}" for item in checks) + "\n"
    (ROOT / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
