# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Exact-row ARIMA extension for locked ILINet Experiments 2 and 3.

Nonseasonal ARIMA orders are selected only from the previously completed
chronological-validation grid. Each selected model is then fitted on the full
pre-test history and used for rolling one-week-ahead predictions on the exact
locked final-test rows. Outputs are isolated from the approved manuscript run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarima_pilot import (
    EXPERIMENTS,
    OUTPUT_ROOT,
    PILOT_ROOT,
    evaluate_candidate,
    file_sha256,
    load_raw,
    locked_paths,
    metrics,
    reproduce_cohorts,
)


EXTENSION_ROOT = OUTPUT_ROOT / "arima_extension_exp2_exp3"
EXPECTED_SELECTIONS = {"exp2": "ARIMA_011", "exp3": "ARIMA_110"}
MODEL_LABELS = {
    "Persistence": "Persistence",
    "MovingAverage3": "Moving average (3 weeks)",
    "SeasonalNaive": "Seasonal naive (52 weeks)",
    "Hybrid_InvMSE": "Hybrid inverse-MSE ensemble",
}


# Calculation provenance:
#   pd.read_csv -> pandas.read_csv (third-party package)
def select_nonseasonal_candidate(exp_key: str, config: dict) -> tuple[dict, pd.DataFrame]:
    validation_path = OUTPUT_ROOT / exp_key / "validation_candidate_summary.csv"
    validation = pd.read_csv(validation_path)
    eligible = validation[
        (~validation["Seasonal"].astype(bool))
        & (validation["Coverage"] >= float(config["minimum_candidate_coverage"]))
    ].copy()
    if eligible.empty:
        raise RuntimeError(f"No nonseasonal candidate met the coverage gate for {exp_key}")
    selected_row = eligible.sort_values(["RMSE", "MAE", "Candidate"]).iloc[0]
    selected_name = str(selected_row["Candidate"])
    if selected_name != EXPECTED_SELECTIONS[exp_key]:
        raise RuntimeError(
            f"Selection changed for {exp_key}: expected {EXPECTED_SELECTIONS[exp_key]}, got {selected_name}"
        )
    candidate = next(item for item in config["candidate_models"] if item["name"] == selected_name)
    evidence = eligible.sort_values(["RMSE", "MAE", "Candidate"]).copy()
    evidence.insert(0, "Selected", evidence["Candidate"].eq(selected_name))
    return candidate, evidence


# Calculation provenance:
#   Path -> pathlib.Path (Python standard library)
#   file_sha256 -> sarima_pilot.file_sha256 (benchmarks/sarima/sarima_pilot.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
def verify_locked_input_hashes(exp_key: str) -> pd.DataFrame:
    manifest = pd.read_csv(OUTPUT_ROOT / "input_manifest_sha256.csv")
    expected = manifest[manifest["experiment"].eq(exp_key)].copy()
    rows = []
    for item in expected.itertuples(index=False):
        path = Path(item.path)
        current_hash = file_sha256(path)
        current_bytes = path.stat().st_size
        rows.append(
            {
                "Experiment": exp_key,
                "Role": item.role,
                "Path": str(path),
                "Expected_SHA256": item.sha256,
                "Current_SHA256": current_hash,
                "Expected_Bytes": int(item.bytes),
                "Current_Bytes": current_bytes,
                "Hash_Match": current_hash == item.sha256,
                "Size_Match": current_bytes == int(item.bytes),
            }
        )
    audit = pd.DataFrame(rows)
    if not audit[["Hash_Match", "Size_Match"]].all().all():
        raise RuntimeError(f"Locked input hash mismatch for {exp_key}\n{audit.to_string(index=False)}")
    return audit


# Calculation provenance:
#   locked_paths -> sarima_pilot.locked_paths (benchmarks/sarima/sarima_pilot.py)
#   metrics -> sarima_pilot.metrics (benchmarks/sarima/sarima_pilot.py)
#   np.allclose -> numpy.allclose (third-party package)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.to_numeric -> pandas.to_numeric (third-party package)
#   pd.to_numeric(locked['epi_order'], errors='raise').astype -> pandas.to_numeric(locked['epi_order'], errors='raise').astype (third-party package)
#   pd.to_numeric(sarima['epi_order'], errors='raise').astype -> pandas.to_numeric(sarima['epi_order'], errors='raise').astype (third-party package)
def load_locked_comparison(exp_key: str, exp, arima_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked_path, _ = locked_paths(exp)
    locked = pd.read_csv(locked_path, low_memory=False)
    for column in exp.groups:
        locked[column] = locked[column].astype(str)
    locked["epi_order"] = pd.to_numeric(locked["epi_order"], errors="raise").astype(int)

    sarima_path = OUTPUT_ROOT / exp_key / "final_test_predictions.csv"
    sarima = pd.read_csv(sarima_path, low_memory=False)
    for column in exp.groups:
        sarima[column] = sarima[column].astype(str)
    sarima["epi_order"] = pd.to_numeric(sarima["epi_order"], errors="raise").astype(int)

    keys = [*exp.groups, "epi_order"]
    if locked.duplicated(keys).any() or sarima.duplicated(keys).any() or arima_predictions.duplicated(keys).any():
        raise RuntimeError(f"Duplicate exact-row keys detected for {exp_key}")
    locked_keys = set(locked[keys].itertuples(index=False, name=None))
    sarima_keys = set(sarima[keys].itertuples(index=False, name=None))
    arima_keys = set(arima_predictions[keys].itertuples(index=False, name=None))
    if not (locked_keys == sarima_keys == arima_keys):
        raise RuntimeError(
            f"Exact-row key mismatch for {exp_key}: locked={len(locked_keys)}, "
            f"SARIMA={len(sarima_keys)}, ARIMA={len(arima_keys)}"
        )

    comparison = locked.merge(
        sarima[keys + [exp.target, "prediction"]].rename(
            columns={exp.target: "SARIMA_Actual", "prediction": "Selected_SARIMA"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    ).merge(
        arima_predictions[keys + [exp.target, "prediction"]].rename(
            columns={exp.target: "ARIMA_Actual", "prediction": "Selected_ARIMA"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    actual = comparison[exp.target].to_numpy(float)
    if not np.allclose(actual, comparison["SARIMA_Actual"].to_numpy(float), equal_nan=True):
        raise RuntimeError(f"Locked and SARIMA actual values differ for {exp_key}")
    if not np.allclose(actual, comparison["ARIMA_Actual"].to_numpy(float), equal_nan=True):
        raise RuntimeError(f"Locked and ARIMA actual values differ for {exp_key}")

    model_columns = [*MODEL_LABELS, "Selected_ARIMA", "Selected_SARIMA"]
    metric_rows = []
    for column in model_columns:
        result = metrics(actual, comparison[column].to_numpy(float))
        metric_rows.append(
            {
                "Experiment": exp_key,
                "Model": MODEL_LABELS.get(column, column.replace("Selected_", "Selected ")),
                **result,
                "Coverage": result["N"] / len(comparison),
            }
        )
    row_audit = pd.DataFrame(
        [
            {
                "Experiment": exp_key,
                "Locked_Rows": len(locked),
                "ARIMA_Rows": len(arima_predictions),
                "SARIMA_Rows": len(sarima),
                "Exact_Key_Set_Match": True,
                "Actual_Value_Match": True,
            }
        ]
    )
    return pd.DataFrame(metric_rows), row_audit


# Calculation provenance:
#   evaluate_candidate -> sarima_pilot.evaluate_candidate (benchmarks/sarima/sarima_pilot.py)
#   json.dumps -> json.dumps (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   load_locked_comparison -> local function in this file
#   load_raw -> sarima_pilot.load_raw (benchmarks/sarima/sarima_pilot.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame([{**summary, **{f'Cohort_{k}': v for k, v in cohort_audit.items()}}]).to_csv -> pandas.DataFrame([{**summary, **{f'Cohort_{k}': v for k, v in cohort_audit.items()}}]).to_csv (third-party package)
#   pd.concat -> pandas.concat (third-party package)
#   pd.concat(all_hash_audits, ignore_index=True).to_csv -> pandas.concat(all_hash_audits, ignore_index=True).to_csv (third-party package)
#   pd.concat(all_row_audits, ignore_index=True).to_csv -> pandas.concat(all_row_audits, ignore_index=True).to_csv (third-party package)
#   reproduce_cohorts -> sarima_pilot.reproduce_cohorts (benchmarks/sarima/sarima_pilot.py)
#   select_nonseasonal_candidate -> local function in this file
#   verify_locked_input_hashes -> local function in this file
def main() -> None:
    EXTENSION_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((PILOT_ROOT / "pilot_config.json").read_text(encoding="utf-8"))
    all_metrics = []
    all_hash_audits = []
    all_row_audits = []
    decision_rows = []

    for exp_key in ("exp2", "exp3"):
        exp = EXPERIMENTS[exp_key]
        exp_out = EXTENSION_ROOT / exp_key
        exp_out.mkdir(parents=True, exist_ok=True)

        hash_audit = verify_locked_input_hashes(exp_key)
        all_hash_audits.append(hash_audit)
        candidate, selection_evidence = select_nonseasonal_candidate(exp_key, config)
        selection_evidence.to_csv(exp_out / "validation_nonseasonal_selection.csv", index=False)
        (exp_out / "selected_candidate.json").write_text(
            json.dumps(
                {
                    **candidate,
                    "selection_rule": "lowest pooled chronological-validation RMSE, then MAE; coverage >= 0.95",
                    "final_test_used_for_selection": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        raw = load_raw(exp)
        _, test, cohort_audit = reproduce_cohorts(raw, exp)
        print(f"[{exp_key}] final test {candidate['name']} on {len(test)} exact locked rows", flush=True)
        summary, predictions, diagnostics = evaluate_candidate(
            raw,
            test,
            exp,
            exp.test_cutoff,
            candidate,
            int(config["maxiter"]),
        )
        summary["Selected_On"] = "chronological_validation_RMSE"
        summary["Final_Test_Used_For_Selection"] = False
        predictions.to_csv(exp_out / "final_test_predictions.csv", index=False)
        diagnostics.to_csv(exp_out / "final_test_group_diagnostics.csv", index=False)
        pd.DataFrame([{**summary, **{f"Cohort_{k}": v for k, v in cohort_audit.items()}}]).to_csv(
            exp_out / "final_test_summary.csv", index=False
        )

        metric_frame, row_audit = load_locked_comparison(exp_key, exp, predictions)
        metric_frame.to_csv(exp_out / "same_row_model_comparison.csv", index=False)
        row_audit.to_csv(exp_out / "exact_row_audit.csv", index=False)
        all_metrics.append(metric_frame)
        all_row_audits.append(row_audit)

        indexed = metric_frame.set_index("Model")
        arima = indexed.loc["Selected ARIMA"]
        sarima = indexed.loc["Selected SARIMA"]
        hybrid = indexed.loc["Hybrid inverse-MSE ensemble"]
        decision_rows.append(
            {
                "Experiment": exp_key,
                "Selected_ARIMA": candidate["name"],
                "ARIMA_RMSE": arima["RMSE"],
                "ARIMA_MAE": arima["MAE"],
                "ARIMA_R2": arima["R2"],
                "SARIMA_RMSE": sarima["RMSE"],
                "SARIMA_MAE": sarima["MAE"],
                "SARIMA_R2": sarima["R2"],
                "Hybrid_RMSE": hybrid["RMSE"],
                "Hybrid_MAE": hybrid["MAE"],
                "Hybrid_R2": hybrid["R2"],
                "ARIMA_vs_SARIMA_RMSE_Delta": arima["RMSE"] - sarima["RMSE"],
                "Hybrid_vs_ARIMA_RMSE_Reduction": (arima["RMSE"] - hybrid["RMSE"]) / arima["RMSE"],
                "Convergence_Rate": summary["Convergence_Rate"],
                "Coverage": summary["Coverage"],
                "Fit_Seconds": summary["Fit_Seconds"],
            }
        )

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(EXTENSION_ROOT / "COMBINED_SAME_ROW_MODEL_COMPARISON.csv", index=False)
    pd.concat(all_hash_audits, ignore_index=True).to_csv(EXTENSION_ROOT / "LOCKED_INPUT_HASH_AUDIT.csv", index=False)
    pd.concat(all_row_audits, ignore_index=True).to_csv(EXTENSION_ROOT / "EXACT_ROW_AUDIT.csv", index=False)
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(EXTENSION_ROOT / "ARIMA_EXTENSION_DECISION_DATA.csv", index=False)

    lines = [
        "# ILINet ARIMA exact-row extension",
        "",
        "The nonseasonal order for each experiment was selected only from the previously completed chronological-validation grid. The locked final test was not used for order selection. Parameters were fitted on the complete pre-test history, and test predictions were rolling one-week-ahead forecasts on the exact rows used by the approved framework and SARIMA comparisons.",
        "",
        "| Experiment | Selected ARIMA | ARIMA RMSE | ARIMA MAE | ARIMA R2 | SARIMA RMSE | Hybrid RMSE | Coverage | Convergence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| {row.Experiment} | {row.Selected_ARIMA} | {row.ARIMA_RMSE:.6f} | {row.ARIMA_MAE:.6f} | "
            f"{row.ARIMA_R2:.6f} | {row.SARIMA_RMSE:.6f} | {row.Hybrid_RMSE:.6f} | "
            f"{row.Coverage:.1%} | {row.Convergence_Rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Retention rule",
            "",
            "Retain ARIMA as the primary nonseasonal classical reference and retain SARIMA as the annual-seasonality benchmark. Keep both outside the hybrid ensemble. Interpret their differences by experiment; do not claim universal superiority from these three datasets.",
            "",
            "## Audit outcome",
            "",
            "All locked input hashes, exact-row key sets, row counts, and target values must match for this report to be produced. Model failures and optimizer convergence are recorded in the experiment-level diagnostics rather than silently omitted.",
        ]
    )
    (EXTENSION_ROOT / "ARIMA_EXTENSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(decisions.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
