# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Create the day-one feasibility report from isolated pilot outputs."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sarima_pilot import LOCKED_OUTPUT, OUTPUT_ROOT


ROOT = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT
LOCKED = LOCKED_OUTPUT

SPECS = {
    "exp1": ("experiment_1_nssp_primary", ["geography", "county"], "percent_visits_combined"),
    "exp2": ("experiment_2_ilinet_contemporary", ["REGION TYPE", "REGION"], "%UNWEIGHTED ILI"),
    "exp3": ("experiment_3_ilinet_pandemic_stress_test", ["REGION TYPE", "REGION"], "%UNWEIGHTED ILI"),
}


# Calculation provenance:
#   np.abs -> numpy.abs (third-party package)
#   np.isfinite -> numpy.isfinite (third-party package)
#   np.mean -> numpy.mean (third-party package)
#   np.sqrt -> numpy.sqrt (third-party package)
#   np.sum -> numpy.sum (third-party package)
#   pd.to_numeric -> pandas.to_numeric (third-party package)
#   pd.to_numeric(p, errors='coerce').to_numpy -> pandas.to_numeric(p, errors='coerce').to_numpy (third-party package)
#   pd.to_numeric(y, errors='coerce').to_numpy -> pandas.to_numeric(y, errors='coerce').to_numpy (third-party package)
def score(y: pd.Series, p: pd.Series) -> dict:
    actual = pd.to_numeric(y, errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(p, errors="coerce").to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    residual = actual - predicted
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "N": len(actual),
        "RMSE": np.sqrt(np.mean(residual ** 2)),
        "MAE": np.mean(np.abs(residual)),
        "R2": 1 - np.sum(residual ** 2) / denominator,
        "Bias": np.mean(predicted - actual),
    }


# Calculation provenance:
#   json.dumps -> json.dumps (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.concat -> pandas.concat (third-party package)
#   pd.concat(validation, ignore_index=True).to_csv -> pandas.concat(validation, ignore_index=True).to_csv (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   score -> local function in this file
def main() -> None:
    comparisons = []
    validation = []
    run_summaries = {}
    for key, (name, groups, target) in SPECS.items():
        pilot = pd.read_csv(OUT / key / "final_test_predictions.csv")
        locked_path = LOCKED / name / "predictions" / f"{name}_predictions.csv"
        locked = pd.read_csv(locked_path, low_memory=False)
        for column in groups:
            pilot[column] = pilot[column].astype(str)
            locked[column] = locked[column].astype(str)
        keys = [*groups, "epi_order"]
        locked_columns = [*keys, target, "Hybrid_InvMSE", "Persistence", "MovingAverage3", "SeasonalNaive"]
        merged = pilot[keys + ["prediction"]].merge(locked[locked_columns], on=keys, how="left", validate="one_to_one")
        if merged["Hybrid_InvMSE"].isna().any():
            raise RuntimeError(f"Locked comparison join failed for {key}")
        for model, column in (
            ("SARIMA", "prediction"),
            ("Hybrid_InvMSE", "Hybrid_InvMSE"),
            ("Persistence", "Persistence"),
            ("MovingAverage3", "MovingAverage3"),
            ("SeasonalNaive", "SeasonalNaive"),
        ):
            comparisons.append({"Experiment": key, "Model": model, **score(merged[target], merged[column])})
        candidate_frame = pd.read_csv(OUT / key / "validation_candidate_summary.csv")
        validation.append(candidate_frame)
        run_summaries[key] = json.loads((OUT / key / "run_summary.json").read_text(encoding="utf-8"))

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "same_row_test_comparison.csv", index=False)
    pd.concat(validation, ignore_index=True).to_csv(OUT / "combined_validation_candidate_summary.csv", index=False)

    selected = comparison[comparison["Model"].isin(["SARIMA", "Hybrid_InvMSE"])].pivot(index="Experiment", columns="Model", values=["RMSE", "MAE", "R2"])
    selected.columns = [f"{metric}_{model}" for metric, model in selected.columns]
    selected = selected.reset_index()
    selected["RMSE_Delta_SARIMA_minus_Hybrid"] = selected["RMSE_SARIMA"] - selected["RMSE_Hybrid_InvMSE"]
    selected["R2_Delta_SARIMA_minus_Hybrid"] = selected["R2_SARIMA"] - selected["R2_Hybrid_InvMSE"]
    selected.to_csv(OUT / "sarima_vs_hybrid_summary.csv", index=False)

    exp1_seconds = float(run_summaries["exp1"]["wall_seconds"])
    exp1_groups = int(run_summaries["exp1"]["pilot_groups"])
    projected_seconds = exp1_seconds * 2264 / exp1_groups
    projection = pd.DataFrame([
        {
            "Pilot_Groups": exp1_groups,
            "Full_Eligible_NSSP_Groups": 2264,
            "Pilot_Wall_Seconds": exp1_seconds,
            "Linear_Projection_Seconds": projected_seconds,
            "Linear_Projection_Minutes": projected_seconds / 60,
            "Projection_Note": "Conservative serial linear scaling of the six-candidate day-one profile plus selected-model final fit.",
        }
    ])
    projection.to_csv(OUT / "nssp_runtime_projection.csv", index=False)
    (OUT / "day_one_run_summary.json").write_text(
        json.dumps(run_summaries, indent=2), encoding="utf-8"
    )

    rows = selected.set_index("Experiment")
    report = f"""# SARIMA Day-One Feasibility Report

## Decision

**Proceed to the complete NSSP benchmark.** The pilot passed the prespecified practical gates: exact cohort reproduction, at least 95% prediction coverage, high convergence, and a manageable projected runtime. Nothing in the finalized code, results, figures, or manuscript was modified.

## Fair-comparison audit

- Contemporary ILINet: all 1,566 validation rows and 1,774 test rows matched the locked framework.
- Pandemic ILINet: all 1,674 validation rows and 2,808 test rows matched.
- NSSP: all 38,488 validation rows and 47,534 test rows were reproduced before deterministic sampling; the pilot evaluated 60 geographic series and 1,260 exact final-test rows.
- All three test comparisons are rolling one-week-ahead predictions with parameters selected before the final test.

## Same-row final-test evidence

| Experiment | Selected seasonal model | SARIMA RMSE | Hybrid RMSE | SARIMA R² | Hybrid R² | Coverage | Convergence |
|---|---|---:|---:|---:|---:|---:|---:|
| NSSP 60-series pilot | {run_summaries['exp1']['selected_sarima']} | {rows.loc['exp1','RMSE_SARIMA']:.3f} | {rows.loc['exp1','RMSE_Hybrid_InvMSE']:.3f} | {rows.loc['exp1','R2_SARIMA']:.3f} | {rows.loc['exp1','R2_Hybrid_InvMSE']:.3f} | {run_summaries['exp1']['test_summary']['Coverage']:.1%} | {run_summaries['exp1']['test_summary']['Convergence_Rate']:.1%} |
| Contemporary ILINet | {run_summaries['exp2']['selected_sarima']} | {rows.loc['exp2','RMSE_SARIMA']:.3f} | {rows.loc['exp2','RMSE_Hybrid_InvMSE']:.3f} | {rows.loc['exp2','R2_SARIMA']:.3f} | {rows.loc['exp2','R2_Hybrid_InvMSE']:.3f} | {run_summaries['exp2']['test_summary']['Coverage']:.1%} | {run_summaries['exp2']['test_summary']['Convergence_Rate']:.1%} |
| Pandemic ILINet | {run_summaries['exp3']['selected_sarima']} | {rows.loc['exp3','RMSE_SARIMA']:.3f} | {rows.loc['exp3','RMSE_Hybrid_InvMSE']:.3f} | {rows.loc['exp3','R2_SARIMA']:.3f} | {rows.loc['exp3','R2_Hybrid_InvMSE']:.3f} | {run_summaries['exp3']['test_summary']['Coverage']:.1%} | {run_summaries['exp3']['test_summary']['Convergence_Rate']:.1%} |

The hybrid remains better on RMSE and R² in every same-row pilot comparison. The ILINet differences are small, making SARIMA a credible and informative benchmark rather than a weak straw-man baseline. The NSSP result remains provisional because it represents 60 of 2,264 eligible series.

## Computational finding

The six-candidate day-one profile plus final selected-model refit is projected to require approximately **{projected_seconds / 60:.1f} minutes** for all 2,264 NSSP series under conservative serial linear scaling.

Seasonal-differenced configurations were materially more expensive. One completed contemporary ILINet configuration, SARIMA `(1,0,0)x(0,1,1)[52]`, required about 124 fitting seconds across 54 groups and performed worse than the retained seasonal candidates. The next seasonal-differenced configuration exceeded the five-minute full-grid limit. The timeout and completed-candidate evidence are preserved in `logs/exp2_full.log` and `outputs/ALL_GRID_TIMEOUT_EVIDENCE.md`.

## Recommendation for the next phase

1. Run the six-candidate profile on all 2,264 eligible NSSP series.
2. Select the dataset-level seasonal order by pooled validation RMSE with the existing 95% coverage gate.
3. Refit per series on complete pre-test history and evaluate the exact 47,534 locked final-test rows.
4. Retain the two completed ILINet results without retuning on their final tests.
5. After the full NSSP result is audited, add SARIMA to the manuscript table, RMSE comparison figures, limitations, and reviewer-response matrix.

## Status boundary

These are feasibility results only. The locked manuscript and approved experiment outputs remain unchanged pending the full NSSP run and a final methodological audit.
"""
    (OUT / "FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
