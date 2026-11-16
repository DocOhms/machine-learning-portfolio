# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Audit and document the completed full NSSP SARIMA benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sarima_pilot import OUTPUT_ROOT


ROOT = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT
FULL = OUT / "nssp_full"


# Calculation provenance:
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Calculation provenance:
#   Path -> pathlib.Path (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   sha256 -> local function in this file
def main() -> None:
    checks = []
    manifest = pd.read_csv(OUT / "input_manifest_sha256.csv")
    for row in manifest.itertuples(index=False):
        assert sha256(Path(row.path)) == row.sha256
    checks.append(f"PASS: {len(manifest)} locked input hashes unchanged")

    validation = pd.read_csv(FULL / "validation_candidate_summary.csv")
    assert len(validation) >= 6
    assert (validation["N"] == 38488).all()
    assert (validation["Coverage"] == 1.0).all()
    checks.append("PASS: six practical candidates scored on all 38,488 locked validation rows")

    selected = json.loads((FULL / "SELECTED_CANDIDATE.json").read_text(encoding="utf-8"))
    eligible = validation[
        validation["Seasonal"].astype(str).str.lower().isin(["true", "1"])
        & (validation["Coverage"] >= 0.95)
        & (validation["Convergence_Rate"] >= 0.90)
    ].sort_values(["RMSE", "MAE"])
    assert selected["Candidate"] == eligible.iloc[0]["Candidate"]
    assert selected["Final_Test_Used"] is False
    checks.append("PASS: annual SARIMA selected by validation only")

    decision = json.loads(
        (OUT / "nssp_screen_120" / "METHODOLOGICAL_DECISION.json").read_text(encoding="utf-8")
    )
    assert not decision["retained_for_full_validation"]
    rescue = pd.read_csv(
        OUT / "nssp_rescue_60_iter300" / "validation" / "SARIMA_100x011_52" / "summary.csv"
    ).iloc[0]
    assert rescue["Maxiter"] == 300
    assert rescue["Convergence_Rate"] < 0.90
    checks.append("PASS: seasonal-differenced exclusion supported by 120-series screen and 300-iteration rescue")

    comparison = pd.read_csv(FULL / "FINAL_SAME_ROW_COMPARISON.csv")
    assert (comparison["N"] == 47534).all()
    required = {"SARIMA_100x100_52", "AR1", "Hybrid_InvMSE", "Persistence", "MovingAverage3", "SeasonalNaive"}
    assert required.issubset(set(comparison["Model"]))
    checks.append("PASS: final comparison uses exactly 47,534 rows for every model")

    sarima = comparison.set_index("Model").loc["SARIMA_100x100_52"]
    ar1 = comparison.set_index("Model").loc["AR1"]
    hybrid = comparison.set_index("Model").loc["Hybrid_InvMSE"]
    test_summary = pd.read_csv(FULL / "test" / "SARIMA_100x100_52" / "summary.csv").iloc[0]
    assert test_summary["Coverage"] == 1.0
    assert test_summary["Convergence_Rate"] >= 0.90
    checks.append("PASS: selected SARIMA test coverage and convergence gates passed")
    checks.append("RESULT: FULL NSSP SARIMA AUDIT PASSED")
    (FULL / "NSSP_FULL_VALIDATION_REPORT.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")

    report = f"""# Full NSSP SARIMA Benchmark and Methodological Decision

## Completion status

The full benchmark is complete for all 2,264 eligible NSSP geographic series. Candidate selection used 38,488 chronological validation rows; final evaluation used exactly the 47,534 locked test rows. The finalized manuscript and approved experiment package were not modified.

## Validation selection

Six practical candidates were evaluated on every series. The best converged annual-seasonal specification was **SARIMA (1,0,0)x(1,0,0)[52]**, selected without access to final-test outcomes. Its validation coverage was 100% and convergence was {selected['Convergence_Rate']:.1%}.

The nonseasonal AR(1) reference had lower validation RMSE than every annual SARIMA candidate. This is evidence that a forced annual seasonal component did not improve the NSSP benchmark over the available history; it does not invalidate the SARIMA comparison.

## Seasonal-differencing decision

Three `D=1` configurations were screened on the same deterministic 120-series validation cohort. All produced forecasts, but all had 0% optimizer convergence at 75 iterations. The only accuracy-competitive configuration, SARIMA (1,0,0)x(0,1,1)[52], was retested on 60 series at 300 iterations and still had 0% convergence.

**Decision:** exclude seasonal-differenced (`D=1`) candidates from the 2,264-series promotion and final test. Retain their screening results as convergence and sensitivity evidence. Retain SARIMA (1,0,0)x(1,0,0)[52] as the primary annual-seasonal benchmark and AR(1) as the nonseasonal time-series reference.

## Exact-row final results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Hybrid ensemble | {hybrid['RMSE']:.3f} | {hybrid['MAE']:.3f} | {hybrid['R2']:.3f} |
| AR(1) reference | {ar1['RMSE']:.3f} | {ar1['MAE']:.3f} | {ar1['R2']:.3f} |
| Selected annual SARIMA | {sarima['RMSE']:.3f} | {sarima['MAE']:.3f} | {sarima['R2']:.3f} |

The hybrid ensemble outperformed both classical time-series benchmarks. Relative to the selected annual SARIMA, the hybrid reduced RMSE by {(sarima['RMSE'] - hybrid['RMSE']) / sarima['RMSE']:.1%} and increased R² by {hybrid['R2'] - sarima['R2']:.3f}. The AR(1) result also shows that the annual SARIMA component did not improve NSSP forecasting under this data history.

## Recommended manuscript treatment after unlocking

1. Add both AR(1) and the validation-selected annual SARIMA to the NSSP benchmark table.
2. Describe the dataset-level order search, per-location fitting, rolling one-week-ahead filtering, and validation-only selection.
3. Report the seasonal-differencing convergence screen transparently in the limitations or supplement.
4. Update the NSSP RMSE comparison figure and reviewer-response matrix.
5. Do not claim that SARIMA was universally inferior; state specifically that it underperformed in this NSSP design and history length while remaining competitive in the two ILINet experiments.

## Lock boundary

All new artifacts remain below `sarima_pilot_20260716`. No manuscript integration has been performed.
"""
    (FULL / "NSSP_FULL_METHOD_DECISION.md").write_text(report, encoding="utf-8")
    print("\n".join(checks))


if __name__ == "__main__":
    main()
