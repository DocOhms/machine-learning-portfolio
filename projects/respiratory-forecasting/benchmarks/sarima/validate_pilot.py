# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Audit pilot inputs, cohorts, selection rule, and final prediction coverage."""
from pathlib import Path
import hashlib
import json

import pandas as pd
from sarima_pilot import OUTPUT_ROOT


ROOT = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT


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
        path = Path(row.path)
        assert path.exists(), path
        assert sha256(path) == row.sha256, f"Input hash changed: {path}"
    checks.append(f"PASS: {len(manifest)} locked input hashes verified")

    for key in ("exp1", "exp2", "exp3"):
        run = json.loads((OUT / key / "run_summary.json").read_text(encoding="utf-8"))
        audit = run["cohort_audit"]
        assert audit["reproduced_validation_rows"] == audit["locked_validation_rows"]
        assert audit["reproduced_test_rows"] == audit["locked_test_rows"]
        assert audit["test_keys_missing_from_reproduction"] == 0
        assert audit["extra_reproduced_test_keys"] == 0
        candidates = pd.read_csv(OUT / key / "validation_candidate_summary.csv")
        eligible = candidates[(candidates["Seasonal"]) & (candidates["Coverage"] >= 0.95)]
        expected = eligible.sort_values(["RMSE", "MAE"]).iloc[0]["Candidate"]
        assert run["selected_sarima"] == expected
        test = pd.read_csv(OUT / key / "final_test_summary.csv").iloc[0]
        assert test["Coverage"] >= 0.95
        assert test["Error_Groups"] == 0
        checks.append(
            f"PASS: {key} exact cohorts, validation-only selection, "
            f"coverage={test['Coverage']:.3f}, convergence={test['Convergence_Rate']:.3f}"
        )

    comparison = pd.read_csv(OUT / "same_row_test_comparison.csv")
    for key in ("exp1", "exp2", "exp3"):
        subset = comparison[comparison["Experiment"] == key]
        assert subset["N"].nunique() == 1
    checks.append("PASS: all models within each experiment use identical final-test row counts")
    checks.append("RESULT: PILOT OUTPUT AUDIT PASSED")
    (OUT / "VALIDATION_REPORT.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("\n".join(checks))


if __name__ == "__main__":
    main()
