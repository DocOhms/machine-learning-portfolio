"""Maintainer-only allowlist packager; copies evidence and hashes, never fits models.

Supply the preserved development workspace and its matching legacy project.
This helper is not needed to review the already included archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-workspace", required=True, type=Path)
    parser.add_argument("--legacy-project", required=True, type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    archive = destination.parent
    records = []

    def copy(source: Path, target: Path, logical_source: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing archive file: {target}")
        shutil.copy2(source, target)
        records.append({"file": target.relative_to(archive).as_posix(),
                        "source": logical_source, "source_sha256": sha256(source),
                        "copied_sha256": sha256(target), "bytes": target.stat().st_size})

    runners = ["run_rolling_component_lags.py", "run_final_lags123.py",
               "run_ilinet_exp2_exp3_provider_lags.py"]
    for name in runners:
        copy(args.development_workspace / "work" / name, destination / name,
             "preserved_development/work/" + name)
    for name in ["common_config.py", "common_io.py", "common_features.py",
                 "common_models.py", "common_metrics.py"]:
        copy(args.legacy_project / "src" / name, destination / "legacy_project/src" / name,
             "preserved_legacy_project/src/" + name)

    evidence_names = [
        "nssp_rolling_component_lags_fold_metrics.csv",
        "nssp_rolling_component_lags_folds.csv",
        "nssp_rolling_component_lags_summary.csv",
        "nssp_rolling_component_lags_weights.csv",
        "nssp_locked_component_lags123_comparison_vs_lag1.csv",
        "nssp_locked_component_lags123_test_metrics.csv",
        "nssp_locked_component_lags123_validation_metrics.csv",
        "nssp_locked_component_lags123_weights.csv",
        "nssp_locked_component_lags123_features.csv",
        "nssp_locked_component_lags123_metadata.csv",
        "nssp_t_plus_1_lag1_comparison.csv",
        "ilinet_exp2_exp3_provider_lags_final_hybrid_comparison.csv",
        "ilinet_exp2_exp3_provider_lags_final_test_metrics.csv",
        "ilinet_exp2_exp3_provider_lags_final_weights.csv",
        "ilinet_exp2_exp3_provider_lags_folds.csv",
        "ilinet_exp2_exp3_provider_lags_metadata.csv",
        "ilinet_exp2_exp3_provider_lags_rolling_metrics.csv",
        "ilinet_exp2_exp3_provider_lags_rolling_summary.csv",
    ]
    for name in evidence_names:
        copy(args.development_workspace / "outputs" / name, destination / "evidence" / name,
             "preserved_development/outputs/" + name)

    classical = args.development_workspace / "sarima_pilot_20260716/outputs"
    retained_names = {"summary.csv", "COMPLETE.json", "METHODOLOGICAL_DECISION.json",
                      "screening_candidate_comparison.csv", "screening_groups.csv",
                      "diagnostics.csv"}
    for folder in ["nssp_screen_120", "nssp_rescue_60_iter300"]:
        for source in sorted((classical / folder).rglob("*")):
            if source.is_file() and source.name in retained_names:
                relative = source.relative_to(classical)
                copy(source, archive / "classical_screening" / relative,
                     "preserved_classical_outputs/" + relative.as_posix())

    manifest = {"purpose": "Source provenance; paths are logical and contain no local username.",
                "training_performed": False, "source_files_changed": False,
                "files": records}
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    print(f"Copied {len(records)} allowlisted archival files. No experiments performed.")


if __name__ == "__main__":
    main()
