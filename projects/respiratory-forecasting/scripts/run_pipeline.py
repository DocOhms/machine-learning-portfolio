# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
# Use --profile primary for the paper primary runs. --profile all includes legacy diagnostics; --with-maps invokes the legacy renderer, not the final paper maps.
"""Run corrected hybrid forecasting entry points in documented order."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = [
    "experiments/experiment_1_nssp_primary.py",
    "experiments/experiment_2_ilinet_contemporary.py",
    "experiments/experiment_3_ilinet_pandemic_stress_test.py",
]
ALL = [
    "experiments/diagnostic_leakage_test_nssp.py",
    "experiments/diagnostic_n_clusters_sensitivity.py",
    "experiments/experiment_1_nssp_primary.py",
    "experiments/experiment_1_nssp_ablation_core_features_only.py",
    "experiments/experiment_1_nssp_ablation_no_clustering.py",
    "experiments/diagnostic_feature_importance_nssp.py",
    "experiments/experiment_2_ilinet_contemporary.py",
    "experiments/experiment_3_ilinet_pandemic_stress_test.py",
]


# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   subprocess.run -> subprocess.run (Python standard library)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["primary", "all"], default="primary")
    parser.add_argument("--with-maps", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    args = parser.parse_args()

    commands = list(PRIMARY if args.profile == "primary" else ALL)
    if args.with_maps:
        commands.append("generate_choropleth_maps.py")

    for relative in commands:
        command = [sys.executable, str(PROJECT_ROOT / relative)]
        print("RUN", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("Pipeline complete." if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    main()
