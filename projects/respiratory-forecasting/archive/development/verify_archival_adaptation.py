"""Verify an archival public copy; this script never imports/fits scientific code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-workspace", required=True, type=Path,
                        help="Maintainer's preserved original workspace for comparison")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    archive = here.parent
    manifest = json.loads((here / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    runners = {"run_rolling_component_lags.py", "run_final_lags123.py",
               "run_ilinet_exp2_exp3_provider_lags.py"}
    records = []
    for item in manifest["files"]:
        target = archive / item["file"]
        data = target.read_bytes()
        record = {"file": item["file"], "source_sha256": item["source_sha256"],
                  "public_sha256": digest(data)}
        if target.name in runners:
            source = args.development_workspace / "work" / target.name
            original = source.read_bytes()
            assert digest(original) == item["source_sha256"]
            marker = "OUTPUT.mkdir(parents=True, exist_ok=True)"
            before = original.decode("utf-8").replace("\r\n", "\n").split(marker, 1)[1]
            after = data.decode("utf-8").replace("\r\n", "\n").split(marker, 1)[1]
            assert before == after, target.name
            record["calculation_body_unchanged"] = True
            record["calculation_body_sha256"] = digest(after.encode("utf-8"))
            record["change"] = "Only startup imports, comments and explicit input/output path arguments."
        else:
            assert digest(data) == item["source_sha256"], target
            record["byte_identical"] = True
        if target.suffix == ".py":
            ast.parse(data.decode("utf-8"), filename=target.name)
            record["syntax_valid"] = True
        records.append(record)
    report = {"training_performed": False, "all_checks_passed": True,
              "verification": "Byte identity for evidence/helper copies; unchanged runner bodies after path setup; Python syntax.",
              "full_rerun_validated": False, "files": records}
    (here / "ADAPTATION_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(records)} archived files; calculation bodies preserved; no models run.")


if __name__ == "__main__":
    main()
