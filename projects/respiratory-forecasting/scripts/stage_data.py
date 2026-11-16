# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Copy preserved raw data into the repository and verify exact hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
TARGET_DIR = PROJECT_ROOT / "data" / "raw"


# Calculation provenance:
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source(source: Path) -> Path:
    candidates = [source, source / "data" / "raw"]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "NSSP.csv").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find NSSP.csv in {source} or {source / 'data' / 'raw'}"
    )


# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   resolve_source -> local function in this file
#   sha256 -> local function in this file
#   shutil.copy2 -> shutil.copy2 (Python standard library)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="A raw-data directory or project directory containing data/raw.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing target only after verifying the source hash.",
    )
    args = parser.parse_args()

    source_dir = resolve_source(args.source.resolve())
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for item in manifest["files"]:
        source = source_dir / item["name"]
        if not source.exists():
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        if source.stat().st_size != item["bytes"] or source_hash != item["sha256"]:
            raise RuntimeError(f"Source does not match manifest: {source}")

        target = TARGET_DIR / item["name"]
        if target.exists() and not args.overwrite:
            if target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"]:
                print(f"PASS existing {target.name}")
                continue
            raise FileExistsError(
                f"Target exists but does not match the manifest: {target}. "
                "Inspect it, then use --overwrite only if replacement is intended."
            )
        shutil.copy2(source, target)
        if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            raise RuntimeError(f"Post-copy verification failed: {target}")
        print(f"COPIED {source} -> {target}")

    print("Data staging and verification complete.")


if __name__ == "__main__":
    main()
