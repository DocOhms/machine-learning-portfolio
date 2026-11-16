# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Verify staged raw datasets against the repository data manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# Calculation provenance:
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Calculation provenance:
#   json.loads -> json.loads (Python standard library)
#   sha256 -> local function in this file
def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in manifest["files"]:
        path = RAW_DIR / item["name"]
        if not path.exists():
            failures.append(f"MISSING {path}")
            continue
        actual_bytes = path.stat().st_size
        actual_hash = sha256(path)
        size_ok = actual_bytes == item["bytes"]
        hash_ok = actual_hash == item["sha256"]
        status = "PASS" if size_ok and hash_ok else "FAIL"
        print(
            f"{status} {item['name']}: bytes={actual_bytes}, sha256={actual_hash}"
        )
        if not size_ok or not hash_ok:
            failures.append(
                f"{item['name']}: expected bytes={item['bytes']} sha256={item['sha256']}"
            )
    if failures:
        raise SystemExit("Data verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(manifest['files'])} raw datasets.")


if __name__ == "__main__":
    main()
