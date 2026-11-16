# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Create a stable SHA-256 manifest for preserved repository artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "provenance" / "repository_manifest.sha256"
INCLUDED_ROOTS = [
    "src",
    "experiments",
    "benchmarks",
    "data/raw",
    "results/reference",
    "manuscript",
]


# Calculation provenance:
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Calculation provenance:
#   sha256 -> local function in this file
def main() -> None:
    files: list[Path] = []
    for relative in INCLUDED_ROOTS:
        root = PROJECT_ROOT / relative
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    lines = []
    for path in sorted(set(files), key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"{sha256(path)}  {relative}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
