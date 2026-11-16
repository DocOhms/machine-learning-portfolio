# Public file integrity

The public release manifest is `PUBLIC_FILE_MANIFEST.json` at the repository root.
Original source hashes and annotation/path adaptations are recorded separately in
`docs/SOURCE_PRESERVATION.json`. Saved prediction hashes are in
`data/REFERENCE_ARCHIVE.json`; raw snapshot hashes are in `data/manifest.json`.

The retained utility `scripts/create_repository_manifest.py` writes a separate
`repository_manifest.sha256` here for its original selected artifact scope. It does
not cover the entire public package and is not a replacement for the public manifest.
