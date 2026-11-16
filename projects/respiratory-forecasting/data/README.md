# Exact study data and saved calculation evidence

The three raw surveillance files are deliberately absent from the public-review package. Their byte lengths and SHA-256 hashes are in [manifest.json](manifest.json). The expected names are `NSSP.csv`, `ILINet.csv`, and `ILINet_2016_2021.csv` in `data/raw/`.

No durable public location or redistribution permission has been established by this packaging step. Obtain author approval and an exact-snapshot access route before claiming fully downloadable reproduction. The NSSP raw file is over 100 MiB; do not place it in ordinary Git history. A future release/archive should identify data vintages and access conditions explicitly. Do not silently replace missing files with current CDC downloads.

For local author use, `scripts/stage_data.py --source <folder>` copies the exact files from an authorized source, and `scripts/verify_data.py` checks their hashes. These scripts are callable from a notebook using `sys.executable`.

## Reference predictions

`reference_calculations.zip` contains the saved row-level prediction CSVs under their original `results/reference/` relative paths. [REFERENCE_ARCHIVE.json](REFERENCE_ARCHIVE.json) gives the ZIP checksum and every member's checksum. These are forecasts and comparison evidence, not serialized fitted models or replacement raw inputs.

The calculation notebook reads this ZIP directly. The archive is a separate GitHub Release asset, not a Git-tracked file. Download the matching `reference_calculations.zip` from the portfolio's [Releases](https://github.com/DocOhms/machine-learning-portfolio/releases), then place it in `projects/respiratory-forecasting/data/` in your checkout. Keep its exact name. `.gitignore` excludes it from normal commits. Saved notebook outputs remain viewable without it. All data access is local in the current notebook; no automatic downloads occur.

Release status: author-approved for publication on 16 November 2026; asset upload is pending completion of the portfolio publication workflow. A direct verified download link will be recorded here after upload. The checksum remains `2cca184052103a473c8beefe2ba1a4b3d4d6c90591edfde4f25bdcf040120c6a`.

The archive and numerical evidence are not covered by the MIT code license. See [terms and attribution](../TERMS_AND_ATTRIBUTION.md) for data-source conditions, non-endorsement, and the distinction between derived results and unbundled raw snapshots.
