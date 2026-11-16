# Respiratory forecasting: public review companion

This research project is part of [Alexander Roque's machine-learning portfolio](../../README.md) and accompanies the six-page ICCA manuscript. It preserves the scientific model calculations, reference results, and final heat maps, with calculation provenance and a readable Jupyter review notebook. No new experiment was run for this publication. Original code is MIT-licensed; manuscript, data, saved figures, and third-party materials have [separate terms and attribution](TERMS_AND_ATTRIBUTION.md).

## Start here

1. Open [notebooks/Calculation_Review.ipynb](notebooks/Calculation_Review.ipynb), the canonical shortened publication notebook. Its saved tables and three unchanged maps can be inspected without running it. To repeat the calculation review, run its cells in order with Python 3.12 and the matching reference archive. Default cells use Python's standard library and Jupyter display tools; they do not train models, alter the paper, or overwrite results. All four execution/export switches are off.
2. Read [docs/CALCULATIONS.md](docs/CALCULATIONS.md) for formulas and [docs/FUNCTION_INDEX.md](docs/FUNCTION_INDEX.md) for exact function locations. Source comments identify imported helpers and packages.
3. Use [supplement/README.md](supplement/README.md) to navigate the GitHub companion evidence. The supplement PDF is a convenient rendering of that material, not an additional document claimed to be accepted by the conference.
4. Read [TERMS_AND_ATTRIBUTION.md](TERMS_AND_ATTRIBUTION.md) before reusing code or accompanying materials. [docs/PUBLIC_RELEASE_REVIEW.md](docs/PUBLIC_RELEASE_REVIEW.md) preserves the earlier release-review checklist; [docs/PORTFOLIO_PUBLICATION_CHECKS.json](docs/PORTFOLIO_PUBLICATION_CHECKS.json) records the later author-authorized portfolio integration.

The notebook retains the core feature, ensemble, error-metric, HSA-week, and map-aggregation calculations with compact result tables. It links to the exact source files/functions for fitting and other implementation detail instead of printing complete source-code dumps. The longer editable author notebook is kept separately; it is not a second public entry point.

The [six-page manuscript](manuscript/latex/main.pdf), [supplement PDF](manuscript/latex/focused_supplement.pdf), and notebook now lead with equal-HSA-week NSSP evaluation: R² = 0.599 and RMSE = 0.972 percentage points across 13,356 HSA-weeks. The original reporting-record-weighted result (R² = 0.671; RMSE = 0.803; 47,534 records) remains a secondary comparison. This post-review reporting amendment reuses existing scores and forecasts; training, validation, source result files, and [final figures](manuscript/latex/figures/) are unchanged. Private reviewer correspondence, old manuscript versions and the personal Jupyter environment are excluded.

## Two levels of reproducibility

**Review saved calculations:** the notebook reconstructs ensemble weights and forecasts, calculates errors and metrics, checks identical classical comparison rows, and verifies geographic means. Re-executing these checks needs `data/reference_calculations.zip`; viewing the saved tables and maps does not. The notebook reads the archive directly without unpacking. The unchanged archive is distributed separately through the portfolio's [Releases](https://github.com/DocOhms/machine-learning-portfolio/releases); consult [data/README.md](data/README.md) for its release status and download instructions. Place it inside this project's `data/` folder without renaming it. Its checksum is in [data/REFERENCE_ARCHIVE.json](data/REFERENCE_ARCHIVE.json), and `.gitignore` excludes it from ordinary Git commits.

The six-page manuscript now links directly to this project and the versioned reference archive in Section V. The notebook and supplement retain their reviewed contents; any historical references to pending public hosting are superseded by this README and `data/README.md`.

## Manuscript and Overleaf downloads

The [six-page PDF](manuscript/latex/main.pdf) and [LaTeX source](manuscript/latex/main.tex) include the public reproducibility links. The [document-only v1.0.1 release](https://github.com/DocOhms/machine-learning-portfolio/releases/tag/respiratory-forecasting-v1.0.1) provides the [public Overleaf ZIP](https://github.com/DocOhms/machine-learning-portfolio/releases/download/respiratory-forecasting-v1.0.1/ICCA_Six_Page_GitHub_Links_Overleaf.zip) and a [direct PDF download](https://github.com/DocOhms/machine-learning-portfolio/releases/download/respiratory-forecasting-v1.0.1/ICCA_Six_Page_GitHub_Links.pdf). In Overleaf, select `main.tex` and XeLaTeX. The ZIP includes the unchanged focused supplement, tables, and figure assets, but excludes reviewer correspondence and internal author notes. It does not include raw surveillance snapshots or the reference-calculation archive.

Only the paper's reproducibility paragraph changed. The PDF remains six A4 pages; models, results, maps, and all four disabled notebook switches are unchanged. The [document-publication check record](docs/MANUSCRIPT_LINKS_PUBLICATION_CHECKS.json) and [file manifest](PUBLIC_FILE_MANIFEST.json) record preservation checks and release-asset checksums. The original [v1.0.0 calculation archive](https://github.com/DocOhms/machine-learning-portfolio/releases/tag/respiratory-forecasting-v1.0.0) remains unchanged.

## Optional calculation replay and refitting

The original verification, sensitivity and benchmark scripts expect CSVs on disk.
Set only `EXTRACT_REFERENCE = True` in the notebook's final cell to stage those
checksum-verified files. Keep all `RUN_*` switches false if you only want to unpack
the evidence. This never overwrites a differing reference file.

**Refit the experiments:** install the preserved [requirements.txt](requirements.txt), obtain the exact three raw snapshots listed in [data/manifest.json](data/manifest.json), stage them in `data/raw/`, and run `python scripts/verify_data.py`. New CDC downloads may not reproduce the preserved study vintage. Then use the explicitly disabled rerun cell in the notebook. Its commands run the same scripts through the notebook's Python interpreter. Training outputs go to experiment-specific folders under `results/`; they do not replace `results/reference/`. Run in a separate checkout for author experiments and never promote a new result without review.

The default notebook is not a full training-environment validation. The recorded classical environment includes pandas 3.0.1, whereas the unchanged dependency file specifies 3.0.3. See the release checklist. Fitted estimator objects were not archived; saved component forecasts permit arithmetic verification, but individual tree paths and MLP activations require refitting.

## Jupyter setup in a fresh checkout

On Windows, use a short checkout location such as `C:\ml\machine-learning-portfolio`. Some preserved evidence filenames are long and can exceed older Windows path limits inside deeply nested folders.

From the portfolio repository root, first enter this project folder:

```text
cd projects/respiratory-forecasting
```

Then create a project-specific environment:

```text
python -m venv .venv
```

Activate that environment with your operating system's usual command, then:

```text
python -m pip install -r requirements-review.txt
python -m jupyterlab
```

Use a Python 3.12 kernel from that environment. For full scientific runs, additionally install `requirements.txt`. The original graphical export helpers may require Chrome/Kaleido; optional manuscript PNG conversion uses Node.js and `sharp`, and framework drawing uses ReportLab plus the original Arial fonts. These are presentation dependencies, not needed for reviewing saved numerical results. No renderer is run by default.

## What has and has not changed

- Imported-function notes and formula documentation were added; annotation-only changes preserve the Python syntax tree.
- Two presentation/verification default paths were made relative, and a machine-specific Node dependency fallback was removed. Local path labels in provenance were generalized; numeric values and original evidence hashes were retained.
- Three independently retrained experiments remain the primary analyses. No zero-shot transfer or operational readiness is claimed. All three final test windows were inspected during auxiliary-lag development.
- NSSP primary evaluation counts each HSA-week once; model fitting and validation retain their original reporting-record weighting. This is not a deduplicated-training experiment. Historical files labeled "primary" preserve the original record-weighted analysis for provenance; the notebook explicitly identifies their current secondary reporting role.
- [docs/HSA_REPORTING_REVISION_CHECKS.json](docs/HSA_REPORTING_REVISION_CHECKS.json) records the reporting-update checks. The earlier `docs/REVIEW_VERIFICATION.json` is retained as the initial package audit, before the HSA-week reporting amendment.
- [docs/PUBLIC_NOTEBOOK_CHECKS.json](docs/PUBLIC_NOTEBOOK_CHECKS.json) records the shortened notebook's arithmetic-only verification and publication-preparation checks. Its saved outputs come from reviewing existing evidence, not a fresh model fit or a new experiment. This report does not establish a live GitHub release or full training-environment reproduction.
- The auxiliary-lag development summaries are historical evidence. The sensitivity script does not retrain those comparisons; their archived source is separately identified.
- The maps remain descriptive test-period means, not validated hotspots or official population-weighted state rates. The runner's legacy geographic bar chart is not the final paper map.

## Publication status

The repository's `.gitattributes` disables automatic line-ending conversion to preserve checksum-verified bytes. [PUBLIC_FILE_MANIFEST.json](PUBLIC_FILE_MANIFEST.json) inventories repository files separately from the reference release asset and excludes its own hash to avoid a circular checksum. [docs/UPLOAD_PREPARATION_CHECKS.json](docs/UPLOAD_PREPARATION_CHECKS.json) records this packaging-only check; no notebook cells, models, tests, or figure renderers were run for upload preparation. Save any final author edits before refreshing the release checksums again.

Original code is released under the [MIT License](LICENSE), following author approval on 16 November 2026. The manuscript, raw and derived data, figures, and third-party assets are excluded from that blanket code grant; see [TERMS_AND_ATTRIBUTION.md](TERMS_AND_ATTRIBUTION.md). Exact raw-snapshot access and the final proceedings citation remain unresolved and are not claimed complete. No new model fit, scientific experiment, or map rendering was performed for portfolio publication. Earlier audit files retain their historical status and do not describe the current hosting state.
