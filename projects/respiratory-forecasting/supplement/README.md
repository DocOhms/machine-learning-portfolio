# GitHub companion evidence

This folder is the entry point for the paper's supplementary information in the machine-learning portfolio. It is GitHub companion evidence, not an additional document claimed to be accepted by the conference. The original code has an MIT license; manuscript, data, saved figures, and third-party materials have [separate terms and attribution](../TERMS_AND_ATTRIBUTION.md).

Start with the [calculation review notebook](../notebooks/Calculation_Review.ipynb), [calculation guide](../docs/CALCULATIONS.md), and [function/source index](../docs/FUNCTION_INDEX.md). The [focused supplement PDF](../manuscript/latex/focused_supplement.pdf) is an optional, readable companion; the calculations and machine-readable evidence remain available separately. The [six-page paper](../manuscript/latex/main.pdf) is the conference manuscript.

The notebook is the canonical shortened publication version. It retains the core arithmetic, compact result tables, and three unchanged heat maps, with saved outputs for inspection without execution. Source-file/function links replace complete implementation dumps; the detailed editable author notebook stays separate. All four execution/export switches are off. [The publication-notebook check record](../docs/PUBLIC_NOTEBOOK_CHECKS.json) documents arithmetic-only verification of this version, not new model training or public hosting; earlier audit reports retain their historical scope.

## Evidence by supplement section

| Section in the focused supplement | What can be reviewed | Code and evidence location |
| --- | --- | --- |
| S1. Additional feature, fitting, and cluster-profile details | Lag/rolling/seasonal definitions, fitting boundaries, learner settings, cluster selection and profiles, bounded classical search and convergence | `src/common_features.py`, `src/common_config.py`, `src/common_models.py`; `manuscript/latex/manuscript_kmeans_diagnostics.py` and `kmeans_evidence/`; `benchmarks/sarima/` and `results/reference/classical/` |
| S2. Auxiliary-lag development history | Archived rolling-origin and same-row final-test comparisons; this is disclosure of development history, not fresh validation | `manuscript/latex/sensitivity_data/source/`; `S1_lag_test_comparison.csv`, `S1_nssp_rolling_validation.csv`, and `S1_nssp_rolling_windows.csv` in `sensitivity_data/` |
| S3. Ensemble-weight sensitivity | Reconstruct the saved inverse-MSE ensemble; compare it with the mean of the same three component predictions | `manuscript/latex/sensitivity_analysis.py`; `sensitivity_data/S2_ensemble_weights.csv` and `S2_ensemble_weighting_metrics.csv`; saved component predictions |
| S4. Primary NSSP HSA-week evaluation and record-weighted comparison | Existing scores after restricting to HSA-source records and averaging within HSA-week; all retained methods use the same primary units | `manuscript/latex/sensitivity_analysis.py`; `sensitivity_data/S3_nssp_source_counts.csv`, `S3_hsa_week_mean_predictions.csv`, and `S3_nssp_geographic_metrics.csv` |
| S5. Residual diagnostics | Predicted-minus-observed residuals and descriptive within-week summaries, not prediction intervals | `src/common_plots.py` and the primary runners for NSSP/contemporary ILINet (`fig06.png`, `fig09.png`); `manuscript/latex/build_calendar_figures.py` and `calendar_weekly_residuals.csv` for historical ILINet only (`fig12.png`) |
| S6. Full-size geographic maps | Test-period mean predictions, displayed with distinct target-appropriate scales and explicit missing-data shading | `manuscript/latex/build_review_figures.py`, `render_review_maps.cjs`, `map_values.csv`, `map_provenance.json`, and figures `fig13.png`–`fig15.png` |
| S7. Observed and predicted temporal trajectories | Unchanged reporting-record weekly means and peak timing; primary NSSP metrics use HSA-weeks, while ILINet metrics use eligible reporting rows | `src/common_plots.py` and the primary runners for NSSP/contemporary ILINet (`fig04.png`, `fig07.png`); `manuscript/latex/build_calendar_figures.py` and `calendar_weekly_trajectory.csv` for historical ILINet only (`fig10.png`) |

Paths in the last column are relative to the repository root. The historical `S1_`, `S2_`, and `S3_` CSV prefixes map to **Sections S2, S3, and S4**, respectively; their older filenames have been retained to preserve provenance. The WHO citation remains in the supplement beside the historical residual figure and in its reference list.

## Calculation versus archived evidence

The primary feature and forecasting calculations are implemented in `src/` and called by the three primary experiment runners. Third-party training algorithms are identified by package and class in the calculation guide; the repository does not claim to reimplement scikit-learn or statsmodels internally.

The sensitivity script recalculates equal-weight and HSA-week scores from saved forecasts without fitting learners. Its auxiliary-lag section instead imports previously computed development summaries. The underlying alternative-lag experiments are **not** rerun by that script. Treat those summaries as archived evidence, not as a newly reproduced selection exercise.

The recovered original development runners, matching legacy helpers, and compact
fold evidence are in [archive/development](../archive/development/README.md).
Slower seasonal-differencing screen and rescue diagnostics are in
[archive/classical_screening](../archive/classical_screening/README.md).
The rolling-development folds used their validation predictions both to estimate
weights and summarize the weighted ensemble; they are not independent nested
estimates of that weighting rule. Those caveats remain visible alongside the code.

Full row-level reference files are supplied separately as `reference_calculations.zip` through the portfolio's [Releases](https://github.com/DocOhms/machine-learning-portfolio/releases). See [data access and current release status](../data/README.md). To repeat calculations after downloading the repository, obtain the matching release asset and place it in this project's `data/` folder. The default notebook reads it directly, with no extraction needed. Its saved tables and maps can be inspected without the archive or a running Python environment. To run the original command-line scripts, deliberately set `EXTRACT_REFERENCE = True` in the final notebook cell. That checksum-checked operation stages its `results/reference/` CSVs without fitting models or overwriting differing evidence. Small metric, setting, and diagnostic files remain directly browsable. Raw surveillance snapshots are separate: see `data/manifest.json` and the data-access instructions. Newer CDC downloads need not match the study's preserved data vintage.

## Interpretation boundaries

- Experiments 1–3 are independently retrained applications of the same procedure; no fitted NSSP model is transferred to ILINet.
- Final-test outcomes are excluded from parameter fitting within a run, but all three final-test windows were inspected during auxiliary-lag development. These are retrospective comparisons, not untouched confirmatory tests.
- Completed-week availability is assumed; publication delay and later revisions were not reconstructed.
- NSSP primary scores now count each HSA-week once (R² = 0.599); original reporting-record-weighted scores remain a comparison (R² = 0.671). Training and validation retain the original record weighting. This post-review reporting amendment uses existing saved scores and does not refit an HSA-level model. Historical CSV labels and numeric values remain unchanged for provenance.
- Maps are descriptive mean predictions, not official state rates, validated hotspots, spatial-correlation tests, or population-density analyses. NSSP uses orange, 0–4%; the two ILINet maps use distinct blue/purple palettes with shared 0–6% limits. Gray means missing eligible mapped data, not zero.
- Equal weighting is competitive. Core lag/rolling/seasonal ablations, seasonal weight stability, new experiments, and uncertainty analyses remain unperformed.

The revised paper and companion explain the new reporting hierarchy; archived numerical results and figures are preserved. No new scientific experiment is implied by this reporting and annotation pass.
