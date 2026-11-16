# Seasonal-differencing screening evidence

These unchanged small files document why the slower seasonal-differenced NSSP
candidates were not promoted to the full 2,264-series comparison. No models were
fitted or predictions changed when assembling this archive.

- `nssp_screen_120/screening_groups.csv`: the deterministic validation screening
  cohort shared by the compared candidates.
- `nssp_screen_120/screening_candidate_comparison.csv`: pooled validation metrics,
  finite-forecast coverage, convergence rates, and timing for the identical cohort.
- `nssp_screen_120/METHODOLOGICAL_DECISION.json`: the actual promotion rule and
  recorded decision. Promotion required coverage at least 0.95, convergence at
  least 0.90, and RMSE no greater than 1.01 times the best practical seasonal RMSE.
- `nssp_screen_120/validation/*/{summary.csv,diagnostics.csv,COMPLETE.json}`:
  the screened candidates' aggregate and per-series convergence evidence.
- `nssp_rescue_60_iter300/`: the 60-series, 300-iteration rescue attempt for the
  accuracy-competitive seasonal-differenced candidate. Its optimizer convergence
  still failed the promotion criterion.

## Where the numbers are calculated

The runnable shared implementation is `benchmarks/sarima/sarima_pilot.py`:
`fit_and_predict_group()` calls statsmodels `SARIMAX.fit(method="lbfgs")`, records
the returned convergence flag, filters the series with the fitted parameters,
then asks for the one-step **predicted** information set. A finite forecast is
not the same as a converged fit. Finite forecasts from nonconverged fits are
retained in the reported score; convergence is reported separately.

`benchmarks/sarima/nssp_full_runner.py` provides `run_batched()`,
`score_summary()`, and `methodological_decision()`. Coverage is the number of
finite scored forecasts divided by eligible scoring rows; convergence rate is
the mean of per-series optimizer convergence flags. RMSE/MAE/R-squared are
calculated by `sarima_pilot.metrics()`. The practical seasonal comparator is
rescored on the exact screening groups before the ratio is calculated.

The normal NSSP full seasonal selector additionally requires at least 90%
convergence. The ILINet pilot selector instead uses its 95% coverage gate and
validation RMSE, then MAE. Do not describe those gates as identical.

Only compact diagnostics are included here, not the screening row-level
predictions or duplicate batch files. Consequently these files let readers audit
the archived decision and per-series convergence, but alone do not permit
recomputing all screening RMSE values from individual forecasts. Full rerunning
requires raw snapshots and reference cohorts, as described by the benchmark
README; that is an optional computation, not a new result in this release.

Source hashes are recorded in `../development/SOURCE_MANIFEST.json`; original
locations are represented by logical archive-relative paths without usernames.
