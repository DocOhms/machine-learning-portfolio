# Calculation guide

This guide follows one surveillance observation from its source CSV through the forecast and reported summaries. The linked Python functions are the authoritative implementation. Formulas explain those functions; they are not a second implementation or a change to the published models.

The same procedure is fitted independently for each experiment. Saved reference predictions and metrics are evidence from the locked runs. A new run creates new estimates and must not be silently substituted for those reference files.

## 1. Where calculations live

| Stage | Local implementation | Imported calculation provider |
| --- | --- | --- |
| Read and clean surveillance files | [`common_io.py`](../src/common_io.py): `load_nssp`, `load_ilinet`, `clean_numeric` | pandas CSV parsing, numeric conversion and dates |
| History features, chronological splits, preprocessing | [`common_features.py`](../src/common_features.py) | pandas shifts/rolling summaries; NumPy trigonometry; scikit-learn scaling and clustering |
| Learners, weights and final forecast | [`common_models.py`](../src/common_models.py) | scikit-learn RF, GBR, MLP and validation MSE |
| Parameter values | [`common_config.py`](../src/common_config.py): `MODEL_PARAMS` and constants | Values supplied to the imported estimators |
| Execution and saved outputs | [`_run_template.py`](../experiments/_run_template.py): `run_experiment` | Calls the local functions above in chronological order |
| Evaluation metrics | [`common_metrics.py`](../src/common_metrics.py) | NumPy arithmetic, scikit-learn error metrics, SciPy correlation |
| Forecast curves and residuals | [`common_plots.py`](../src/common_plots.py) | pandas aggregation and Matplotlib rendering |
| Final descriptive maps | [`build_review_figures.py`](../manuscript/latex/build_review_figures.py) | pandas reporting-unit means; SVG rendering using archived state geometry |

An import is a reference to code in another module or installed package. For example, `compute_validation_weights` is project code in `common_models.py`, whereas `RandomForestRegressor.fit` is implemented in scikit-learn. The project does not implement its own random forest, boosting optimizer or neural-network optimizer. The dependency versions are recorded in [`requirements.txt`](../requirements.txt); any difference between an archived environment and a review environment should be reported, not hidden.

For a local function, `inspect.getsource(function_name)` displays its calculation in Jupyter. `inspect.getfile(function_name)` identifies its file. For library estimators, inspect the estimator's `get_params()` output and installed package version as well as the local call site. Copying the entire third-party package into the notebook is unnecessary.

## 2. Source observations and time keys

The targets are already reported percentages in the source datasets; this code does **not** reconstruct them from patient-level counts:

- NSSP: `percent_visits_combined`, grouped by `geography` and `county`.
- Both ILINet experiments: `%UNWEIGHTED ILI`, grouped by `REGION TYPE` and `REGION`.

See the [NSSP runner](../experiments/experiment_1_nssp_primary.py), [contemporary ILINet runner](../experiments/experiment_2_ilinet_contemporary.py) and [historical ILINet runner](../experiments/experiment_3_ilinet_pandemic_stress_test.py).

`clean_numeric` replaces explicit `X`, `x` and blank markers, then uses `pandas.to_numeric(errors="coerce")`; other unparseable numeric values become missing. The ILINet loader skips the first CSV line. Missing targets or required predictors are excluded later; they are not imputed.

NSSP year and week are derived from the parsed `week_end` date using its ISO calendar. ILINet retains the supplied `YEAR` and `WEEK`. In both cases the ordering key is:

`epi_order = 100 × YEAR + WEEK`.

This is an ordering label, not an elapsed-time measurement. ILINet display dates use `epi_week_start`: the Sunday starting the MMWR week containing January 4, plus seven days for each subsequent week. Invalid calendar values produce missing display dates; the source week keys remain authoritative. The [calendar figure builder](../manuscript/latex/build_calendar_figures.py) applies this conversion to archived historical ILINet predictions for the corrected trajectory and residual displays. Its `calendar_weekly_*.csv` files are historical-only; the primary plotting helpers supply the NSSP and contemporary ILINet plots. Presentation-date correction does not alter scientific row keys or forecasts.

## 3. One observation's forecasting inputs

[`add_time_series_features`](../src/common_features.py) sorts each reporting series by `epi_order`. Let `y[t]` be its target for the row being forecast. The feature calculation uses preceding records within that series:

| Feature | Calculation |
| --- | --- |
| `lag_1`, `lag_2`, `lag_3` | `y[t−1]`, `y[t−2]`, `y[t−3]` |
| `rolling_mean_3` | `(y[t−1] + y[t−2] + y[t−3]) / 3` |
| `rolling_std_3` | `sqrt(sum((y[t−j] − rolling_mean_3)^2, j=1…3) / 2)` |
| `week_sin` | `sin(2π × WEEK / 52)` |
| `week_cos` | `cos(2π × WEEK / 52)` |
| NSSP auxiliary inputs | Previous-record COVID-19, influenza and RSV visit percentages |
| ILINet auxiliary input | Previous-record `NUM. OF PROVIDERS` |

The rolling operations first shift the target by one record and then require all three observations. The standard deviation is the **sample** standard deviation (`pandas` default `ddof=1`), hence denominator 2. Week 53 has the same encoded phase as Week 1.

These are observation shifts, not calendar joins. The feature function does not insert missing weeks or resolve duplicate weeks. Reviewing a particular point therefore includes examining its preceding source records and their week keys. A claim that every shifted record represents the preceding calendar week requires that series' continuity to be checked.

[`choose_feature_columns`](../src/common_features.py) uses an explicit allowlist: the seven temporal features plus the lagged auxiliaries. Current-week pathogen shares, raw geographic identifiers, smoothed values and trend fields are not implicitly included. The seasonal-naïve value is a benchmark, not a fitted-model feature.

This is a retrospective completed-week availability assumption. It does not reconstruct release delays, revisions or data vintages.

## 4. Eligibility and chronological boundaries

[`run_experiment`](../experiments/_run_template.py) drops rows missing the target or required cohort features before splitting. The NSSP core-only ablation keeps the full-model eligibility rule while omitting the auxiliary predictors from training; see [`experiment_1_nssp_ablation_core_features_only.py`](../experiments/experiment_1_nssp_ablation_core_features_only.py).

For the two automatic splits, the final test contains the latest `ceil(0.20 × number_of_unique_weeks)` weeks. The weight-validation period similarly contains the latest 20% of the remaining pretest weeks. All reporting units for a given week stay on the same side of each cutoff. These calculations are in `chronological_split` and `chronological_validation_split`.

The historical experiment instead uses an explicit inclusive pretest interval of 2016-W40–2019-W39 and test interval of 2019-W40–2020-W39. It uses the same chronological validation procedure within its pretest history.

No final-test outcomes enter a given run's scaler fit, centroid fit, learner fit or ensemble-weight calculation. This procedural boundary is separate from the disclosed development history: the final-test windows were inspected during auxiliary-lag development. The results are retrospective exploratory comparisons, not untouched confirmatory tests.

## 5. Scaling and cluster indicators

[`fit_preprocessor`](../src/common_features.py) fits scikit-learn `StandardScaler` using only the supplied fitting segment:

`z[i,j] = (x[i,j] − mean_fit[j]) / standard_deviation_fit[j]`.

Unlike the three-observation rolling standard deviation, `StandardScaler` uses a population variance convention (`ddof=0`). Its library implementation handles zero-variance columns. Validation or test rows use `transform_preprocessor`, which reuses the fitted means, scales and centroids without fitting them again.

K-means minimizes the sum of squared Euclidean distances from standardized observations to their assigned centroid:

`sum_i ||z[i] − centroid[cluster[i]]||²`.

The configuration requests four clusters, seed 42 and ten initializations. The helper limits the effective cluster count if there are too few unique rows. Each assigned label becomes a one-hot indicator: `cluster_k = 1` when the label equals `k`, otherwise `0`. These indicators are appended to the standardized features; they are not themselves standardized.

The reported silhouette is calculated by scikit-learn on at most 5,000 sampled fitting rows (seed 42). It is a descriptive clustering diagnostic, not a forecast score or a demonstration of spatial dependence. The configured cluster count and separate selection evidence should not be confused with this per-run silhouette calculation.

## 6. Learners and the library boundary

[`build_models`](../src/common_models.py) passes [`MODEL_PARAMS`](../src/common_config.py) to three scikit-learn regressors:

- Random forest: 150 trees, unrestricted maximum depth, seed 42.
- Gradient boosting: 150 estimators, maximum depth 3, learning rate 0.05, seed 42.
- Multilayer perceptron: hidden layers `(32, 16)`, ReLU, Adam, at most 300 iterations, seed 42.

The configuration file also specifies the remaining explicit settings. Unspecified options use the installed library's defaults, which is why the version matters. The MLP's internal early-stopping subset is a random 15% of the rows supplied to that fit, not the chronological ensemble-weight validation segment. Its tolerance is `1e−4` and patience is 20 iterations.

The actual tree construction, gradient updates, neural-network optimization and prediction operations occur inside the imported estimators' `.fit` and `.predict` methods. These are package-provided calculations, not hidden project-specific formulas.

## 7. Validation weights and the final forecast

[`compute_validation_weights`](../src/common_models.py) fits each learner on the earlier fitting segment and predicts the later validation segment. For learner `j`:

`E[j] = sum_i (y_validation[i] − prediction[j,i])² / N_validation`.

[`inverse_mse_weights`](../src/common_models.py) then calculates:

`a[j] = 1 / max(E[j], 1e−12)`

`w[j] = a[j] / sum_l a[l]`.

Non-finite validation errors are rejected. The floor avoids division by zero. Weights sum to one and are **frozen** after validation.

The pipeline then refits a fresh scaler, cluster model and set of learners on the complete pretest history. [`fit_predict_final_models`](../src/common_models.py) combines their final-test predictions using those already fixed weights:

`hybrid[i] = w[RF] × RF[i] + w[GBR] × GBR[i] + w[MLP] × MLP[i]`.

The saved ensemble-weight CSV contains both weights and validation MSEs. For any saved test row, its component predictions permit the three weighted contributions and final sum to be recalculated exactly, subject to CSV floating-point precision.

The primary runner returns fitted estimators in memory but does not archive serialized fitted objects. Saved forecasts support exact ensemble and metric arithmetic checks; tracing an individual random-forest leaf or MLP activation requires fitted objects from a rerun. A saved prediction alone does not contain those internal states.

## 8. Naïve references and scoring

[`add_baseline_predictions`](../src/common_features.py) sets persistence to `lag_1`, moving average to `rolling_mean_3`, and seasonal naïve to the within-series target shifted by 52 observations. The latter is not a same-calendar-week join. Classical ARIMA/SARIMA calculations and their state updates are implemented separately under [`benchmarks/sarima`](../benchmarks/sarima).

For paired observed and predicted values, define `e[i] = prediction[i] − observed[i]`. [`common_metrics.py`](../src/common_metrics.py) calculates:

| Quantity | Calculation and interpretation |
| --- | --- |
| MAE | `mean(abs(e))`, percentage points |
| RMSE | `sqrt(mean(e²))`, percentage points |
| R² | `1 − sum(e²)/sum((observed − mean(observed))²)` for nonconstant observed values; scikit-learn handles degenerate cases |
| Bias | `mean(e)`; positive means overprediction |
| SMAPE | `100 × mean(abs(e) / ((abs(observed)+abs(prediction))/2))`, excluding zero denominators; all-zero denominators produce missing SMAPE |
| Pearson | SciPy Pearson correlation coefficient; missing for fewer than two observations or a constant series |
| N | Number of paired observations actually scored |

`evaluate_predictions` drops missing observed/predicted pairs separately for each model. Consequently, equal-row comparisons must be verified using row keys and eligible-row masks, not assumed from the function name or matching rounded metrics. Report each method's N and any excluded rows. RMSE improvement relative to a reference is `100 × (RMSE_reference − RMSE_hybrid) / RMSE_reference` on the same cohort.

## 9. Figures, maps and peak summaries

### Primary NSSP scoring unit (reporting amendment, 15 November 2026)

The primary NSSP evaluation uses the existing `S3_nssp_geographic_metrics.csv` rows labeled `Equal HSA-week means`: 636 HSAs over 21 weeks (13,356 units). The archived sensitivity calculation first selects `trend_source == "HSA"`, verifies that observed targets agree within each `(hsa_nci_id, epi_order)` group, and averages targets and each model's predictions within those groups. The hybrid targets and forecasts are identical within each group, so the change removes repeated weighting rather than changing those forecasts. The same MAE, RMSE and R² formulas are then applied to all models on the same HSA-week units. Hybrid R² is 0.5986221211893501 and RMSE is 0.9724591726771514 percentage points.

The original mixed-source result (47,534 reporting records; hybrid R² 0.6712882288945727) remains the record-weighted comparison. HSA-source records before collapsing give R² 0.6693042512181708 across 46,515 records. Original archived labels such as `Primary mixed-source records` and `Inverse validation MSE (primary)` are retained unchanged; they describe the historical result archive, not the revised geographic reporting hierarchy. Training and validation still used the original record weighting. No model was refitted, no new forecasting experiment was run, and this post-review amendment is not represented as prespecified. Record-level ablations and ensemble-weight sensitivities remain record-level analyses.

### Descriptive displays

`plot_actual_vs_predictions` in [`common_plots.py`](../src/common_plots.py) takes arithmetic reporting-record means by date. `plot_residuals` computes row errors, weekly mean/median and empirical 5th–95th percentiles, plus a row-level histogram. The percentile band is descriptive dispersion, **not** a forecast prediction interval.

The final paper maps come from [`build_review_figures.py`](../manuscript/latex/build_review_figures.py), not the misleadingly named legacy `plot_state_choropleth`, which draws a ranked bar chart. The final map builder writes `map_values.csv` and `map_provenance.json`: each reported value, contributing record count, eligible-week count and geometry provenance can be inspected independently of the colors.

NSSP state colors average included reporting records, including repeated HSA assignments and available state aggregates; they are not population-weighted or official state rates. ILINet state colors average each state's eligible series over test weeks. New York City and territories remain separate companion-data units rather than being silently merged into state colors. Gray means no eligible data, not zero. The unchanged maps are descriptive mean predicted burdens, not validated hotspots or spatial-correlation evidence.

The [historical experiment runner](../experiments/experiment_3_ilinet_pandemic_stress_test.py) selects national rows for its weekly peak summary when present; otherwise it averages the retained reporting rows. It identifies the observed and predicted maxima separately, subtracts their dates for timing error, and subtracts their maximum values for magnitude error. These are not necessarily values at the same week. Before/after 11 March 2020 summaries are descriptive date partitions, not causal estimates of the pandemic declaration.

## 10. Minimum review of a reported point

1. Identify its source file, target, reporting key and source week.
2. Display the preceding records used by each lag and rolling statistic.
3. Confirm eligibility and the fitting, validation and test cutoffs.
4. Identify the local function, imported estimator and exact configuration producing each component prediction.
5. Recalculate the weights and weighted contributions from the saved validation errors and component predictions.
6. Recalculate its error and its contribution to the aggregate metric on the documented scoring cohort.
7. If it enters a curve or map, list the records and averaging rule used in that displayed aggregate.

This trace distinguishes values supplied by CDC, features calculated locally, estimates produced by third-party learners, and summaries derived from saved forecasts. It also makes clear which checks can be performed without fitting the models again.
