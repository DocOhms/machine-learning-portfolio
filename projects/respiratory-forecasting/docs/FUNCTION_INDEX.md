# Function and imported-calculation index

Generated from the public review source. Links open the exact local implementation. Imported libraries retain their own implementations; requirements.txt records the declared versions. Object methods such as DataFrame.groupby and estimator.fit are explained in CALCULATIONS.md and module headers.

## benchmarks/sarima/finalize_arima_extension.py

- [main](../benchmarks/sarima/finalize_arima_extension.py#L22)

## benchmarks/sarima/finalize_nssp_full.py

- [sha256](../benchmarks/sarima/finalize_nssp_full.py#L22)
- [main](../benchmarks/sarima/finalize_nssp_full.py#L35)

## benchmarks/sarima/nssp_full_runner.py

- [candidate_by_name](../benchmarks/sarima/nssp_full_runner.py#L25)
- [load_experiment](../benchmarks/sarima/nssp_full_runner.py#L36)
- [group_keys](../benchmarks/sarima/nssp_full_runner.py#L42)
- [score_summary](../benchmarks/sarima/nssp_full_runner.py#L48)
- [run_batched](../benchmarks/sarima/nssp_full_runner.py#L84)
- [collect_summaries](../benchmarks/sarima/nssp_full_runner.py#L180)
- [methodological_decision](../benchmarks/sarima/nssp_full_runner.py#L199)
- [select_candidate](../benchmarks/sarima/nssp_full_runner.py#L257)
- [final_comparison](../benchmarks/sarima/nssp_full_runner.py#L282)
- [main](../benchmarks/sarima/nssp_full_runner.py#L331)

## benchmarks/sarima/run_ilinet_arima_extension.py

- [select_nonseasonal_candidate](../benchmarks/sarima/run_ilinet_arima_extension.py#L44)
- [verify_locked_input_hashes](../benchmarks/sarima/run_ilinet_arima_extension.py#L70)
- [load_locked_comparison](../benchmarks/sarima/run_ilinet_arima_extension.py#L106)
- [main](../benchmarks/sarima/run_ilinet_arima_extension.py#L193)

## benchmarks/sarima/sarima_pilot.py

- [clean_numeric](../benchmarks/sarima/sarima_pilot.py#L73)
- [file_sha256](../benchmarks/sarima/sarima_pilot.py#L79)
- [load_raw](../benchmarks/sarima/sarima_pilot.py#L91)
- [add_cohort_features](../benchmarks/sarima/sarima_pilot.py#L124)
- [locked_paths](../benchmarks/sarima/sarima_pilot.py#L155)
- [key_tuples](../benchmarks/sarima/sarima_pilot.py#L163)
- [reproduce_cohorts](../benchmarks/sarima/sarima_pilot.py#L174)
- [choose_nssp_groups](../benchmarks/sarima/sarima_pilot.py#L214)
- [subset_groups](../benchmarks/sarima/sarima_pilot.py#L238)
- [metrics](../benchmarks/sarima/sarima_pilot.py#L250)
- [fit_and_predict_group](../benchmarks/sarima/sarima_pilot.py#L274)
- [evaluate_candidate](../benchmarks/sarima/sarima_pilot.py#L335)
- [run_experiment](../benchmarks/sarima/sarima_pilot.py#L386)
- [write_manifest](../benchmarks/sarima/sarima_pilot.py#L464)
- [main](../benchmarks/sarima/sarima_pilot.py#L489)

## benchmarks/sarima/summarize_pilot.py

- [score](../benchmarks/sarima/summarize_pilot.py#L35)
- [main](../benchmarks/sarima/summarize_pilot.py#L59)

## benchmarks/sarima/validate_ilinet_arima_extension.py

- [close](../benchmarks/sarima/validate_ilinet_arima_extension.py#L24)
- [main](../benchmarks/sarima/validate_ilinet_arima_extension.py#L34)

## benchmarks/sarima/validate_pilot.py

- [sha256](../benchmarks/sarima/validate_pilot.py#L19)
- [main](../benchmarks/sarima/validate_pilot.py#L32)

## experiments/_run_template.py

- [_validate_columns](../experiments/_run_template.py#L40)
- [run_experiment](../experiments/_run_template.py#L67)

## experiments/diagnostic_feature_importance_nssp.py

- [main](../experiments/diagnostic_feature_importance_nssp.py#L30)

## experiments/diagnostic_leakage_test_nssp.py

- [_hybrid_row](../experiments/diagnostic_leakage_test_nssp.py#L45)
- [main](../experiments/diagnostic_leakage_test_nssp.py#L58)

## experiments/diagnostic_n_clusters_sensitivity.py

- [_model_row](../experiments/diagnostic_n_clusters_sensitivity.py#L24)
- [main](../experiments/diagnostic_n_clusters_sensitivity.py#L37)

## experiments/experiment_1_nssp_ablation_core_features_only.py

- [main](../experiments/experiment_1_nssp_ablation_core_features_only.py#L33)

## experiments/experiment_1_nssp_ablation_no_clustering.py

- [main](../experiments/experiment_1_nssp_ablation_no_clustering.py#L24)

## experiments/experiment_1_nssp_primary.py

- [main](../experiments/experiment_1_nssp_primary.py#L24)

## experiments/experiment_2_ilinet_contemporary.py

- [main](../experiments/experiment_2_ilinet_contemporary.py#L24)

## experiments/experiment_3_ilinet_pandemic_stress_test.py

- [_national_weekly_series](../experiments/experiment_3_ilinet_pandemic_stress_test.py#L25)
- [save_stress_test_outputs](../experiments/experiment_3_ilinet_pandemic_stress_test.py#L49)
- [main](../experiments/experiment_3_ilinet_pandemic_stress_test.py#L114)

## generate_choropleth_maps.py

- [get_location_column](../generate_choropleth_maps.py#L38)
- [normalize_state](../generate_choropleth_maps.py#L47)
- [aggregate_to_state](../generate_choropleth_maps.py#L58)
- [save_map](../generate_choropleth_maps.py#L84)
- [process_experiment](../generate_choropleth_maps.py#L126)
- [main](../generate_choropleth_maps.py#L168)

## manuscript/latex/build_calendar_figures.py

- [line](../manuscript/latex/build_calendar_figures.py#L18)
- [path](../manuscript/latex/build_calendar_figures.py#L21)
- [svg_start](../manuscript/latex/build_calendar_figures.py#L26)
- [axes](../manuscript/latex/build_calendar_figures.py#L33)
- [trajectory](../manuscript/latex/build_calendar_figures.py#L53)
- [residuals](../manuscript/latex/build_calendar_figures.py#L81)
- [main](../manuscript/latex/build_calendar_figures.py#L128)

## manuscript/latex/build_framework_vector.py

- [make_figure](../manuscript/latex/build_framework_vector.py#L31)
- [box](../manuscript/latex/build_framework_vector.py#L48)
- [text](../manuscript/latex/build_framework_vector.py#L55)
- [lines](../manuscript/latex/build_framework_vector.py#L62)
- [segment](../manuscript/latex/build_framework_vector.py#L68)
- [arrow](../manuscript/latex/build_framework_vector.py#L75)

## manuscript/latex/build_review_figures.py

- [interpolate](../manuscript/latex/build_review_figures.py#L28)
- [state_paths](../manuscript/latex/build_review_figures.py#L36)
- [text](../manuscript/latex/build_review_figures.py#L64)
- [build_map](../manuscript/latex/build_review_figures.py#L72)
- [main](../manuscript/latex/build_review_figures.py#L129)

## manuscript/latex/code_patches/common_io.py

- [clean_numeric](../manuscript/latex/code_patches/common_io.py#L13)
- [epi_week_start](../manuscript/latex/code_patches/common_io.py#L25)
- [load_nssp](../manuscript/latex/code_patches/common_io.py#L54)
- [load_ilinet](../manuscript/latex/code_patches/common_io.py#L81)
- [ensure_dirs](../manuscript/latex/code_patches/common_io.py#L105)

## manuscript/latex/code_patches/common_plots.py

- [_save_figure](../manuscript/latex/code_patches/common_plots.py#L34)
- [_plot_dates](../manuscript/latex/code_patches/common_plots.py#L45)
- [plot_actual_vs_predictions](../manuscript/latex/code_patches/common_plots.py#L62)
- [plot_metric_bar](../manuscript/latex/code_patches/common_plots.py#L112)
- [plot_residuals](../manuscript/latex/code_patches/common_plots.py#L144)
- [plot_state_choropleth](../manuscript/latex/code_patches/common_plots.py#L214)

## manuscript/latex/manuscript_kmeans_diagnostics.py

- [prepare_frames](../manuscript/latex/manuscript_kmeans_diagnostics.py#L57)
- [candidate_metrics](../manuscript/latex/manuscript_kmeans_diagnostics.py#L82)
- [final_profiles](../manuscript/latex/manuscript_kmeans_diagnostics.py#L128)
- [make_figure](../manuscript/latex/manuscript_kmeans_diagnostics.py#L172)
- [main](../manuscript/latex/manuscript_kmeans_diagnostics.py#L273)

## manuscript/latex/sensitivity_analysis.py

- [metrics](../manuscript/latex/sensitivity_analysis.py#L40)
- [sha256](../manuscript/latex/sensitivity_analysis.py#L56)
- [main](../manuscript/latex/sensitivity_analysis.py#L83)
- [read](../manuscript/latex/sensitivity_analysis.py#L97)

## manuscript/latex/test_calendar_correction.py

- [load_module](../manuscript/latex/test_calendar_correction.py#L50)
- [__init__](../manuscript/latex/test_calendar_correction.py#L87)
- [hist](../manuscript/latex/test_calendar_correction.py#L90)
- [fill_between](../manuscript/latex/test_calendar_correction.py#L92)
- [__getattr__](../manuscript/latex/test_calendar_correction.py#L94)
- [subplots](../manuscript/latex/test_calendar_correction.py#L97)

## manuscript/latex/verify_model_cohorts.py


## manuscript/latex/verify_saved_results.py

- [score](../manuscript/latex/verify_saved_results.py#L32)

## scripts/create_repository_manifest.py

- [sha256](../scripts/create_repository_manifest.py#L25)
- [main](../scripts/create_repository_manifest.py#L35)

## scripts/run_pipeline.py

- [main](../scripts/run_pipeline.py#L35)

## scripts/stage_data.py

- [sha256](../scripts/stage_data.py#L21)
- [resolve_source](../scripts/stage_data.py#L29)
- [main](../scripts/stage_data.py#L45)

## scripts/verify_data.py

- [sha256](../scripts/verify_data.py#L19)
- [main](../scripts/verify_data.py#L30)

## scripts/verify_reference_results.py

- [main](../scripts/verify_reference_results.py#L44)

## src/common_config.py


## src/common_features.py

- [add_time_series_features](../src/common_features.py#L37)
- [core_feature_columns](../src/common_features.py#L91)
- [choose_feature_columns](../src/common_features.py#L104)
- [_validate_feature_columns](../src/common_features.py#L135)
- [chronological_split](../src/common_features.py#L146)
- [chronological_validation_split](../src/common_features.py#L174)
- [explicit_epi_split](../src/common_features.py#L203)
- [_append_cluster_indicators](../src/common_features.py#L219)
- [fit_preprocessor](../src/common_features.py#L238)
- [transform_preprocessor](../src/common_features.py#L309)
- [fit_transform_preprocessor](../src/common_features.py#L334)
- [add_baseline_predictions](../src/common_features.py#L361)

## src/common_io.py

- [clean_numeric](../src/common_io.py#L13)
- [epi_week_start](../src/common_io.py#L25)
- [load_nssp](../src/common_io.py#L54)
- [load_ilinet](../src/common_io.py#L81)
- [ensure_dirs](../src/common_io.py#L105)

## src/common_metrics.py

- [smape](../src/common_metrics.py#L17)
- [bias](../src/common_metrics.py#L30)
- [pearson_corr](../src/common_metrics.py#L38)
- [evaluate_predictions](../src/common_metrics.py#L57)

## src/common_models.py

- [build_models](../src/common_models.py#L20)
- [inverse_mse_weights](../src/common_models.py#L30)
- [compute_validation_weights](../src/common_models.py#L53)
- [fit_predict_final_models](../src/common_models.py#L75)
- [train_predict_models](../src/common_models.py#L97)

## src/common_plots.py

- [_save_figure](../src/common_plots.py#L34)
- [_plot_dates](../src/common_plots.py#L45)
- [plot_actual_vs_predictions](../src/common_plots.py#L62)
- [plot_metric_bar](../src/common_plots.py#L112)
- [plot_residuals](../src/common_plots.py#L144)
- [plot_state_choropleth](../src/common_plots.py#L214)

## tests/test_mmwr_calendar.py

- [test_week_53_and_year_boundary](../tests/test_mmwr_calendar.py#L20)
- [test_index_and_invalid_values](../tests/test_mmwr_calendar.py#L33)

## tests/test_repository.py

- [test_target_and_covariate_features_are_lagged](../tests/test_repository.py#L26)
- [test_chronological_split_keeps_weeks_disjoint](../tests/test_repository.py#L55)
- [test_data_manifest_has_expected_files](../tests/test_repository.py#L71)
- [test_primary_reference_metrics_are_preserved](../tests/test_repository.py#L84)
- [test_metric_evaluation_orders_by_rmse](../tests/test_repository.py#L108)

# Archived development calculation sources

Historical code: not the primary-model execution path. See archive/development/README.md before rerunning.

## archive/development/legacy_project/src/common_config.py


## archive/development/legacy_project/src/common_features.py

- [add_time_series_features](../archive/development/legacy_project/src/common_features.py#L29)
- [core_feature_columns](../archive/development/legacy_project/src/common_features.py#L83)
- [choose_feature_columns](../archive/development/legacy_project/src/common_features.py#L94)
- [_validate_feature_columns](../archive/development/legacy_project/src/common_features.py#L125)
- [chronological_split](../archive/development/legacy_project/src/common_features.py#L133)
- [chronological_validation_split](../archive/development/legacy_project/src/common_features.py#L158)
- [explicit_epi_split](../archive/development/legacy_project/src/common_features.py#L187)
- [_append_cluster_indicators](../archive/development/legacy_project/src/common_features.py#L203)
- [fit_preprocessor](../archive/development/legacy_project/src/common_features.py#L214)
- [transform_preprocessor](../archive/development/legacy_project/src/common_features.py#L281)
- [fit_transform_preprocessor](../archive/development/legacy_project/src/common_features.py#L303)
- [add_baseline_predictions](../archive/development/legacy_project/src/common_features.py#L330)

## archive/development/legacy_project/src/common_io.py

- [clean_numeric](../archive/development/legacy_project/src/common_io.py#L8)
- [epi_week_start](../archive/development/legacy_project/src/common_io.py#L16)
- [load_nssp](../archive/development/legacy_project/src/common_io.py#L31)
- [load_ilinet](../archive/development/legacy_project/src/common_io.py#L54)
- [ensure_dirs](../archive/development/legacy_project/src/common_io.py#L76)

## archive/development/legacy_project/src/common_metrics.py

- [smape](../archive/development/legacy_project/src/common_metrics.py#L9)
- [bias](../archive/development/legacy_project/src/common_metrics.py#L19)
- [pearson_corr](../archive/development/legacy_project/src/common_metrics.py#L23)
- [evaluate_predictions](../archive/development/legacy_project/src/common_metrics.py#L31)

## archive/development/legacy_project/src/common_models.py

- [build_models](../archive/development/legacy_project/src/common_models.py#L12)
- [inverse_mse_weights](../archive/development/legacy_project/src/common_models.py#L20)
- [compute_validation_weights](../archive/development/legacy_project/src/common_models.py#L38)
- [fit_predict_final_models](../archive/development/legacy_project/src/common_models.py#L57)
- [train_predict_models](../archive/development/legacy_project/src/common_models.py#L76)

## archive/development/package_archival_evidence.py

- [sha256](../archive/development/package_archival_evidence.py#L15)
- [main](../archive/development/package_archival_evidence.py#L19)
- [copy](../archive/development/package_archival_evidence.py#L28)

## archive/development/run_final_lags123.py

- [main](../archive/development/run_final_lags123.py#L45)

## archive/development/run_ilinet_exp2_exp3_provider_lags.py

- [hybrid_row](../archive/development/run_ilinet_exp2_exp3_provider_lags.py#L66)
- [main](../archive/development/run_ilinet_exp2_exp3_provider_lags.py#L70)

## archive/development/run_rolling_component_lags.py

- [save_rows](../archive/development/run_rolling_component_lags.py#L48)
- [main](../archive/development/run_rolling_component_lags.py#L52)

## archive/development/verify_archival_adaptation.py

- [digest](../archive/development/verify_archival_adaptation.py#L11)
- [main](../archive/development/verify_archival_adaptation.py#L15)

