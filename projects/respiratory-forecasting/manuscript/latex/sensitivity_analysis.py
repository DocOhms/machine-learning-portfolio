# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
# S1 copies archived development summaries; it does not repeat those model fits. S2/S3 recalculate weights and geographic-unit scores from saved predictions. Use --output-dir results/reproduced/sensitivity to retain reference evidence.
"""Reproduce the retrospective sensitivity supplement without fitting any model.

Inputs are the immutable reference predictions and copied historical development
summaries in sensitivity_data/source. All outputs are supplementary and never
replace primary results. Requires Python, NumPy and pandas only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPERIMENTS = [
    ('Experiment 1', 'experiment_1_nssp_primary', 'percent_visits_combined', ['geography', 'county', 'epi_order']),
    ('Experiment 2', 'experiment_2_ilinet_contemporary', '%UNWEIGHTED ILI', ['REGION TYPE', 'REGION', 'epi_order']),
    ('Experiment 3', 'experiment_3_ilinet_pandemic_stress_test', '%UNWEIGHTED ILI', ['REGION TYPE', 'REGION', 'epi_order']),
]
BASE_MODELS = ['RandomForest', 'GradientBoosting', 'MLP']
MODELS = [*BASE_MODELS, 'Hybrid_InvMSE', 'Persistence', 'MovingAverage3', 'SeasonalNaive', 'ARIMA', 'SARIMA']


# Calculation provenance:
#   np.abs -> numpy.abs (third-party package)
#   np.abs(e).mean -> numpy.abs(e).mean (third-party package)
#   np.asarray -> numpy.asarray (third-party package)
#   np.corrcoef -> numpy.corrcoef (third-party package)
#   np.count_nonzero -> numpy.count_nonzero (third-party package)
#   np.isfinite -> numpy.isfinite (third-party package)
#   np.isfinite(p).all -> numpy.isfinite(p).all (third-party package)
#   np.isfinite(y).all -> numpy.isfinite(y).all (third-party package)
#   np.mean -> numpy.mean (third-party package)
#   np.sqrt -> numpy.sqrt (third-party package)
#   np.sum -> numpy.sum (third-party package)
def metrics(actual, prediction) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Non-finite observations: do not silently change the scoring denominator.')
    e = p - y
    d = (np.abs(y) + np.abs(p)) / 2
    sst = np.sum((y - y.mean()) ** 2)
    return {'N': int(len(y)), 'MAE': float(np.abs(e).mean()), 'RMSE': float(np.sqrt(np.mean(e ** 2))),
            'R2': float(1 - np.sum(e ** 2) / sst), 'SMAPE': float(np.mean(np.abs(e[d != 0]) / d[d != 0]) * 100),
            'SMAPE_N': int(np.count_nonzero(d)),
            'Bias': float(e.mean()), 'Pearson': float(np.corrcoef(y, p)[0, 1])}


# Calculation provenance:
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Calculation provenance:
#   Path -> pathlib.Path (Python standard library)
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   json.dumps -> json.dumps (Python standard library)
#   metrics -> local function in this file
#   np.allclose -> numpy.allclose (third-party package)
#   np.isclose -> numpy.isclose (third-party package)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(ensemble).to_csv -> pandas.DataFrame(ensemble).to_csv (third-party package)
#   pd.DataFrame(ensemble)[['Experiment', 'Weighting', 'N', 'MAE', 'RMSE', 'R2']].to_string -> pandas.DataFrame(ensemble)[['Experiment', 'Weighting', 'N', 'MAE', 'RMSE', 'R2']].to_string (third-party package)
#   pd.DataFrame(scoring).query -> pandas.DataFrame(scoring).query (third-party package)
#   pd.DataFrame(scoring).query("Scoring_basis == 'Equal HSA-week means'")[['Model', 'N', 'MAE', 'RMSE', 'R2']].to_string -> pandas.DataFrame(scoring).query("Scoring_basis == 'Equal HSA-week means'")[['Model', 'N', 'MAE', 'RMSE', 'R2']].to_string (third-party package)
#   pd.DataFrame(scoring).to_csv -> pandas.DataFrame(scoring).to_csv (third-party package)
#   pd.DataFrame(sources).to_csv -> pandas.DataFrame(sources).to_csv (third-party package)
#   pd.DataFrame(weights).to_csv -> pandas.DataFrame(weights).to_csv (third-party package)
#   pd.concat -> pandas.concat (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   read -> local function in this file
#   sha256 -> local function in this file
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('sensitivity_data'))
    ap.add_argument('--source-dir', type=Path)
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir.resolve()
    source = args.source_dir.resolve() if args.source_dir else out / 'source'
    out.mkdir(parents=True, exist_ok=True)
    inputs = set()

    # Calculation provenance:
    #   pd.read_csv -> pandas.read_csv (third-party package)
    def read(path):
        inputs.add(path.resolve())
        return pd.read_csv(path)

    rolling = read(source / 'nssp_rolling_component_lags_summary.csv')
    rolling.to_csv(out / 'S1_nssp_rolling_validation.csv', index=False)
    folds = read(source / 'nssp_rolling_component_lags_folds.csv')
    folds.to_csv(out / 'S1_nssp_rolling_windows.csv', index=False)
    lag1 = read(source / 'nssp_locked_component_lags123_comparison_vs_lag1.csv')
    lag1.insert(0, 'Experiment', 'Experiment 1')
    lag1['Auxiliary_lags'] = lag1.Configuration.map({'Components_Lag1': '[1]', 'Components_Lag1_2_3': '[1,2,3]'})
    lag23 = read(source / 'ilinet_exp2_exp3_provider_lags_final_hybrid_comparison.csv')
    lag23['Experiment'] = lag23.Experiment.map({'Experiment_2_Contemporary': 'Experiment 2', 'Experiment_3_Pandemic': 'Experiment 3'})
    lag23['Auxiliary_lags'] = lag23.Configuration.map({'Provider_Lag1': '[1]', 'Provider_Lags1_2_3': '[1,2,3]'})
    cols = ['Experiment', 'Auxiliary_lags', 'N', 'MAE', 'RMSE', 'R2', 'SMAPE', 'Bias', 'Pearson']
    lag = pd.concat([lag1[cols], lag23[cols]], ignore_index=True)
    lag['Evidence_status'] = 'Historical development comparison; not newly trained or independently selected'
    lag.to_csv(out / 'S1_lag_test_comparison.csv', index=False)

    ensemble, weights = [], []
    nssp = None
    for label, name, target, keys in EXPERIMENTS:
        folder = repo / 'results/reference/hybrid' / name
        frame = read(folder / 'predictions' / f'{name}_predictions.csv')
        saved_metrics = read(folder / 'metrics' / f'{name}_metrics.csv')
        w = read(folder / 'metrics' / f'{name}_ensemble_weights.csv')
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate prediction keys in {name}')
        weight_map = w.set_index('Model')['Weight']
        reconstructed = sum(frame[m] * weight_map[m] for m in BASE_MODELS)
        if not np.allclose(reconstructed, frame['Hybrid_InvMSE'], atol=1e-12, rtol=1e-12):
            raise AssertionError('Saved inverse-MSE prediction reconstruction failed.')
        equal = frame[BASE_MODELS].mean(axis=1)
        for rule, pred in [('Inverse validation MSE (primary)', frame['Hybrid_InvMSE']), ('Equal weights (retrospective)', equal)]:
            score = metrics(frame[target], pred)
            ensemble.append({'Experiment': label, 'Weighting': rule, **score})
        ref = saved_metrics.set_index('Model').loc['Hybrid_InvMSE']
        primary = ensemble[-2]
        for k in ['N', 'MAE', 'RMSE', 'R2']:
            if not np.isclose(primary[k], ref[k], atol=1e-12, rtol=1e-12):
                raise AssertionError(f'Primary metric {k} changed for {label}')
        archived = lag[(lag.Experiment == label) & (lag.Auxiliary_lags == '[1]')].iloc[0]
        for k in ['N', 'MAE', 'RMSE', 'R2']:
            if not np.isclose(primary[k], archived[k], atol=1e-12, rtol=1e-12):
                raise AssertionError(f'Archived lag comparison differs for {label}: {k}')
        for row in w.to_dict('records'):
            weights.append({'Experiment': label, **row, 'Equal_weight': 1 / 3})
        if label == 'Experiment 1':
            nssp = frame

    pd.DataFrame(ensemble).to_csv(out / 'S2_ensemble_weighting_metrics.csv', index=False)
    pd.DataFrame(weights).to_csv(out / 'S2_ensemble_weights.csv', index=False)
    assert nssp is not None
    target = 'percent_visits_combined'
    keys = ['geography', 'county', 'epi_order']
    for name, rel in [('ARIMA', 'AR1'), ('SARIMA', 'SARIMA_100x100_52')]:
        classical = read(repo / 'results/reference/classical/nssp_full/test' / rel / 'predictions.csv')
        source_target = target + '_classical'
        classical = classical[keys + [target, 'prediction']].rename(columns={target: source_target, 'prediction': name})
        nssp = nssp.merge(classical, on=keys, how='left', validate='one_to_one', indicator=True)
        if not nssp['_merge'].eq('both').all() or len(nssp) != 47534:
            raise AssertionError('Classical predictions do not cover the exact primary rows.')
        if not np.allclose(nssp[target], nssp[source_target], atol=1e-12, rtol=0):
            raise AssertionError('Classical observed outcomes differ from primary outcomes.')
        nssp = nssp.drop(columns=[source_target, '_merge'])

    sources = []
    for source_name, sub in nssp.groupby('trend_source', dropna=False):
        sources.append({'Source': source_name, 'Test_records': len(sub),
                        'Geography_county_keys': len(sub[keys[:2]].drop_duplicates()),
                        'HSA_identifiers': sub['hsa_nci_id'].nunique() if source_name == 'HSA' else 0,
                        'Epidemiological_weeks': sub.epi_order.nunique(),
                        'Min_records_per_key': int(sub.groupby(keys[:2]).size().min()),
                        'Max_records_per_key': int(sub.groupby(keys[:2]).size().max())})
    pd.DataFrame(sources).to_csv(out / 'S3_nssp_source_counts.csv', index=False)
    hsa = nssp[nssp.trend_source.eq('HSA')].copy()
    if hsa.hsa_nci_id.isna().any():
        raise AssertionError('Missing HSA identifiers.')
    grouped = hsa.groupby(['hsa_nci_id', 'epi_order'], sort=True)
    target_spread = grouped[target].agg(lambda s: float(s.max() - s.min()))
    if not np.allclose(target_spread, 0, atol=1e-12, rtol=0):
        raise AssertionError('Targets conflict within an HSA-week.')
    aggregate = grouped[[target, *MODELS]].mean().reset_index()
    counts = grouped.size().rename('Source_records').reset_index()
    aggregate = aggregate.merge(counts, on=['hsa_nci_id', 'epi_order'], validate='one_to_one')
    aggregate.to_csv(out / 'S3_hsa_week_mean_predictions.csv', index=False)
    scoring = []
    cohorts = [('Primary mixed-source records', nssp), ('HSA-source county-key records', hsa),
               ('Equal HSA-week means', aggregate)]
    for cohort, data in cohorts:
        for model in MODELS:
            scoring.append({'Scoring_basis': cohort, 'Model': model, **metrics(data[target], data[model])})
    pd.DataFrame(scoring).to_csv(out / 'S3_nssp_geographic_metrics.csv', index=False)
    duplicate = grouped.size()
    summary = {
        'primary_records': len(nssp), 'primary_geography_county_keys': len(nssp[keys[:2]].drop_duplicates()),
        'hsa_source_records': len(hsa), 'hsa_county_keys': len(hsa[keys[:2]].drop_duplicates()),
        'unique_hsa_ids': int(hsa.hsa_nci_id.nunique()), 'unique_hsa_weeks': len(aggregate),
        'all_aggregate_records': int(nssp.county.eq('All').sum()),
        'all_aggregate_keys': len(nssp.loc[nssp.county.eq('All'), keys[:2]].drop_duplicates()),
        'hsa_weeks_with_repeated_records': int((duplicate > 1).sum()),
        'records_in_repeated_hsa_weeks': int(duplicate[duplicate > 1].sum()),
        'hsa_weeks_with_target_disagreement': int((target_spread > 1e-12).sum()),
        'hsa_weeks_with_hybrid_prediction_spread_above_1e_12': int((grouped.Hybrid_InvMSE.max() - grouped.Hybrid_InvMSE.min() > 1e-12).sum()),
        'test_epi_start': int(nssp.epi_order.min()), 'test_epi_end': int(nssp.epi_order.max()),
        'training_performed': False, 'primary_results_replaced': False,
        'inference': 'Descriptive sensitivities; no independent-sample or significance claim.'}
    (out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    manifest = []
    for p in sorted(inputs):
        try:
            name = 'repo/' + p.relative_to(repo).as_posix()
        except ValueError:
            name = 'source/' + p.name
        manifest.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(ensemble)[['Experiment', 'Weighting', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))
    print(pd.DataFrame(scoring).query("Scoring_basis == 'Equal HSA-week means'")[['Model', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))


if __name__ == '__main__':
    main()
