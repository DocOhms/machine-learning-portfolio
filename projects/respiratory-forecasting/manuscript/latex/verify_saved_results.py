# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Read-only arithmetic and exact-row audit; does not fit any models."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True, type=Path)
ROOT = parser.parse_args().repo.resolve()
HYBRID = ROOT / 'results/reference/hybrid'
CLASSICAL = ROOT / 'results/reference/classical'
EXPERIMENTS = [
    ('experiment_1_nssp_primary', 'percent_visits_combined', ['geography', 'county', 'epi_order'],
     'nssp_full/test/AR1/predictions.csv', 'nssp_full/test/SARIMA_100x100_52/predictions.csv'),
    ('experiment_2_ilinet_contemporary', '%UNWEIGHTED ILI', ['REGION TYPE', 'REGION', 'epi_order'],
     'arima_extension_exp2_exp3/exp2/final_test_predictions.csv', 'exp2/final_test_predictions.csv'),
    ('experiment_3_ilinet_pandemic_stress_test', '%UNWEIGHTED ILI', ['REGION TYPE', 'REGION', 'epi_order'],
     'arima_extension_exp2_exp3/exp3/final_test_predictions.csv', 'exp3/final_test_predictions.csv'),
]

# Calculation provenance:
#   np.abs -> numpy.abs (third-party package)
#   np.asarray -> numpy.asarray (third-party package)
#   np.corrcoef -> numpy.corrcoef (third-party package)
#   np.isfinite -> numpy.isfinite (third-party package)
#   np.mean -> numpy.mean (third-party package)
#   np.sqrt -> numpy.sqrt (third-party package)
#   np.sum -> numpy.sum (third-party package)
def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    e = p-y
    den = (np.abs(y)+np.abs(p))/2
    return dict(N=len(y), MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)),
                R2=1-np.sum(e**2)/np.sum((y-y.mean())**2),
                Bias=e.mean(), SMAPE=100*np.mean(np.abs(e[den!=0])/den[den!=0]),
                Pearson=np.corrcoef(y,p)[0,1])

for name, target, keys, arima_path, sarima_path in EXPERIMENTS:
    folder = HYBRID / name
    pred = pd.read_csv(folder/'predictions'/f'{name}_predictions.csv', low_memory=False)
    metrics = pd.read_csv(folder/'metrics'/f'{name}_metrics.csv')
    weights = pd.read_csv(folder/'metrics'/f'{name}_ensemble_weights.csv')
    assert not pred.duplicated(keys).any()
    inv = 1/np.maximum(weights.Validation_MSE.to_numpy(), 1e-12)
    np.testing.assert_allclose(weights.Weight, inv/inv.sum(), rtol=1e-12, atol=1e-12)
    hybrid = sum(row.Weight*pred[row.Model] for row in weights.itertuples())
    maxdiff = np.max(np.abs(hybrid-pred.Hybrid_InvMSE))
    np.testing.assert_allclose(hybrid, pred.Hybrid_InvMSE, rtol=1e-12, atol=1e-12)
    for row in metrics.to_dict('records'):
        actual = score(pred[target], pred[row['Model']])
        for field, value in actual.items():
            np.testing.assert_allclose(value, row[field], rtol=1e-11, atol=1e-11)
        assert actual['N'] == len(pred)
    print(name, 'PASS: every stored model metric; weights; ensemble identity; all model N identical', 'max_prediction_difference=', maxdiff)
    print('HYBRID', score(pred[target],pred.Hybrid_InvMSE))
    for label, path in [('ARIMA', arima_path), ('SARIMA', sarima_path)]:
        c = pd.read_csv(CLASSICAL/path, low_memory=False)
        assert not c.duplicated(keys).any()
        j = pred[keys+[target]].merge(c, on=keys, how='outer', validate='one_to_one', indicator=True, suffixes=('_hybrid','_classical'))
        assert j['_merge'].eq('both').all() and len(j)==len(pred)
        np.testing.assert_allclose(j[target+'_hybrid'],j[target+'_classical'],rtol=0,atol=1e-12)
        assert j.prediction.notna().all()
        print(label, 'PASS: exact keys/targets and full coverage', score(j[target+'_hybrid'],j.prediction))

print('READ-ONLY AUDIT PASSED; no model training or output changes')
