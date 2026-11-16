# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
import ast
import argparse
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Reconstruct test cohorts and features without fitting models.')
parser.add_argument('--repo', required=True, type=Path)
ROOT = parser.parse_args().repo.resolve()
sys.path.insert(0, str(ROOT / 'src'))
import common_config
from common_io import load_nssp, load_ilinet

# Execute the shipped pure data functions without importing its unused sklearn
# preprocessing classes; no replacement implementation and no model fitting.
source = ast.parse((ROOT/'src/common_features.py').read_text(encoding='utf-8'))
names = {'add_time_series_features', 'core_feature_columns', 'choose_feature_columns',
         'chronological_split', 'chronological_validation_split', 'explicit_epi_split'}
module = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
ns = {'np':np,'pd':pd,'common_config':common_config,'Iterable':Iterable}
exec(compile(module, str(ROOT/'src/common_features.py'), 'exec'),ns)

cases=[('experiment_1_nssp_primary','NSSP.csv','nssp','percent_visits_combined',['geography','county'],common_config.NSSP_COMPONENT_COLUMNS),
       ('experiment_2_ilinet_contemporary','ILINet.csv','ilinet','%UNWEIGHTED ILI',['REGION TYPE','REGION'],common_config.ILINET_REPORTING_COLUMNS),
       ('experiment_3_ilinet_pandemic_stress_test','ILINet_2016_2021.csv','ilinet','%UNWEIGHTED ILI',['REGION TYPE','REGION'],common_config.ILINET_REPORTING_COLUMNS)]
for name,file,kind,target,groups,aux in cases:
    df=(load_nssp if kind=='nssp' else load_ilinet)(ROOT/'data/raw'/file,target=target)
    en=ns['add_time_series_features'](df,target,groups,lagged_covariates={a:[1] for a in aux})
    features=ns['choose_feature_columns'](en,target,kind)
    model=en.dropna(subset=[target]+features).copy()
    if 'pandemic' in name:
        train,test=ns['explicit_epi_split'](model,201640,201939,201940,202039)
    else:
        train,test,_=ns['chronological_split'](model)
    fit,val,_=ns['chronological_validation_split'](train)
    saved=pd.read_csv(ROOT/'results/reference/hybrid'/name/'predictions'/f'{name}_predictions.csv',low_memory=False)
    metrics=pd.read_csv(ROOT/'results/reference/hybrid'/name/'metrics'/f'{name}_metrics.csv')
    keys=groups+['epi_order']
    assert not test.duplicated(keys).any()
    assert not saved.duplicated(keys).any()
    match=test[keys+[target]+features+['seasonal_naive']].merge(saved, on=keys, how='outer',suffixes=('_fresh','_saved'),indicator=True,validate='one_to_one')
    assert match['_merge'].eq('both').all(),match['_merge'].value_counts()
    maxdiff=0
    for col in [target]+features+['seasonal_naive']:
        a=match[col+'_fresh'].to_numpy(float);b=match[col+'_saved'].to_numpy(float)
        assert np.allclose(a,b,atol=1e-12,rtol=1e-12,equal_nan=True),(name,col)
        if np.isfinite(a-b).any():maxdiff=max(maxdiff,float(np.nanmax(np.abs(a-b))))
    for m in metrics.itertuples(index=False):
        y=saved[target].to_numpy(float);p=saved[m.Model].to_numpy(float)
        mask=np.isfinite(y)&np.isfinite(p);y=y[mask];p=p[mask];e=p-y
        assert len(y)==m.N
        vals={'MAE':np.mean(np.abs(e)),'RMSE':np.sqrt(np.mean(e**2)),'R2':1-np.sum(e**2)/np.sum((y-y.mean())**2),'Bias':np.mean(e),'Pearson':np.corrcoef(y,p)[0,1]}
        den=(np.abs(y)+np.abs(p))/2
        vals['SMAPE']=np.mean(np.abs(e[den!=0])/den[den!=0])*100
        for field,v in vals.items():assert np.isclose(v,getattr(m,field),atol=1e-11,rtol=1e-11),(name,m.Model,field,v,getattr(m,field))
    assert np.allclose(saved.Persistence,saved.lag_1)
    assert np.allclose(saved.MovingAverage3,saved.rolling_mean_3)
    assert np.allclose(saved.SeasonalNaive,saved.seasonal_naive,equal_nan=True)
    print(name)
    print('  raw / target-valid / model / fit / val / train / test:',len(df),df[target].notna().sum(),len(model),len(fit),len(val),len(train),len(test))
    print('  weeks fit / val / train / test:',*[x.epi_order.nunique() for x in [fit,val,train,test]])
    print('  ranges train / test:',train.epi_order.min(),train.epi_order.max(),test.epi_order.min(),test.epi_order.max())
    print('  exact matching test keys, target, features, seasonal baseline; max numeric difference:',maxdiff)
    print('  recomputed metrics match all',len(metrics),'models; N values:',metrics.N.unique().tolist())
    print('  test missing display dates:',test.date.isna().sum())
print('PASS all cohort, feature, baseline and reference-metric checks; no model training performed.')
