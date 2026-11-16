# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Fast no-training regression checks for the MMWR display-date correction."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import numpy as np
import pandas as pd
from typing import Iterable

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('patched_io', HERE/'code_patches/common_io.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
years=pd.Series([2025,2025,2026,2026,2019,2020], index=list('abcdef'))
weeks=pd.Series([1,53,1,53,40,39], index=list('abcdef'))
dates=module.epi_week_start(years,weeks)
assert dates.index.equals(years.index)
assert dates['a']==pd.Timestamp('2024-12-29')
assert dates['b']==pd.Timestamp('2025-12-28')
assert dates['c']==pd.Timestamp('2026-01-04')
assert pd.isna(dates['d'])
assert dates['e']==pd.Timestamp('2019-09-29')
assert dates['f']==pd.Timestamp('2020-09-20')
parser=argparse.ArgumentParser()
parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
root=parser.parse_args().project_root
name='experiment_2_ilinet_contemporary'
frame=pd.read_csv(root/'results/reference/hybrid'/name/'predictions'/f'{name}_predictions.csv')
keys=frame[['REGION TYPE','REGION','YEAR','WEEK','epi_order']].copy()
predictions=frame['Hybrid_InvMSE'].copy()
frame['date']=module.epi_week_start(frame.YEAR,frame.WEEK)
assert len(frame)==1774 and frame.date.nunique()==37 and frame.date.notna().all()
pd.testing.assert_frame_equal(frame[['REGION TYPE','REGION','YEAR','WEEK','epi_order']],keys)
pd.testing.assert_series_equal(frame.Hybrid_InvMSE,predictions)
print('PASS: CDC 2025-W53 retained; 37 weeks and 1774 scored rows; keys/predictions unchanged.')
for name in ['experiment_2_ilinet_contemporary','experiment_3_ilinet_pandemic_stress_test']:
    d=pd.read_csv(root/'results/reference/hybrid'/name/'predictions'/f'{name}_predictions.csv')
    d['correct_date']=module.epi_week_start(d.YEAR,d.WEEK)
    print(name, 'test Sunday starts',d.correct_date.min().date(),d.correct_date.max().date(), 'last Saturday',(d.correct_date.max()+pd.Timedelta(days=6)).date())

# Execute the unchanged feature/split functions directly from the repository's
# AST, avoiding an unnecessary scikit-learn dependency for a no-training audit.
# Calculation provenance:
#   importlib.util.module_from_spec -> importlib.util.util.module_from_spec (Python standard library)
#   importlib.util.spec_from_file_location -> importlib.util.util.spec_from_file_location (Python standard library)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

config=load_module('config',root/'src/common_config.py')
old_io=load_module('old_io',root/'src/common_io.py')
functions={'add_time_series_features','core_feature_columns','choose_feature_columns','chronological_split','explicit_epi_split'}
tree=ast.parse((root/'src/common_features.py').read_text(encoding='utf-8'))
selected=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in functions],type_ignores=[])
namespace=dict(pd=pd,np=np,Iterable=Iterable,common_config=config)
exec(compile(selected,'common_features.py','exec'),namespace)
cohorts=[]
for filename,expected_train,expected_test in [('ILINet.csv',7776,1774),('ILINet_2016_2021.csv',8262,2808)]:
    before=old_io.load_ilinet(root/'data/raw'/filename)
    after=module.load_ilinet(root/'data/raw'/filename)
    pd.testing.assert_frame_equal(before.drop(columns='date'),after.drop(columns='date'))
    kwargs=dict(target='%UNWEIGHTED ILI',group_cols=['REGION TYPE','REGION'],lagged_covariates={'NUM. OF PROVIDERS':[1]})
    old_features=namespace['add_time_series_features'](before,**kwargs)
    new_features=namespace['add_time_series_features'](after,**kwargs)
    pd.testing.assert_frame_equal(old_features.drop(columns='date'),new_features.drop(columns='date'))
    features=namespace['choose_feature_columns'](new_features,'%UNWEIGHTED ILI','ilinet')
    valid=new_features.dropna(subset=['%UNWEIGHTED ILI']+features)
    if filename=='ILINet.csv':
        train,test,_=namespace['chronological_split'](valid)
    else:
        train,test=namespace['explicit_epi_split'](valid,201640,201939,201940,202039)
    assert len(train)==expected_train and len(test)==expected_test
    row=dict(dataset=filename,train_rows=len(train),train_start=int(train.epi_order.min()),train_end=int(train.epi_order.max()),test_rows=len(test),test_start=int(test.epi_order.min()),test_end=int(test.epi_order.max()))
    cohorts.append(row)
    print('PASS: display-date change leaves every non-date raw/engineered value unchanged:',row)
(HERE/'calendar_cohort_audit.json').write_text(json.dumps(cohorts,indent=2),encoding='utf-8')

# Exercise the plotting-data path with a tiny plotting stub. The assertions
# concern denominators, not rendering; matplotlib need not be installed.
class Axis:
    def __init__(self):
        self.histogram_rows=None
        self.weekly_rows=None
    def hist(self,values,**kwargs):
        self.histogram_rows=len(values)
    def fill_between(self,dates,*args,**kwargs):
        self.weekly_rows=len(dates)
    def __getattr__(self,name):
        return lambda *args,**kwargs:None
class Plot:
    def subplots(self,*args,**kwargs):
        self.axes=[Axis(),Axis()]
        return None,self.axes
plot=Plot()
tree=ast.parse((HERE/'code_patches/common_plots.py').read_text(encoding='utf-8'))
chosen=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in {'_plot_dates','plot_residuals'}],type_ignores=[])
ns=dict(pd=pd,np=np,Path=Path,epi_week_start=module.epi_week_start,plt=plot,_save_figure=lambda p:None)
exec(compile(chosen,'common_plots.py','exec'),ns)
f=pd.DataFrame({'date':['2026-01-04',None,'2026-01-11'],'y':[1.,2.,3.],'Hybrid_InvMSE':[1.1,2.2,2.8]})
ns['plot_residuals'](f,'unused.png',target_col='y')
assert plot.axes[0].weekly_rows==2 and plot.axes[1].histogram_rows==3
f=pd.read_csv(root/'results/reference/hybrid/experiment_2_ilinet_contemporary/predictions/experiment_2_ilinet_contemporary_predictions.csv')
ns['plot_residuals'](f,'unused.png',target_col='%UNWEIGHTED ILI')
assert plot.axes[0].weekly_rows==37 and plot.axes[1].histogram_rows==1774
print('PASS: plotting retains all 37 contemporary ILINet weeks and all 1774 residuals; missing date never drops a histogram record.')
