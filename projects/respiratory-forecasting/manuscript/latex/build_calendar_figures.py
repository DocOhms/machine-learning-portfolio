# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Re-label saved ILINet displays using source epiweeks; never refit models."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from build_review_figures import text, DEFAULT_ROOT, HERE

spec=importlib.util.spec_from_file_location('patched_io',HERE/'code_patches/common_io.py')
io=importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)

def line(x1,y1,x2,y2,color='#ddd',width=1,dash=''):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'

def path(points,color,width=2,dash='',fill='none'):
    return f'<path d="M'+ ' L'.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'

# Calculation provenance:
#   text -> build_review_figures.text (manuscript/latex/build_review_figures.py)
def svg_start(title,height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">', f'<rect width="1200" height="{height}" fill="white"/>','<g font-family="Arial, Helvetica, sans-serif">',text(600,35,title,26,anchor='middle',weight='bold')]

# Calculation provenance:
#   line -> local function in this file
#   np.linspace -> numpy.linspace (third-party package)
#   text -> build_review_figures.text (manuscript/latex/build_review_figures.py)
def axes(svg,w,top,bottom,ymin,ymax,ticks):
    xx=lambda k:95+1045*k/(len(w)-1)
    yy=lambda v:bottom-(bottom-top)*(v-ymin)/(ymax-ymin)
    for v in ticks:
        svg.append(line(95,yy(v),1140,yy(v)))
        svg.append(text(82,yy(v)+6,f'{v:g}',18,anchor='end'))
    for k in np.linspace(0,len(w)-1,8,dtype=int):
        svg.append(line(xx(k),top,xx(k),bottom))
        r=w.iloc[k]
        svg.append(text(xx(k),bottom+27,f'{int(r.YEAR)}-W{int(r.WEEK):02d}',16,anchor='middle'))
    svg.append(line(95,top,95,bottom,'#333'))
    svg.append(line(95,bottom,1140,bottom,'#333'))
    return xx,yy

# Calculation provenance:
#   axes -> local function in this file
#   line -> local function in this file
#   path -> local function in this file
#   svg_start -> local function in this file
#   text -> build_review_figures.text (manuscript/latex/build_review_figures.py)
def trajectory(w):
    svg=svg_start('Experiment 3 | Historical ILINet forecast trajectory',680)
    svg.append(text(600,65,'Unweighted mean across the included reporting-unit series',19,anchor='middle'))
    xx,yy=axes(svg,w,100,570,0,7,range(8))
    colors=['#1565a9','#e37400','#22823b','#c62828','#8b62b3']
    cols=['%UNWEIGHTED ILI','Hybrid_InvMSE','RandomForest','GradientBoosting','MLP']
    labels=['Observed','Hybrid ensemble','Random Forest','Gradient Boosting','MLP']
    for col,color in zip(cols,colors):
        svg.append(path([(xx(k),yy(v)) for k,v in enumerate(w[col])],color,3 if col in cols[:2] else 2))
    for k,(label,color) in enumerate(zip(labels,colors)):
        xpos=[110,295,550,785,1060][k]
        svg.append(line(xpos,85,xpos+25,85,color,3))
        svg.append(text(xpos+31,91,label,17))
    svg.append('<text transform="translate(28,335) rotate(-90)" font-size="20" text-anchor="middle">Unweighted ILI (% of outpatient visits)</text>')
    svg.append(text(600,630,'CDC epidemiological week (YEAR-WEEK as reported)',19,anchor='middle'))
    svg.append(text(600,661,'52 test weeks and 2,808 scored records; not a population-weighted national estimate.',17,anchor='middle'))
    return '\n'.join(svg+['</g></svg>'])

# Calculation provenance:
#   axes -> local function in this file
#   line -> local function in this file
#   np.arange -> numpy.arange (third-party package)
#   np.ceil -> numpy.ceil (third-party package)
#   np.histogram -> numpy.histogram (third-party package)
#   path -> local function in this file
#   pd.Timestamp -> pandas.Timestamp (third-party package)
#   svg_start -> local function in this file
#   text -> build_review_figures.text (manuscript/latex/build_review_figures.py)
def residuals(d,w):
    svg=svg_start('Experiment 3 | Historical ILINet residuals',850)
    svg.append(text(600,66,'Residual = predicted minus observed unweighted ILI (percentage points)',19,anchor='middle'))
    xx,yy=axes(svg,w,125,470,-4.5,4.5,range(-4,5))
    points=[(xx(k),yy(v)) for k,v in enumerate(w.p05)]+[(xx(k),yy(v)) for k,v in reversed(list(enumerate(w.p95)))]
    svg.append(path(points,'none',0,fill='#d5e7f3'))
    svg.append(line(95,yy(0),1140,yy(0),'#333',1.4,'5,4'))
    svg.append(path([(xx(k),yy(v)) for k,v in enumerate(w['median'])],'#1565a9',2.5))
    svg.append(path([(xx(k),yy(v)) for k,v in enumerate(w['mean'])],'#e37400',2,'7,4'))
    event_x=xx((pd.Timestamp('2020-03-11')-w.date.min()).days/7)
    svg.append(line(event_x,125,event_x,470,'#111',1.8,'2,5'))
    legend=[('5th-95th percentile','#d5e7f3',''),('Weekly median','#1565a9',''),('Weekly mean','#e37400','7,4'),('WHO declaration (March 11)','#111','2,5')]
    for x,(label,color,dash) in zip([100,380,620,835],legend):
        svg.append(line(x,99,x+27,99,color,7 if label.startswith('5th') else 2.5,dash))
        svg.append(text(x+34,105,label,17))
    svg.append('<text transform="translate(28,300) rotate(-90)" font-size="20" text-anchor="middle">Residual (percentage points)</text>')
    svg.append(text(600,530,'CDC epidemiological week (YEAR-WEEK as reported)',19,anchor='middle'))
    residual=d.Hybrid_InvMSE-d['%UNWEIGHTED ILI']
    hist,edges=np.histogram(residual,bins=50)
    assert int(hist.sum())==len(d)==2808
    xh=lambda v:95+1045*(v-edges[0])/(edges[-1]-edges[0])
    ymax=np.ceil(hist.max()/200)*200
    yh=lambda v:745-170*v/ymax
    for v in np.arange(0,ymax+1,200):
        svg.append(line(95,yh(v),1140,yh(v)))
        svg.append(text(82,yh(v)+6,f'{v:g}',18,anchor='end'))
    for i,count in enumerate(hist):
        svg.append(f'<rect x="{xh(edges[i]):.2f}" y="{yh(count):.2f}" width="{max(.1,xh(edges[i+1])-xh(edges[i])-.4):.2f}" height="{745-yh(count):.2f}" fill="#448fc0"/>')
    for v in range(-10,7,2):
        if edges[0]<=v<=edges[-1]:
            svg.append(text(xh(v),771,str(v),18,anchor='middle'))
    svg.append(line(xh(0),575,xh(0),745,'#111',1.4,'5,4'))
    svg.append(line(95,575,95,745,'#333'))
    svg.append(line(95,745,1140,745,'#333'))
    svg.append('<text transform="translate(28,660) rotate(-90)" font-size="20" text-anchor="middle">Record count</text>')
    svg.append(text(600,805,'Residual (percentage points)',19,anchor='middle'))
    svg.append(text(600,837,'Distribution includes every scored record (n = 2,808); 50 bins.',17,anchor='middle'))
    return '\n'.join(svg+['</g></svg>'])

# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
#   hashlib.sha256(source.read_bytes()).hexdigest -> hashlib.sha256(source.read_bytes()).hexdigest (Python standard library)
#   json.dumps -> json.dumps (Python standard library)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   residuals -> local function in this file
#   trajectory -> local function in this file
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,default=DEFAULT_ROOT)
    a=p.parse_args()
    out=HERE/'map_sources'
    out.mkdir(exist_ok=True)
    name='experiment_3_ilinet_pandemic_stress_test'
    source=a.project_root/'results/reference/hybrid'/name/'predictions'/f'{name}_predictions.csv'
    d=pd.read_csv(source)
    assert len(d)==2808 and d[['YEAR','WEEK']].drop_duplicates().shape[0]==52
    d['date']=io.epi_week_start(d.YEAR,d.WEEK)
    d['Residual']=d.Hybrid_InvMSE-d['%UNWEIGHTED ILI']
    cols=['%UNWEIGHTED ILI','Hybrid_InvMSE','RandomForest','GradientBoosting','MLP']
    w=d.groupby(['YEAR','WEEK','date'])[cols].mean().reset_index().sort_values('date')
    r=d.groupby(['YEAR','WEEK','date']).Residual.agg(median='median',mean='mean',p05=lambda x:x.quantile(.05),p95=lambda x:x.quantile(.95)).reset_index().sort_values('date')
    (out/'fig10.svg').write_text(trajectory(w),encoding='utf-8')
    (out/'fig12.svg').write_text(residuals(d,r),encoding='utf-8')
    w.to_csv(HERE/'calendar_weekly_trajectory.csv',index=False)
    r.to_csv(HERE/'calendar_weekly_residuals.csv',index=False)
    summary={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'records':len(d),'weeks':len(w),'observed_peak':w.loc[w[cols[0]].idxmax(),['YEAR','WEEK',cols[0]]].to_dict(),'predicted_peak':w.loc[w.Hybrid_InvMSE.idxmax(),['YEAR','WEEK','Hybrid_InvMSE']].to_dict()}
    (HERE/'calendar_figure_provenance.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
