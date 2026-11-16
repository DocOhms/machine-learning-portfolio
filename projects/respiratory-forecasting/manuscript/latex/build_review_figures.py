# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Regenerate descriptive manuscript maps from locked predictions only.

No model fitting or modification of raw data/reference results. Uses pinned,
Census-derived us-atlas geometry already stored in geometry/.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPERIMENTS = [
    dict(name="experiment_1_nssp_primary", file="fig13", panel="A", title="Experiment 1 | NSSP emergency-department surveillance", target="percent_visits_combined", region="geography", unit="Combined respiratory visits (% of all ED visits)", period="May 11 - September 28, 2024 | 21 test weeks", colors=["#fff5eb", "#fdd0a2", "#f16913", "#7f2704"], maximum=4.0),
    dict(name="experiment_2_ilinet_contemporary", file="fig14", panel="B", title="Experiment 2 | Contemporary ILINet surveillance", target="%UNWEIGHTED ILI", region="REGION", unit="Unweighted ILI (% of outpatient visits)", period="2025-W31 - 2026-W14 | 37 test weeks", colors=["#f7fbff", "#c6dbef", "#4292c6", "#08306b"], maximum=6.0),
    dict(name="experiment_3_ilinet_pandemic_stress_test", file="fig15", panel="C", title="Experiment 3 | Historical ILINet stress interval", target="%UNWEIGHTED ILI", region="REGION", unit="Unweighted ILI (% of outpatient visits)", period="2019-W40 - 2020-W39 | 52 test weeks", colors=["#fcfbfd", "#dadaeb", "#9e9ac8", "#3f007d"], maximum=6.0),
]

def interpolate(colors, value):
    t = min(1.0, max(0.0, value)) * (len(colors)-1)
    index = min(int(t), len(colors)-2)
    frac = t-index
    a = [int(colors[index][i:i+2], 16) for i in (1,3,5)]
    b = [int(colors[index+1][i:i+2], 16) for i in (1,3,5)]
    return "#"+"".join(f"{round(x+(y-x)*frac):02x}" for x,y in zip(a,b))

def state_paths(topology):
    sx, sy = topology["transform"]["scale"]
    tx, ty = topology["transform"]["translate"]
    arcs = []
    for encoded in topology["arcs"]:
        x = y = 0
        points = []
        for dx,dy in encoded:
            x += dx
            y += dy
            points.append((x*sx+tx, y*sy+ty))
        arcs.append(points)
    out = {}
    for geometry in topology["objects"]["states"]["geometries"]:
        polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
        path = []
        for polygon in polygons:
            for ring in polygon:
                points = []
                for i in ring:
                    part = arcs[i] if i >= 0 else arcs[~i][::-1]
                    points.extend(part if not points else part[1:])
                path.append("M"+" L".join(f"{x:.3f},{y:.3f}" for x,y in points)+" Z")
        out[geometry["properties"]["name"]] = " ".join(path)
    return out

# Calculation provenance:
#   html.escape -> html.escape (Python standard library)
def text(x,y,value,size=19,fill="#263238",weight="normal",anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'

# Calculation provenance:
#   html.escape -> html.escape (Python standard library)
#   interpolate -> local function in this file
#   np.arange -> numpy.arange (third-party package)
#   text -> local function in this file
def build_map(spec, values, paths):
    maximum = spec["maximum"]
    if values.loc[values.mapped, "Mean_Prediction"].max() > maximum:
        raise ValueError(f"Map scale clips {spec['name']}")
    colors = spec["colors"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850">', '<rect width="1200" height="850" fill="white"/>', '<g font-family="Arial, Helvetica, sans-serif">']
    svg.append(f'<rect x="30" y="20" width="42" height="42" rx="3" fill="{colors[-1]}"/>')
    svg.append(text(51,50,spec["panel"],27,"white","bold","middle"))
    svg.append(text(87,49,spec["title"],27,colors[-1],"bold"))
    svg.append(text(35,88,spec["period"],22))
    svg.append(text(35,120,"Mean hybrid prediction over the held-out interval",21,weight="bold"))
    svg.append(text(35,148,spec["unit"],21))
    svg.append('<g transform="translate(85,165) scale(0.91)">')
    value_dict = dict(zip(values.Reporting_Unit, values.Mean_Prediction))
    for state, path in paths.items():
        value = value_dict.get(state)
        fill = "#dddddd" if value is None else interpolate(colors,value/maximum)
        tip = f"{state}: no eligible records" if value is None else f"{state}: {value:.4f}%"
        svg.append(f'<path d="{path}" fill="{fill}" stroke="#707070" stroke-width="0.72" fill-rule="evenodd"><title>{html.escape(tip)}</title></path>')
    svg.append('</g>')
    svg.append(text(1072,191,"Predicted",18,weight="bold"))
    svg.append(text(1072,215,"visits (%)",18,weight="bold"))
    for j in range(300):
        v = 1-j/299
        svg.append(f'<rect x="1072" y="{233+j*1.25:.2f}" width="27" height="1.4" fill="{interpolate(colors,v)}"/>')
    for value in np.arange(0,maximum+0.5,1):
        yy = 233+375*(1-value/maximum)
        svg.append(f'<line x1="1099" x2="1106" y1="{yy}" y2="{yy}" stroke="#555"/>')
        svg.append(text(1113,yy+6,f"{value:.0f}",18))
    svg.append('<rect x="1055" y="637" width="20" height="20" fill="#dddddd" stroke="#707070"/>')
    svg.append(text(1083,653,"No data",17))
    svg.append(text(35,740,"Alaska and Hawaii are shown as repositioned insets; grey is missing, not zero.",18))
    if spec["panel"] == "A":
        notes = ["State color averages all included reporting records within each state; no population weighting.",
                 "County labels may share HSA-level values. State aggregates are included; the U.S. aggregate is not mapped.",
                 "NSSP uses a separate 0-4% scale and a different target from the two ILINet maps."]
    else:
        notes = ["State color averages that state's reporting series over its eligible test weeks; no population weighting.",
                 "New York uses the state series only. NYC, Puerto Rico and Virgin Islands remain in companion data.",
                 "Both ILINet maps use 0-6% scales. Different periods are shown; colors describe burden, not accuracy."]
    for i,note in enumerate(notes):
        svg.append(text(35,773+i*27,note,17))
    svg.extend(['</g>','</svg>'])
    return "\n".join(svg)

# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   build_map -> local function in this file
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
#   hashlib.sha256(geometry.read_bytes()).hexdigest -> hashlib.sha256(geometry.read_bytes()).hexdigest (Python standard library)
#   hashlib.sha256(pred_path.read_bytes()).hexdigest -> hashlib.sha256(pred_path.read_bytes()).hexdigest (Python standard library)
#   json.dumps -> json.dumps (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   pd.concat -> pandas.concat (third-party package)
#   pd.concat(summaries, ignore_index=True).to_csv -> pandas.concat(summaries, ignore_index=True).to_csv (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   state_paths -> local function in this file
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    args=parser.parse_args()
    geometry=HERE/'geometry/states-albers-10m.json'
    paths=state_paths(json.loads(geometry.read_text()))
    output=args.output_dir
    (output/'map_sources').mkdir(parents=True,exist_ok=True)
    summaries=[]
    provenance=[]
    for spec in EXPERIMENTS:
        pred_path=args.project_root/'results/reference/hybrid'/spec['name']/'predictions'/f"{spec['name']}_predictions.csv"
        df=pd.read_csv(pred_path,low_memory=False)
        by=df.groupby(spec['region'],dropna=False).agg(Mean_Prediction=('Hybrid_InvMSE','mean'),Mean_Observed=(spec['target'],'mean'),Records=('Hybrid_InvMSE','size'),Weeks=('epi_order','nunique')).reset_index().rename(columns={spec['region']:'Reporting_Unit'})
        by['mapped']=by.Reporting_Unit.isin(paths)
        by['Experiment']=spec['name']
        by['Target']=spec['target']
        by['Map_Scale_Min']=0.0
        by['Map_Scale_Max']=spec['maximum']
        by['Mean_Bias'] = by['Mean_Prediction'] - by['Mean_Observed']
        by['Aggregation']='mean across included reporting records; no population weighting' if spec['panel']=='A' else 'mean of distinct reporting-unit series over eligible weeks; NYC not merged into NY'
        summaries.append(by)
        (output/'map_sources'/f"{spec['file']}.svg").write_text(build_map(spec,by,paths),encoding='utf-8')
        provenance.append(dict(Experiment=spec['name'],Input=str(pred_path),SHA256=hashlib.sha256(pred_path.read_bytes()).hexdigest(),Test_Rows=len(df),Test_Weeks=int(df.epi_order.nunique()),Mapped_Units=int(by.mapped.sum()),Unmapped_Units=by.loc[~by.mapped,'Reporting_Unit'].tolist(),Missing_Map_States=sorted(set(paths)-set(by.Reporting_Unit)),Mapped_Minimum=float(by.loc[by.mapped,'Mean_Prediction'].min()),Mapped_Maximum=float(by.loc[by.mapped,'Mean_Prediction'].max()),Scale=[0,spec['maximum']]))
    pd.concat(summaries,ignore_index=True).to_csv(output/'map_values.csv',index=False)
    (output/'map_provenance.json').write_text(json.dumps(dict(Geometry_URL='https://cdn.jsdelivr.net/npm/us-atlas@3/states-albers-10m.json',Geometry_SHA256=hashlib.sha256(geometry.read_bytes()).hexdigest(),Geometry_Description='Census-derived US Atlas 3 preprojected Albers USA state geometry; Alaska/Hawaii inset transform',Experiments=provenance),indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':
    main()
