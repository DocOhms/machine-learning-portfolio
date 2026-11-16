# Archived auxiliary-lag development calculations

These files explain where the development numbers in the supplement originated.
They are **historical development evidence**, not new experiments and not the
current primary-model execution path. Preparing this archive did not run them.
The three runners' local input/output paths have been replaced with explicit
command-line arguments. Their calculation bodies are otherwise unchanged. The
five matching legacy helper modules are byte-for-byte copies from the preserved
legacy project; do not silently substitute the current primary helpers.

## Calculation and source map

| Evidence | Calculation source | What is actually calculated |
| --- | --- | --- |
| NSSP three rolling-origin folds and core / [1] / [1,2] / [1,2,3] comparison | `run_rolling_component_lags.py`, `main()` | Build an identical complete-case cohort for all configurations; use three expanding origins with ten validation weeks each; fit preprocessing and learners for each configuration/fold; summarize hybrid scores by an **unweighted mean over the three folds**. The saved standard deviations are pandas sample standard deviations. |
| NSSP [1,2,3] final-test comparison with [1] | `run_final_lags123.py`, `main()` | Construct three auxiliary lags, choose chronological weights and refit on pre-test data, check the target test rows, then compare with the saved `nssp_t_plus_1_lag1_comparison.csv`. The single-lag values are recalled from that file, not recomputed by this runner. |
| Contemporary/historical ILINet provider-lag comparison | `run_ilinet_exp2_exp3_provider_lags.py`, `main()` and its local helpers | Independently build and evaluate the two ILINet experiments for provider lags [1] and [1,2,3]; write fold, weight, metric, and final-comparison files. |
| File parsing and epidemiological ordering | `legacy_project/src/common_io.py` | pandas loading/numeric cleaning and the original archived date/order handling. Do not mistake these historical date labels for later corrected presentation dates. |
| Lags, three-observation rolling summaries, seasonal features, scaler and cluster augmentation | `legacy_project/src/common_features.py`, imported by each runner | pandas grouped shifts and rolling operations; NumPy trigonometry; scikit-learn `StandardScaler`, `KMeans`, and silhouette calculations. Rolling standard deviation uses pandas' sample convention. |
| Learner fitting and inverse-MSE combination | `legacy_project/src/common_models.py`, `compute_validation_weights()` and `fit_predict_final_models()` | scikit-learn RF/GBR/MLP estimators; validation mean squared errors; normalized reciprocal errors; weighted prediction sums. Fixed settings come from `legacy_project/src/common_config.py`. The libraries implement the internal optimization/tree calculations; this archive does not reimplement them. |
| Reported scores | `legacy_project/src/common_metrics.py`, `evaluate_predictions()` | MAE, RMSE, R-squared, SMAPE, bias, and Pearson correlation, using NumPy/scikit-learn/SciPy as shown in the helper. |

The rolling-development routine uses each fold's validation predictions both to
estimate ensemble weights and to summarize that weighted ensemble on the same
fold. Those summaries are therefore not a nested, independently held-out estimate
of the weighting rule. The final test windows were also inspected in the auxiliary-
lag development process. The retained primary results remain retrospective and
exploratory; this archive does not convert them into untouched confirmatory tests.

## Reviewing without training

Open the runners and follow the named helper calls into `legacy_project/src/`.
The small CSVs in `evidence/` contain the saved fold-level values, weights,
comparison values, and metadata. In the public supplement, S1-prefixed CSVs are
copies/reformatting of these archived calculations, not newly generated training
results. Row-level development predictions are not included here. The primary
prediction files used for the published results have a separate release manifest.

`SOURCE_MANIFEST.json` records the original file hash before path adaptation.
`ADAPTATION_VERIFICATION.json` records the public hashes and verifies that the
runners' code from `OUTPUT.mkdir(...)` onward is identical to the source.

## Optional execution from Jupyter

This is not a requirement for reviewing the paper and may take substantial time.
The archival runners are executable scripts, but **a full rerun of this export has
not been validated**. Exact output reproduction also depends on the original raw
snapshots, third-party versions, and machine/runtime behavior. Never use a rerun
to replace reference results automatically.

1. Obtain and verify the three raw snapshots using the public data instructions.
2. Stage copies under `archive/development/legacy_project/data/raw/` (the same
   filenames). Do not add this data directory to Git.
3. Use a kernel with the scientific dependencies and invoke the script as a child
   process, so historical helper imports cannot pollute the notebook's imports.
4. Use a **new** directory under `results/reproduced/` for outputs.

Example notebook cell (run only deliberately):

```python
from pathlib import Path
import subprocess, sys
repo = Path.cwd()  # Set to the public repository root if needed.
legacy = repo / "archive/development/legacy_project"
out = repo / "results/reproduced/development_review"
subprocess.run([
    sys.executable, str(repo / "archive/development/run_rolling_component_lags.py"),
    "--legacy-project", str(legacy), "--output-dir", str(out),
], check=True)
```

`run_final_lags123.py` additionally reads the saved single-lag comparison from its
output directory. Copy `evidence/nssp_t_plus_1_lag1_comparison.csv` there first;
its source is preserved explicitly rather than pretending this script calculates
the single-lag model. The ILINet runner takes the same two path arguments.

No author-specific paths, notebooks, authentication tokens, or review correspondence
are needed. `package_archival_evidence.py` is only the maintainer's allowlist-copy
utility; it is not part of any scientific calculation.
