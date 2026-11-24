# Machine Learning Portfolio

A collection of my machine-learning coursework, research, and independent projects. The portfolio covers data preparation, exploratory analysis, feature engineering, predictive modeling, evaluation, and reproducible computational workflows.

Each project keeps its own notebooks, source code, documentation, and data-access instructions.

## Projects

| Project | Focus | Start here |
| --- | --- | --- |
| Honolulu Traffic Data Preprocessing | Cleaning, feature engineering, and visualization of fixed traffic incident and camera-location snapshots; no trained model | [Traffic preprocessing](projects/traffic-preprocessing/README.md) |
| Respiratory Surveillance Forecasting | Retrospective one-week-ahead forecasting using temporal features, K-means context, and an RF/GBR/MLP ensemble; calculation review, classical benchmarks, sensitivity evidence, and descriptive geographic maps | [Respiratory forecasting](projects/respiratory-forecasting/) |

### Honolulu traffic preprocessing

Coursework exploring Honolulu traffic incidents and camera locations. The project contains separate notebooks, fixed example datasets, setup instructions, and notes on the limits of its engineered features. The notebooks do not report predictive accuracy.

### Respiratory surveillance forecasting

A research-manuscript companion applying the same forecasting procedure independently to NSSP and two ILINet settings. The public calculation notebook traces saved forecasts to reported metrics and geographic summaries.

This is a retrospective proof of concept, not an operational forecasting service. The notebook retains saved outputs and starts with all training and export switches disabled. See the project documentation for evaluation boundaries, data access, and reproducibility limitations.

## Working with the projects

1. Open a project's folder and review its documentation.
2. Inspect saved notebook outputs before deciding whether to run code.
3. Follow that project's environment and data-access instructions.
4. Keep exploratory reruns separate from archived reference results.

There is no single shared environment or dataset for the entire portfolio.

## Tools

Python, Jupyter, pandas, NumPy, SciPy, scikit-learn, statsmodels, Matplotlib, Plotly, and seaborn. Exact requirements vary by project.

## Licensing and attribution

Original software code is available under the [MIT License](LICENSE), subject to the [license scope](LICENSE_SCOPE.md) and project-specific notices. Manuscripts, data, figures, and third-party materials are not automatically covered by that code license. See the respiratory project's [terms and attribution](projects/respiratory-forecasting/TERMS_AND_ATTRIBUTION.md) for its material-specific distinctions.

Maintained by Alexander Roque ([@DocOhms](https://github.com/DocOhms)).
