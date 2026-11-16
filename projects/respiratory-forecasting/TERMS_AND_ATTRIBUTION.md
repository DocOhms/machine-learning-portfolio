# Terms and attribution

This project contains different categories of material. The [MIT License](LICENSE) is a code license, not a blanket license for every file in this directory or the companion release archive.

## Original software: MIT

Original Python and JavaScript source, original notebook code cells, and associated software configuration are offered under the MIT License. This includes the original implementations in `src/`, `scripts/`, `experiments/`, `benchmarks/`, `tests/`, preserved development code, and original plotting/verification scripts under `manuscript/latex/`. Third-party code, if separately identified, retains its own terms. Preserve the copyright and permission notice when reusing MIT-licensed material.

## Manuscript, scientific text, figures, and maps

The manuscript and supplement PDFs/LaTeX, scientific prose in notebooks, and saved figures/maps are provided for reading, inspection, and citation. They are excluded from the MIT code grant. No additional blanket reuse or republication license for those materials is granted by this repository. Applicable author, publisher, and third-party rights remain with their respective holders; obtain any permission required for your intended reuse.

Sharing these files does not represent permission to reproduce an IEEE logo or imply endorsement by IEEE, CDC, or another institution. The repository preserves the reviewed manuscript snapshot; repository documentation is the current guide to code and reference-archive access.

## Surveillance data and saved numerical evidence

Sources: the U.S. Centers for Disease Control and Prevention (CDC), National Syndromic Surveillance Program (NSSP), and ILINet. CDC surveillance resources are available without charge through CDC; exact historical snapshot availability may differ from current downloads. The three original raw snapshots are not included here. Their expected names and checksums are recorded in `data/manifest.json`.

The derived prediction, metric, cohort, sensitivity, and map-value files, including the companion `reference_calculations.zip`, are shared for inspecting the study's calculations. They are not licensed as software under MIT, and this repository does not relicense underlying surveillance data. Follow applicable source-specific terms and attribution requirements. The availability of derived evidence does not establish permission to redistribute every underlying raw snapshot.

Many CDC materials are public domain, but exceptions and third-party conditions can apply. Consult [CDC's use-of-agency-materials policy](https://www.cdc.gov/other/agencymaterials.html) and the source identified for the particular material. This notice does not assert copyright over public-domain data or reduce any rights that applicable law or source terms already provide.

Interpretations and analyses are those of the authors. Reference to CDC, ATSDR, HHS, or the U.S. Government does not imply their endorsement of this project or its findings. Source surveillance observations and the study's derived predictions must not be presented as interchangeable official estimates.

## Third-party geography and software

- `manuscript/latex/geometry/states-albers-10m.json` is from [U.S. Atlas](https://github.com/topojson/us-atlas), using the source URL and checksum preserved in `manuscript/latex/map_provenance.json`. U.S. Atlas redistributes Census-derived geographic boundaries. Its upstream copyright/permission notice is included in [third_party/us-atlas-LICENSE.txt](third_party/us-atlas-LICENSE.txt); this material is not relicensed under this project's MIT grant.
- Installed dependencies, including scikit-learn, statsmodels, pandas, NumPy, SciPy, Jupyter, Matplotlib, Plotly, and rendering utilities, retain their own upstream licenses. Dependency version files do not transfer ownership or replace those terms.
- Fonts and external software used to create existing figures are not bundled or relicensed here.

Historical audit records document their original scope and date; they are not additional licenses. Questions about reuse of the manuscript or other excluded author-controlled material should be directed to the authors through the portfolio's GitHub account.
