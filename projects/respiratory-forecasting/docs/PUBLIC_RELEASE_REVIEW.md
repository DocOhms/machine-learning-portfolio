# Public release review

Historical checklist from preparation of the local review package. On 16 November 2026 the author subsequently approved public portfolio publication and MIT licensing for original code, with separate terms for other materials. See [current project terms](../TERMS_AND_ATTRIBUTION.md), [current data/release access](../data/README.md), and [portfolio integration checks](PORTFOLIO_PUBLICATION_CHECKS.json). The original checklist below is retained as history; unresolved raw-data access and methodological limitations still apply.

This package is prepared for author review before public release. It has not been uploaded to GitHub or submitted to ICCA. The six-page manuscript now leads with existing HSA-week NSSP scores and retains the original record-weighted scores as a comparison. Model calculations, archived numerical results, and heat-map assets are unchanged; documentation and source annotations explain how to trace them.

## What belongs in the public package

- Calculation-focused notebook, readable source, exact configuration, dependency files, tests, and function/source index.
- Data snapshot names and checksums, with a truthful access route for those exact snapshots.
- Primary metric tables, ensemble weights, feature lists, cohort/split metadata, classical candidate/convergence evidence, and the archived sensitivity outputs.
- Row-level reference predictions in a versioned companion archive, so published numbers can be recalculated rather than merely quoted.
- The current six-page paper and GitHub companion evidence, including unchanged full-size maps and their numeric values/provenance.

Local environments, browser/Jupyter runtime files, authentication tokens, caches, private reviewer correspondence, author-only notes, obsolete manuscript copies, and machine-specific workspace inventories do not belong in the public repository.

## Required decisions and checks before publication

| Item | Release requirement |
| --- | --- |
| Author approval | Review the final public notebook, explanatory text, and the extent of AI-assistance disclosure. Packaging does not substitute for author verification. |
| Code license | Choose and add an author-approved license. No license choice is inferred; do not describe the package as open-source until rights and terms are settled. Dataset and third-party asset rights require separate consideration. |
| Exact data access | Confirm redistribution terms and a durable access route for all three recorded snapshots. Raw CSVs are not bundled here. The code and hashes alone cannot reproduce fitting without those inputs. Do not replace them silently with current CDC downloads. |
| Reference archive access | Publish the matching `reference_calculations.zip` alongside the selected release and give its checksum and extraction instructions. The local archive is not already a public download. |
| Software environment | Preserve the current dependency specification, but disclose that archived classical `environment.json` records pandas 3.0.1 whereas the current requirements pin pandas 3.0.3. A fresh full scientific rerun under the public install is not established by arithmetic checks or notebook launch tests. |
| Evidence provenance | Retain selection grids, failed seasonal-differencing screens, convergence counts, and auxiliary-lag development history. Do not relabel imported summaries as freshly recomputed experiments. |
| Scientific claims | Preserve the distinction between within-run fitting safeguards and development-level test reuse in **all three experiments**. No zero-shot transfer, independent county accuracy, operational readiness, universal ensemble superiority, or validated-hotspot claim is supported. |
| Manuscript/repository alignment | Point the paper's final code/data availability statement to the actual public release only after it exists. Identify the GitHub evidence as companion material, not as an additional accepted conference document. The 15 November reporting amendment promotes existing HSA-week NSSP scores without changing the fitted analysis or heat maps. |
| Citation metadata | Add the final proceedings citation/DOI when known and a versioned release identifier. Do not invent a DOI or repository URL. |
| Public privacy/path check | Check the release ZIP and extracted files for tokens, private correspondence, and workstation paths. Any sanitized public provenance must retain original file hashes and explain the path-only presentation change. |

## What review checks do and do not establish

Reading the notebook should expose the calculation formula or the precise local function/library that performs it. An unchanged-code/annotation audit establishes that comments have not altered model logic. Recalculating saved metrics tests arithmetic and row alignment; it does not retrain the original estimators or validate real-time data availability. Static notebook checks do not establish a full clean-machine scientific reproduction.

Use the package's generated verification report for the checks actually completed. If a check was skipped because a data archive or scientific dependency was absent, retain that status explicitly rather than marking it passed. No new experiment is required to review this release package, and no new model should be substituted for the locked result.

## Deferred scientific improvements

New untouched forecast windows, reporting-vintage analyses, separate target-lag/rolling/seasonal ablations, weight stability across seasons, broader ARIMA/SARIMA searches, LSTM models, probabilistic forecasts, and formal spatial analyses remain deferred. Their absence should remain a limitation, not an undocumented promise or a claim that those methods would perform worse.

See the [supplement guide](../supplement/README.md) for the section-to-evidence mapping and the [calculation guide](CALCULATIONS.md) for the numerical workflow.
