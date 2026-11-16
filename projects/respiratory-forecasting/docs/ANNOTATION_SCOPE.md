# Scope of the calculation annotations

Comments immediately before functions identify calls to other local functions, imported project modules, and Python or third-party packages. The function index links to the annotated source. Mathematical definitions, object methods such as `groupby().shift()` and `.fit()`, and scientific caveats are explained in CALCULATIONS.md.

The index is a static call aid, not a complete dynamic dependency graph. Python object methods and dynamically loaded modules can only be resolved in context. For example, `model.fit` is the scikit-learn estimator constructed by `common_models.build_models`; `SARIMAX.fit/filter/get_prediction` belongs to statsmodels. There is no claim that their library internals are locally reimplemented. The notebook can display each local function body without importing or running it.

`SOURCE_PRESERVATION.json` records source hashes, public path adaptations and annotation checks. `REVIEW_VERIFICATION.json` records completed checks, including full saved-row arithmetic. A comment-only AST check is not a fresh scientific fit. No production model configuration, target, feature definition, split, forecast or manuscript result is intentionally changed.

Original locally recorded paths have been generalized in public metadata. Original scientific CSV values and input hashes remain identifiable; rewritten metadata files have new public hashes. The publication manifest covers the final public files separately from source hashes.
