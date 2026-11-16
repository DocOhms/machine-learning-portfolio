# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
# statsmodels SARIMAX supplies maximum-likelihood estimation, filtering and one-step prediction. Finite nonconverged forecasts can be retained; convergence and coverage are separate diagnostics.
"""Isolated, same-cohort SARIMA feasibility pilot.

The script reads locked inputs and writes only below sarima_pilot_20260716.
Candidate selection uses chronological validation data. Final-test observations
are used only for one-step filtering and scoring after an order has been locked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX


PILOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PILOT_ROOT.parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
LOCKED_OUTPUT = PROJECT_ROOT / "results" / "reference" / "hybrid"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "reproduced" / "classical"


@dataclass(frozen=True)
class Experiment:
    key: str
    name: str
    data_file: str
    kind: str
    target: str
    groups: tuple[str, ...]
    covariate: str | None
    test_cutoff: int
    validation_cutoff: int
    train_start: int | None = None
    test_end: int | None = None


EXPERIMENTS = {
    "exp1": Experiment(
        "exp1", "experiment_1_nssp_primary", "NSSP.csv", "nssp",
        "percent_visits_combined", ("geography", "county"), None,
        202419, 202402,
    ),
    "exp2": Experiment(
        "exp2", "experiment_2_ilinet_contemporary", "ILINet.csv", "ilinet",
        "%UNWEIGHTED ILI", ("REGION TYPE", "REGION"), "NUM. OF PROVIDERS",
        202531, 202502,
    ),
    "exp3": Experiment(
        "exp3", "experiment_3_ilinet_pandemic_stress_test", "ILINet_2016_2021.csv", "ilinet",
        "%UNWEIGHTED ILI", ("REGION TYPE", "REGION"), "NUM. OF PROVIDERS",
        201940, 201909, train_start=201640, test_end=202039,
    ),
}


# Calculation provenance:
#   pd.to_numeric -> pandas.to_numeric (third-party package)
def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"X": np.nan, "x": np.nan, "": np.nan}), errors="coerce")


# Calculation provenance:
#   hashlib.sha256 -> hashlib.sha256 (Python standard library)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Calculation provenance:
#   clean_numeric -> local function in this file
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.to_datetime -> pandas.to_datetime (third-party package)
def load_raw(exp: Experiment) -> pd.DataFrame:
    path = RAW_ROOT / exp.data_file
    if exp.kind == "nssp":
        columns = [
            "week_end", *exp.groups, exp.target,
            "percent_visits_covid", "percent_visits_influenza", "percent_visits_rsv",
        ]
        df = pd.read_csv(path, usecols=columns, low_memory=False)
        df["date"] = pd.to_datetime(df["week_end"], errors="coerce")
        iso = df["date"].dt.isocalendar()
        df["YEAR"] = iso.year.astype("Int64")
        df["WEEK"] = iso.week.astype("Int64")
        for column in [exp.target, "percent_visits_covid", "percent_visits_influenza", "percent_visits_rsv"]:
            df[column] = clean_numeric(df[column])
    else:
        df = pd.read_csv(path, skiprows=1, low_memory=False)
        df["YEAR"] = clean_numeric(df["YEAR"]).astype("Int64")
        df["WEEK"] = clean_numeric(df["WEEK"]).astype("Int64")
        df[exp.target] = clean_numeric(df[exp.target])
        df[exp.covariate] = clean_numeric(df[exp.covariate])
    df = df.dropna(subset=["YEAR", "WEEK", *exp.groups]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["WEEK"] = df["WEEK"].astype(int)
    df["epi_order"] = df["YEAR"] * 100 + df["WEEK"]
    for column in exp.groups:
        df[column] = df[column].astype(str)
    return df.sort_values([*exp.groups, "epi_order"])


# Calculation provenance:
#   np.cos -> numpy.cos (third-party package)
#   np.sin -> numpy.sin (third-party package)
#   pd.to_numeric -> pandas.to_numeric (third-party package)
def add_cohort_features(df: pd.DataFrame, exp: Experiment) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    grouped = out.groupby(list(exp.groups), dropna=False, sort=False)
    target_group = grouped[exp.target]
    features: list[str] = []
    for lag in (1, 2, 3):
        name = f"lag_{lag}"
        out[name] = target_group.shift(lag)
        features.append(name)
    out["rolling_mean_3"] = target_group.transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    out["rolling_std_3"] = target_group.transform(lambda x: x.shift(1).rolling(3, min_periods=3).std())
    out["seasonal_naive"] = target_group.shift(52)
    week = pd.to_numeric(out["WEEK"], errors="coerce")
    out["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * week / 52.0)
    features.extend(["rolling_mean_3", "rolling_std_3", "week_sin", "week_cos"])
    if exp.kind == "nssp":
        for column in ("percent_visits_covid", "percent_visits_influenza", "percent_visits_rsv"):
            name = f"{column}_lag_1"
            out[name] = grouped[column].shift(1)
            features.append(name)
    else:
        name = f"{exp.covariate}_lag_1"
        out[name] = grouped[exp.covariate].shift(1)
        features.append(name)
    # seasonal_naive is required to reproduce the locked complete-case cohort,
    # although the hybrid model does not use it as a supervised feature.
    cohort_columns = [exp.target, *features, "seasonal_naive"]
    return out.dropna(subset=cohort_columns).copy(), features


def locked_paths(exp: Experiment) -> tuple[Path, Path]:
    root = LOCKED_OUTPUT / exp.name
    return (
        root / "predictions" / f"{exp.name}_predictions.csv",
        root / "metrics" / f"{exp.name}_run_metadata.csv",
    )


def key_tuples(frame: pd.DataFrame, exp: Experiment) -> set[tuple]:
    return set(frame[[*exp.groups, "epi_order"]].itertuples(index=False, name=None))


# Calculation provenance:
#   add_cohort_features -> local function in this file
#   key_tuples -> local function in this file
#   locked_paths -> local function in this file
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.to_numeric -> pandas.to_numeric (third-party package)
#   pd.to_numeric(locked['epi_order'], errors='raise').astype -> pandas.to_numeric(locked['epi_order'], errors='raise').astype (third-party package)
def reproduce_cohorts(raw: pd.DataFrame, exp: Experiment) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    modeled, features = add_cohort_features(raw, exp)
    if exp.train_start is not None:
        modeled = modeled[modeled["epi_order"] >= exp.train_start].copy()
    train = modeled[modeled["epi_order"] < exp.test_cutoff].copy()
    if exp.test_end is None:
        test = modeled[modeled["epi_order"] >= exp.test_cutoff].copy()
    else:
        test = modeled[(modeled["epi_order"] >= exp.test_cutoff) & (modeled["epi_order"] <= exp.test_end)].copy()
    validation = train[train["epi_order"] >= exp.validation_cutoff].copy()

    pred_path, meta_path = locked_paths(exp)
    locked = pd.read_csv(pred_path, low_memory=False)
    for column in exp.groups:
        locked[column] = locked[column].astype(str)
    locked["epi_order"] = pd.to_numeric(locked["epi_order"], errors="raise").astype(int)
    locked_test_keys = key_tuples(locked, exp)
    reproduced_test_keys = key_tuples(test, exp)
    meta = pd.read_csv(meta_path).iloc[0]
    audit = {
        "feature_count": len(features),
        "reproduced_validation_rows": len(validation),
        "locked_validation_rows": int(meta["Validation_Rows"]),
        "reproduced_test_rows": len(test),
        "locked_test_rows": int(meta["Test_Rows"]),
        "test_keys_missing_from_reproduction": len(locked_test_keys - reproduced_test_keys),
        "extra_reproduced_test_keys": len(reproduced_test_keys - locked_test_keys),
    }
    if audit["reproduced_validation_rows"] != audit["locked_validation_rows"]:
        raise RuntimeError(f"Validation cohort mismatch for {exp.key}: {audit}")
    if locked_test_keys != reproduced_test_keys:
        raise RuntimeError(f"Final-test cohort mismatch for {exp.key}: {audit}")
    return validation, test, audit


# Calculation provenance:
#   np.random.default_rng -> numpy.random.default_rng (third-party package)
#   pd.concat -> pandas.concat (third-party package)
#   pd.concat([aggregate, *sampled_parts], ignore_index=True).head -> pandas.concat([aggregate, *sampled_parts], ignore_index=True).head (third-party package)
#   pd.qcut -> pandas.qcut (third-party package)
def choose_nssp_groups(raw: pd.DataFrame, test: pd.DataFrame, cutoff: int, count: int, seed: int) -> list[tuple]:
    eligible = test[list(EXPERIMENTS["exp1"].groups)].drop_duplicates()
    history = raw[raw["epi_order"] < cutoff].groupby(["geography", "county"], as_index=False)["percent_visits_combined"].agg(["count", "std"]).reset_index()
    eligible = eligible.merge(history, on=["geography", "county"], how="left")
    eligible["is_aggregate"] = eligible["county"].str.casefold().eq("all")
    eligible["variability_bin"] = pd.qcut(eligible["std"].rank(method="first"), 5, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    aggregate_n = min(max(1, count // 12), int(eligible["is_aggregate"].sum()))
    aggregate = eligible[eligible["is_aggregate"]].sample(n=aggregate_n, random_state=seed)
    ordinary = eligible[~eligible["is_aggregate"]].copy()
    ordinary["random"] = rng.random(len(ordinary))
    ordinary = ordinary.sort_values(["variability_bin", "random"])
    need = count - len(aggregate)
    sampled_parts = []
    bins = sorted(ordinary["variability_bin"].dropna().unique())
    base = need // max(1, len(bins))
    remainder = need % max(1, len(bins))
    for index, value in enumerate(bins):
        part = ordinary[ordinary["variability_bin"] == value]
        sampled_parts.append(part.head(base + (1 if index < remainder else 0)))
    sample = pd.concat([aggregate, *sampled_parts], ignore_index=True).head(count)
    return list(sample[["geography", "county"]].itertuples(index=False, name=None))


def subset_groups(frame: pd.DataFrame, exp: Experiment, groups: list[tuple]) -> pd.DataFrame:
    allowed = set(groups)
    mask = [key in allowed for key in frame[list(exp.groups)].itertuples(index=False, name=None)]
    return frame.loc[mask].copy()


# Calculation provenance:
#   np.abs -> numpy.abs (third-party package)
#   np.isfinite -> numpy.isfinite (third-party package)
#   np.mean -> numpy.mean (third-party package)
#   np.sqrt -> numpy.sqrt (third-party package)
#   np.sum -> numpy.sum (third-party package)
def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid]
    p = predicted[valid]
    if not len(y):
        return {"N": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    residual = y - p
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "N": int(len(y)),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan,
    }


# Calculation provenance:
#   SARIMAX -> statsmodels.tsa.statespace.sarimax.SARIMAX (third-party package)
#   np.asarray -> numpy.asarray (third-party package)
#   np.isfinite -> numpy.isfinite (third-party package)
#   np.searchsorted -> numpy.searchsorted (third-party package)
#   time.perf_counter -> time.perf_counter (Python standard library)
#   warnings.catch_warnings -> warnings.catch_warnings (Python standard library)
#   warnings.simplefilter -> warnings.simplefilter (Python standard library)
def fit_and_predict_group(
    raw_group: pd.DataFrame,
    exp: Experiment,
    score_rows: pd.DataFrame,
    cutoff: int,
    candidate: dict,
    maxiter: int,
) -> tuple[pd.DataFrame, dict]:
    start = exp.train_start if exp.train_start is not None else int(raw_group["epi_order"].min())
    score_end = int(score_rows["epi_order"].max())
    timeline = sorted(raw_group.loc[(raw_group["epi_order"] >= start) & (raw_group["epi_order"] <= score_end), "epi_order"].unique())
    values = raw_group.groupby("epi_order")[exp.target].mean().reindex(timeline).astype(float)
    fit_end = int(np.searchsorted(np.asarray(timeline), cutoff, side="left"))
    train_values = values.iloc[:fit_end]
    outcome = {"converged": False, "error": "", "fit_seconds": 0.0, "train_n": int(train_values.notna().sum())}
    prediction_frame = score_rows[[*exp.groups, "epi_order", exp.target]].copy()
    prediction_frame["prediction"] = np.nan
    if train_values.notna().sum() < 20:
        outcome["error"] = "fewer_than_20_training_observations"
        return prediction_frame, outcome
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_model = SARIMAX(
                train_values.to_numpy(),
                order=tuple(candidate["order"]),
                seasonal_order=tuple(candidate["seasonal_order"]),
                trend=candidate["trend"],
                enforce_stationarity=False,
                enforce_invertibility=False,
                concentrate_scale=True,
            )
            fitted = train_model.fit(disp=False, maxiter=maxiter, method="lbfgs")
            full_model = SARIMAX(
                values.to_numpy(),
                order=tuple(candidate["order"]),
                seasonal_order=tuple(candidate["seasonal_order"]),
                trend=candidate["trend"],
                enforce_stationarity=False,
                enforce_invertibility=False,
                concentrate_scale=True,
            )
            filtered = full_model.filter(fitted.params)
            one_step = np.asarray(filtered.get_prediction(start=0, end=len(values) - 1, information_set="predicted").predicted_mean)
        mapping = dict(zip(timeline, one_step))
        prediction_frame["prediction"] = prediction_frame["epi_order"].map(mapping)
        outcome["converged"] = bool(fitted.mle_retvals.get("converged", False))
        outcome["warnings"] = len(caught)
        outcome["aic"] = float(fitted.aic) if np.isfinite(fitted.aic) else np.nan
    except Exception as exc:  # recorded rather than hidden; pilot must quantify failures
        outcome["error"] = f"{type(exc).__name__}: {exc}"[:500]
    outcome["fit_seconds"] = time.perf_counter() - started
    return prediction_frame, outcome


# Calculation provenance:
#   fit_and_predict_group -> local function in this file
#   metrics -> local function in this file
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.concat -> pandas.concat (third-party package)
def evaluate_candidate(
    raw: pd.DataFrame,
    score: pd.DataFrame,
    exp: Experiment,
    cutoff: int,
    candidate: dict,
    maxiter: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    predictions = []
    diagnostics = []
    grouped_raw = {key if isinstance(key, tuple) else (key,): group for key, group in raw.groupby(list(exp.groups), sort=False)}
    for key, score_group in score.groupby(list(exp.groups), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_prediction, outcome = fit_and_predict_group(grouped_raw[key_tuple], exp, score_group, cutoff, candidate, maxiter)
        predictions.append(group_prediction)
        diagnostics.append({**dict(zip(exp.groups, key_tuple)), **outcome})
    pred = pd.concat(predictions, ignore_index=True)
    diag = pd.DataFrame(diagnostics)
    scored = metrics(pred[exp.target].to_numpy(float), pred["prediction"].to_numpy(float))
    total = len(pred)
    summary = {
        "Experiment": exp.key,
        "Candidate": candidate["name"],
        "Seasonal": bool(candidate["seasonal"]),
        **scored,
        "Coverage": scored["N"] / total if total else 0.0,
        "Groups": len(diag),
        "Converged_Groups": int(diag["converged"].sum()),
        "Convergence_Rate": float(diag["converged"].mean()),
        "Fit_Seconds": float(diag["fit_seconds"].sum()),
        "Median_Group_Seconds": float(diag["fit_seconds"].median()),
        "P95_Group_Seconds": float(diag["fit_seconds"].quantile(0.95)),
        "Error_Groups": int(diag["error"].astype(bool).sum()),
    }
    return summary, pred, diag


# Calculation provenance:
#   choose_nssp_groups -> local function in this file
#   evaluate_candidate -> local function in this file
#   json.dumps -> json.dumps (Python standard library)
#   load_raw -> local function in this file
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame([test_summary]).to_csv -> pandas.DataFrame([test_summary]).to_csv (third-party package)
#   pd.DataFrame(group_list, columns=exp.groups).to_csv -> pandas.DataFrame(group_list, columns=exp.groups).to_csv (third-party package)
#   pd.DataFrame(summaries).to_csv -> pandas.DataFrame(summaries).to_csv (third-party package)
#   pd.concat -> pandas.concat (third-party package)
#   pd.concat(all_diags, ignore_index=True).to_csv -> pandas.concat(all_diags, ignore_index=True).to_csv (third-party package)
#   reproduce_cohorts -> local function in this file
#   subset_groups -> local function in this file
#   time.perf_counter -> time.perf_counter (Python standard library)
def run_experiment(
    exp: Experiment,
    config: dict,
    candidate_profile: str,
    quick_candidates: int | None = None,
) -> dict:
    out = OUTPUT_ROOT / exp.key
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw(exp)
    validation, test, audit = reproduce_cohorts(raw, exp)
    group_list = list(test[list(exp.groups)].drop_duplicates().itertuples(index=False, name=None))
    if exp.key == "exp1":
        group_list = choose_nssp_groups(raw, test, exp.test_cutoff, int(config["nssp_pilot_groups"]), int(config["random_seed"]))
        pd.DataFrame(group_list, columns=exp.groups).to_csv(out / "nssp_sampled_groups.csv", index=False)
    raw = subset_groups(raw, exp, group_list)
    validation = subset_groups(validation, exp, group_list)
    test = subset_groups(test, exp, group_list)

    candidates = config["candidate_models"]
    if candidate_profile == "day1":
        names = set(config["day_one_candidate_names"])
        candidates = [item for item in candidates if item["name"] in names]
    if quick_candidates is not None:
        candidates = candidates[:quick_candidates]
    if not any(item["seasonal"] for item in candidates):
        raise ValueError(
            "The selected candidate subset has no seasonal model. "
            "Use --quick-candidates 4 or greater for a development run."
        )
    summaries = []
    all_diags = []
    started = time.perf_counter()
    for candidate in candidates:
        print(f"[{exp.key}] validation {candidate['name']} ({len(group_list)} groups)", flush=True)
        summary, _, diag = evaluate_candidate(raw, validation, exp, exp.validation_cutoff, candidate, int(config["maxiter"]))
        summaries.append(summary)
        diag.insert(0, "Candidate", candidate["name"])
        all_diags.append(diag)
        pd.DataFrame(summaries).to_csv(out / "validation_candidate_summary.partial.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(out / "validation_candidate_summary.csv", index=False)
    pd.concat(all_diags, ignore_index=True).to_csv(out / "validation_group_diagnostics.csv", index=False)

    eligible = summary_frame[summary_frame["Coverage"] >= float(config["minimum_candidate_coverage"])].copy()
    seasonal_eligible = eligible[eligible["Seasonal"]].copy()
    if seasonal_eligible.empty:
        raise RuntimeError(f"No seasonal candidate met the coverage gate for {exp.key}")
    selected_name = seasonal_eligible.sort_values(["RMSE", "MAE"]).iloc[0]["Candidate"]
    selected = next(item for item in candidates if item["name"] == selected_name)
    print(f"[{exp.key}] final test selected {selected_name}", flush=True)
    test_summary, test_predictions, test_diag = evaluate_candidate(raw, test, exp, exp.test_cutoff, selected, int(config["maxiter"]))
    test_summary["Selected_On"] = "chronological_validation_RMSE"
    test_predictions.to_csv(out / "final_test_predictions.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(out / "final_test_summary.csv", index=False)
    test_diag.to_csv(out / "final_test_group_diagnostics.csv", index=False)
    result = {
        "experiment": exp.key,
        "candidate_profile": candidate_profile,
        "selected_sarima": selected_name,
        "pilot_groups": len(group_list),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "wall_seconds": time.perf_counter() - started,
        "cohort_audit": audit,
        "test_summary": test_summary,
    }
    (out / "run_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


# Calculation provenance:
#   file_sha256 -> local function in this file
#   json.dumps -> json.dumps (Python standard library)
#   locked_paths -> local function in this file
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(rows).to_csv -> pandas.DataFrame(rows).to_csv (third-party package)
#   platform.platform -> platform.platform (Python standard library)
#   platform.python_version -> platform.python_version (Python standard library)
def write_manifest(config: dict) -> None:
    rows = []
    for exp in EXPERIMENTS.values():
        pred, meta = locked_paths(exp)
        for role, path in (("raw_data", RAW_ROOT / exp.data_file), ("locked_predictions", pred), ("locked_metadata", meta)):
            rows.append({"experiment": exp.key, "role": role, "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "input_manifest_sha256.csv", index=False)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "config": config,
    }
    (OUTPUT_ROOT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   json.dumps -> json.dumps (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   run_experiment -> local function in this file
#   write_manifest -> local function in this file
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=["exp2", "exp3", "exp1"])
    parser.add_argument("--nssp-groups", type=int)
    parser.add_argument("--candidate-profile", choices=["day1", "all"], default="day1")
    parser.add_argument("--quick-candidates", type=int, help="Development smoke test only; omit for the complete pilot grid.")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((PILOT_ROOT / "pilot_config.json").read_text(encoding="utf-8"))
    if args.nssp_groups is not None:
        config["nssp_pilot_groups"] = args.nssp_groups
    write_manifest(config)
    results = []
    for key in args.experiments:
        results.append(
            run_experiment(
                EXPERIMENTS[key],
                config,
                args.candidate_profile,
                args.quick_candidates,
            )
        )
    (OUTPUT_ROOT / "pilot_run_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
