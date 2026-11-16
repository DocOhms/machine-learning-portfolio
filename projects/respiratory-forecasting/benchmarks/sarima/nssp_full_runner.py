# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Resumable full-NSSP SARIMA runner with candidate and group batching."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sarima_pilot as core


ROOT = Path(__file__).resolve().parent
OUT = core.OUTPUT_ROOT
EXP = core.EXPERIMENTS["exp1"]


# Calculation provenance:
#   json.loads -> json.loads (Python standard library)
def candidate_by_name(name: str) -> dict:
    config = json.loads((ROOT / "pilot_config.json").read_text(encoding="utf-8"))
    for candidate in config["candidate_models"]:
        if candidate["name"] == name:
            return candidate
    raise KeyError(name)


# Calculation provenance:
#   core.load_raw -> sarima_pilot.load_raw (benchmarks/sarima/sarima_pilot.py)
#   core.reproduce_cohorts -> sarima_pilot.reproduce_cohorts (benchmarks/sarima/sarima_pilot.py)
def load_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = core.load_raw(EXP)
    validation, test, audit = core.reproduce_cohorts(raw, EXP)
    return raw, validation, test, audit


def group_keys(frame: pd.DataFrame) -> list[tuple]:
    return sorted(frame[list(EXP.groups)].drop_duplicates().itertuples(index=False, name=None))


# Calculation provenance:
#   core.metrics -> sarima_pilot.metrics (benchmarks/sarima/sarima_pilot.py)
def score_summary(pred: pd.DataFrame, diag: pd.DataFrame, candidate: dict, phase: str) -> dict:
    scored = core.metrics(pred[EXP.target].to_numpy(float), pred["prediction"].to_numpy(float))
    return {
        "Experiment": "exp1_nssp_full",
        "Phase": phase,
        "Candidate": candidate["name"],
        "Seasonal": bool(candidate["seasonal"]),
        **scored,
        "Coverage": scored["N"] / len(pred),
        "Groups": len(diag),
        "Converged_Groups": int(diag["converged"].sum()),
        "Convergence_Rate": float(diag["converged"].mean()),
        "Fit_Seconds": float(diag["fit_seconds"].sum()),
        "Median_Group_Seconds": float(diag["fit_seconds"].median()),
        "P95_Group_Seconds": float(diag["fit_seconds"].quantile(0.95)),
        "Error_Groups": int(diag["error"].fillna("").astype(bool).sum()),
    }


# Calculation provenance:
#   candidate_by_name -> local function in this file
#   core.choose_nssp_groups -> sarima_pilot.choose_nssp_groups (benchmarks/sarima/sarima_pilot.py)
#   core.fit_and_predict_group -> sarima_pilot.fit_and_predict_group (benchmarks/sarima/sarima_pilot.py)
#   group_keys -> local function in this file
#   json.dumps -> json.dumps (Python standard library)
#   json.loads -> json.loads (Python standard library)
#   load_experiment -> local function in this file
#   np.ceil -> numpy.ceil (third-party package)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame([summary]).to_csv -> pandas.DataFrame([summary]).to_csv (third-party package)
#   pd.DataFrame(diagnostics).to_csv -> pandas.DataFrame(diagnostics).to_csv (third-party package)
#   pd.concat -> pandas.concat (third-party package)
#   pd.concat(predictions, ignore_index=True).to_csv -> pandas.concat(predictions, ignore_index=True).to_csv (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   score_summary -> local function in this file
#   time.perf_counter -> time.perf_counter (Python standard library)
def run_batched(
    candidate_name: str,
    phase: str,
    subset_count: int | None,
    batch_size: int,
    maxiter_override: int | None = None,
) -> dict:
    candidate = candidate_by_name(candidate_name)
    config = json.loads((ROOT / "pilot_config.json").read_text(encoding="utf-8"))
    raw, validation, test, audit = load_experiment()
    score = validation if phase == "validation" else test
    cutoff = EXP.validation_cutoff if phase == "validation" else EXP.test_cutoff
    if subset_count is None:
        base_name = "nssp_full"
    elif maxiter_override is None:
        base_name = f"nssp_screen_{subset_count}"
    else:
        base_name = f"nssp_rescue_{subset_count}_iter{maxiter_override}"
    base = OUT / base_name
    run_dir = base / phase / candidate_name
    batch_dir = run_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    keys = group_keys(test)
    if subset_count is not None:
        keys = core.choose_nssp_groups(raw, test, EXP.test_cutoff, subset_count, int(config["random_seed"]))
        screening_path = base / "screening_groups.csv"
        expected_screen = pd.DataFrame(keys, columns=EXP.groups)
        if screening_path.exists():
            existing_screen = pd.read_csv(screening_path, dtype=str)
            if set(existing_screen[list(EXP.groups)].itertuples(index=False, name=None)) != set(keys):
                raise RuntimeError("Existing screening group file does not match deterministic selection")
        else:
            expected_screen.to_csv(screening_path, index=False)
    allowed = set(keys)
    score = score[
        [key in allowed for key in score[list(EXP.groups)].itertuples(index=False, name=None)]
    ].copy()
    raw_groups = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in raw.groupby(list(EXP.groups), sort=False)
        if (key if isinstance(key, tuple) else (key,)) in allowed
    }
    score_groups = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in score.groupby(list(EXP.groups), sort=False)
    }
    maxiter = int(config["maxiter"] if maxiter_override is None else maxiter_override)
    started = time.perf_counter()

    for batch_index, offset in enumerate(range(0, len(keys), batch_size), start=1):
        batch_keys = keys[offset : offset + batch_size]
        pred_path = batch_dir / f"batch_{batch_index:04d}_predictions.csv"
        diag_path = batch_dir / f"batch_{batch_index:04d}_diagnostics.csv"
        if pred_path.exists() and diag_path.exists():
            print(f"{candidate_name} {phase} batch {batch_index}: checkpoint exists", flush=True)
            continue
        predictions = []
        diagnostics = []
        for key in batch_keys:
            if key not in score_groups or key not in raw_groups:
                continue
            pred, outcome = core.fit_and_predict_group(
                raw_groups[key], EXP, score_groups[key], cutoff, candidate, maxiter
            )
            predictions.append(pred)
            diagnostics.append({**dict(zip(EXP.groups, key)), **outcome})
        if not predictions:
            raise RuntimeError(f"No predictions produced for batch {batch_index}")
        pd.concat(predictions, ignore_index=True).to_csv(pred_path, index=False)
        pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
        print(
            f"{candidate_name} {phase} batch {batch_index}/"
            f"{int(np.ceil(len(keys) / batch_size))} complete",
            flush=True,
        )

    pred_files = sorted(batch_dir.glob("batch_*_predictions.csv"))
    diag_files = sorted(batch_dir.glob("batch_*_diagnostics.csv"))
    predictions = pd.concat([pd.read_csv(path) for path in pred_files], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in diag_files], ignore_index=True)
    summary = score_summary(predictions, diagnostics, candidate, phase)
    summary["Wall_Seconds_Current_Invocation"] = time.perf_counter() - started
    summary["Maxiter"] = maxiter
    summary["Cohort_Audit"] = json.dumps(audit, sort_keys=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    diagnostics.to_csv(run_dir / "diagnostics.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "COMPLETE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


# Calculation provenance:
#   pd.concat -> pandas.concat (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
def collect_summaries(base_name: str, phase: str) -> pd.DataFrame:
    root = OUT / base_name / phase
    paths = sorted(root.glob("*/summary.csv"))
    if not paths:
        raise RuntimeError(f"No summaries found below {root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.sort_values(["RMSE", "MAE"]).to_csv(OUT / base_name / f"{phase}_candidate_summary.csv", index=False)
    return frame


# Calculation provenance:
#   candidate_by_name -> local function in this file
#   collect_summaries -> local function in this file
#   json.dumps -> json.dumps (Python standard library)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(rows).drop_duplicates -> pandas.DataFrame(rows).drop_duplicates (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.read_csv(path).iloc[0].to_dict -> pandas.read_csv(path).iloc[0].to_dict (third-party package)
#   score_summary -> local function in this file
def methodological_decision(screen_count: int) -> dict:
    full = collect_summaries("nssp_full", "validation")
    screen_root = OUT / f"nssp_screen_{screen_count}"
    screen_groups = pd.read_csv(screen_root / "screening_groups.csv", dtype=str)
    allowed = set(screen_groups[list(EXP.groups)].itertuples(index=False, name=None))

    rows = []
    # Re-score completed full-profile predictions on the identical screening groups.
    for path in sorted((OUT / "nssp_full" / "validation").glob("*/predictions.csv")):
        candidate = path.parent.name
        pred = pd.read_csv(path)
        for column in EXP.groups:
            pred[column] = pred[column].astype(str)
        subset = pred[
            [key in allowed for key in pred[list(EXP.groups)].itertuples(index=False, name=None)]
        ]
        diag = pd.read_csv(path.parent / "diagnostics.csv")
        for column in EXP.groups:
            diag[column] = diag[column].astype(str)
        diag = diag[
            [key in allowed for key in diag[list(EXP.groups)].itertuples(index=False, name=None)]
        ]
        rows.append(score_summary(subset, diag, candidate_by_name(candidate), "screen_validation"))

    for path in sorted((screen_root / "validation").glob("*/summary.csv")):
        rows.append(pd.read_csv(path).iloc[0].to_dict())
    comparison = pd.DataFrame(rows).drop_duplicates("Candidate", keep="last")
    comparison.sort_values(["RMSE", "MAE"]).to_csv(screen_root / "screening_candidate_comparison.csv", index=False)

    practical = comparison[~comparison["Candidate"].str.contains("x011_52")]
    reference_rmse = float(practical[practical["Seasonal"].astype(str).str.lower().isin(["true", "1"])]["RMSE"].min())
    slow = comparison[comparison["Candidate"].str.contains("x011_52")].copy()
    slow["RMSE_Ratio_to_Best_Practical"] = slow["RMSE"] / reference_rmse
    slow["Retain_Full"] = (
        (slow["Coverage"] >= 0.95)
        & (slow["Convergence_Rate"] >= 0.90)
        & (slow["RMSE_Ratio_to_Best_Practical"] <= 1.01)
    )
    retained = slow.loc[slow["Retain_Full"], "Candidate"].tolist()
    decision = {
        "screen_count": screen_count,
        "best_practical_screen_rmse": reference_rmse,
        "retention_rule": "coverage>=0.95, convergence>=0.90, RMSE<=1.01*best practical seasonal RMSE",
        "retained_for_full_validation": retained,
        "excluded_after_screen": slow.loc[~slow["Retain_Full"], "Candidate"].tolist(),
        "slow_candidate_details": slow.to_dict(orient="records"),
        "final_test_used_for_decision": False,
    }
    (screen_root / "METHODOLOGICAL_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2), flush=True)
    return decision


# Calculation provenance:
#   collect_summaries -> local function in this file
#   json.dumps -> json.dumps (Python standard library)
def select_candidate() -> dict:
    frame = collect_summaries("nssp_full", "validation")
    eligible = frame[
        frame["Seasonal"].astype(str).str.lower().isin(["true", "1"])
        & (frame["Coverage"] >= 0.95)
        & (frame["Convergence_Rate"] >= 0.90)
    ].sort_values(["RMSE", "MAE"])
    if eligible.empty:
        raise RuntimeError("No seasonal candidate passed the full validation gates")
    selected = eligible.iloc[0].to_dict()
    selected["Selection_Rule"] = "lowest pooled validation RMSE, then MAE; coverage>=0.95; convergence>=0.90"
    selected["Final_Test_Used"] = False
    target = OUT / "nssp_full" / "SELECTED_CANDIDATE.json"
    target.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2), flush=True)
    return selected


# Calculation provenance:
#   core.locked_paths -> sarima_pilot.locked_paths (benchmarks/sarima/sarima_pilot.py)
#   core.metrics -> sarima_pilot.metrics (benchmarks/sarima/sarima_pilot.py)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame(rows).to_csv -> pandas.DataFrame(rows).to_csv (third-party package)
#   pd.DataFrame(rows).to_string -> pandas.DataFrame(rows).to_string (third-party package)
#   pd.read_csv -> pandas.read_csv (third-party package)
def final_comparison(candidate_name: str) -> None:
    pred = pd.read_csv(OUT / "nssp_full" / "test" / candidate_name / "predictions.csv")
    locked_path, _ = core.locked_paths(EXP)
    locked = pd.read_csv(locked_path, low_memory=False)
    keys = [*EXP.groups, "epi_order"]
    for column in EXP.groups:
        pred[column] = pred[column].astype(str)
        locked[column] = locked[column].astype(str)
    merged = pred[keys + ["prediction"]].merge(
        locked[keys + [EXP.target, "Hybrid_InvMSE", "Persistence", "MovingAverage3", "SeasonalNaive"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(locked) or merged["Hybrid_InvMSE"].isna().any():
        raise RuntimeError("Final comparison does not match the locked 47,534-row cohort")
    additional_models = []
    ar1_path = OUT / "nssp_full" / "test" / "AR1" / "predictions.csv"
    if candidate_name != "AR1" and ar1_path.exists():
        ar1 = pd.read_csv(ar1_path)
        for column in EXP.groups:
            ar1[column] = ar1[column].astype(str)
        merged = merged.merge(
            ar1[keys + ["prediction"]].rename(columns={"prediction": "AR1_prediction"}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        additional_models.append(("AR1", "AR1_prediction"))
    rows = []
    for model, column in (
        (candidate_name, "prediction"),
        *additional_models,
        ("Hybrid_InvMSE", "Hybrid_InvMSE"),
        ("Persistence", "Persistence"),
        ("MovingAverage3", "MovingAverage3"),
        ("SeasonalNaive", "SeasonalNaive"),
    ):
        rows.append({"Model": model, **core.metrics(merged[EXP.target].to_numpy(float), merged[column].to_numpy(float))})
    pd.DataFrame(rows).to_csv(OUT / "nssp_full" / "FINAL_SAME_ROW_COMPARISON.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   final_comparison -> local function in this file
#   methodological_decision -> local function in this file
#   run_batched -> local function in this file
#   select_candidate -> local function in this file
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--candidate", required=True)
    run.add_argument("--phase", choices=["validation", "test"], required=True)
    run.add_argument("--subset-count", type=int)
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--maxiter", type=int)
    decision = sub.add_parser("decision")
    decision.add_argument("--screen-count", type=int, default=120)
    sub.add_parser("select")
    compare = sub.add_parser("compare")
    compare.add_argument("--candidate", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_batched(
            args.candidate,
            args.phase,
            args.subset_count,
            args.batch_size,
            args.maxiter,
        )
    elif args.command == "decision":
        methodological_decision(args.screen_count)
    elif args.command == "select":
        select_candidate()
    else:
        final_comparison(args.candidate)


if __name__ == "__main__":
    main()
