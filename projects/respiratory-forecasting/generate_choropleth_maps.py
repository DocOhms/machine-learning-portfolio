# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Generate static state-level PNG maps from saved experiment predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "experiment_1_nssp_primary",
    "experiment_2_ilinet_contemporary",
    "experiment_3_ilinet_pandemic_stress_test",
]

STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "New York City": "NY",
}
VALID_ABBREVIATIONS = set(STATE_ABBREV.values())


def get_location_column(df: pd.DataFrame) -> str:
    for column in ["REGION", "region", "geography", "state", "STATE"]:
        if column in df.columns:
            return column
    raise KeyError("No state/location column was found in the predictions file.")


# Calculation provenance:
#   pd.isna -> pandas.isna (third-party package)
def normalize_state(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in VALID_ABBREVIATIONS:
        return text
    return STATE_ABBREV.get(text)


# Calculation provenance:
#   get_location_column -> local function in this file
def aggregate_to_state(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    location_col = get_location_column(df)
    target = (
        "percent_visits_combined"
        if "nssp" in experiment_name.lower()
        else "%UNWEIGHTED ILI"
    )
    required = {target, "Hybrid_InvMSE", location_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Predictions file is missing columns: {sorted(missing)}")

    frame = df.copy()
    frame["state"] = frame[location_col].map(normalize_state)
    frame = frame.dropna(subset=["state", target, "Hybrid_InvMSE"])
    state_df = (
        frame.groupby("state", as_index=False)[[target, "Hybrid_InvMSE"]]
        .mean(numeric_only=True)
        .rename(columns={target: "Observed"})
    )
    state_df["Residual"] = state_df["Hybrid_InvMSE"] - state_df["Observed"]
    return state_df


# Calculation provenance:
#   px.choropleth -> plotly.express.choropleth (third-party package)
def save_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    *,
    residual: bool = False,
) -> None:
    kwargs = {
        "data_frame": df,
        "locations": "state",
        "locationmode": "USA-states",
        "color": value_col,
        "scope": "usa",
        "title": title,
    }
    if residual:
        limit = float(df[value_col].abs().max())
        kwargs.update(
            color_continuous_scale="RdBu_r",
            range_color=(-limit, limit),
            color_continuous_midpoint=0,
        )
    else:
        kwargs.update(color_continuous_scale="Reds")

    fig = px.choropleth(**kwargs)
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar_title=(
            "Residual (percentage points)" if residual else "Mean predicted burden"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), width=1200, height=700, scale=2)
    print(f"Saved: {output_path}")


# Calculation provenance:
#   aggregate_to_state -> local function in this file
#   pd.read_csv -> pandas.read_csv (third-party package)
#   save_map -> local function in this file
def process_experiment(project_root: Path, experiment_name: str) -> None:
    pred_path = (
        project_root
        / "results"
        / experiment_name
        / "predictions"
        / f"{experiment_name}_predictions.csv"
    )
    if not pred_path.exists():
        print(f"Skipping missing file: {pred_path}")
        return

    state_df = aggregate_to_state(pd.read_csv(pred_path), experiment_name)
    output_dir = project_root / "results" / experiment_name / "figures" / "choropleth"
    readable = experiment_name.replace("_", " ").title().replace("Nssp", "NSSP").replace("Ilinet", "ILINet")

    save_map(
        state_df,
        "Hybrid_InvMSE",
        f"{readable} — Mean Test-Period Predicted Burden",
        output_dir / f"{experiment_name}_predicted_hotspots.png",
    )
    save_map(
        state_df,
        "Residual",
        f"{readable} — Mean Test-Period Residual",
        output_dir / f"{experiment_name}_residuals.png",
        residual=True,
    )
    state_df.to_csv(
        project_root
        / "results"
        / experiment_name
        / "metrics"
        / f"{experiment_name}_state_map_values.csv",
        index=False,
    )


# Calculation provenance:
#   argparse.ArgumentParser -> argparse.ArgumentParser (Python standard library)
#   process_experiment -> local function in this file
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project directory containing the results folder.",
    )
    args = parser.parse_args()
    for experiment in EXPERIMENTS:
        process_experiment(args.project_root.resolve(), experiment)


if __name__ == "__main__":
    main()
