# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
# Presentation only. plot_state_choropleth is a legacy ranked bar plot, not the final manuscript heat map. Final descriptive maps come from manuscript/latex/build_review_figures.py.
"""Publication-quality figure helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_io import epi_week_start


DISPLAY_NAMES = {
    "Hybrid_InvMSE": "Hybrid Ensemble",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "MLP": "MLP",
    "Persistence": "Persistence",
    "MovingAverage3": "Moving Average-3",
    "SeasonalNaive": "Seasonal Naïve",
}


# Calculation provenance:
#   Path -> pathlib.Path (Python standard library)
#   plt.close -> matplotlib.pyplot.close (third-party package)
#   plt.savefig -> matplotlib.pyplot.savefig (third-party package)
#   plt.tight_layout -> matplotlib.pyplot.tight_layout (third-party package)
def _save_figure(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Calculation provenance:
#   epi_week_start -> common_io.epi_week_start (src/common_io.py)
#   pd.to_datetime -> pandas.to_datetime (third-party package)
def _plot_dates(frame: pd.DataFrame) -> pd.Series:
    """Use CDC MMWR dates for ILINet, including saved legacy ISO proxies."""
    if {"REGION TYPE", "YEAR", "WEEK"}.issubset(frame.columns):
        return epi_week_start(frame["YEAR"], frame["WEEK"])
    return pd.to_datetime(frame["date"], errors="coerce")


# Calculation provenance:
#   _plot_dates -> local function in this file
#   _save_figure -> local function in this file
#   plt.figure -> matplotlib.pyplot.figure (third-party package)
#   plt.grid -> matplotlib.pyplot.grid (third-party package)
#   plt.legend -> matplotlib.pyplot.legend (third-party package)
#   plt.plot -> matplotlib.pyplot.plot (third-party package)
#   plt.title -> matplotlib.pyplot.title (third-party package)
#   plt.xlabel -> matplotlib.pyplot.xlabel (third-party package)
#   plt.ylabel -> matplotlib.pyplot.ylabel (third-party package)
def plot_actual_vs_predictions(
    pred_df: pd.DataFrame,
    target: str,
    output_path: str | Path,
    title: str,
):
    columns = [
        target,
        "Hybrid_InvMSE",
        "RandomForest",
        "GradientBoosting",
        "MLP",
    ]
    available = [column for column in columns if column in pred_df.columns]
    plot_df = (
        pred_df.assign(date=_plot_dates(pred_df)).dropna(subset=["date"])
        .groupby("date", as_index=False)[available]
        .mean(numeric_only=True)
        .sort_values("date")
    )

    plt.figure(figsize=(10, 5.5))
    plt.plot(plot_df["date"], plot_df[target], label="Observed", linewidth=2.2)
    for column in available:
        if column == target:
            continue
        plt.plot(
            plot_df["date"],
            plot_df[column],
            label=DISPLAY_NAMES.get(column, column),
            linewidth=2.0 if column == "Hybrid_InvMSE" else 1.4,
            alpha=0.9,
        )
    plt.title(title)
    plt.xlabel("Week")
    plt.ylabel("Mean surveillance outcome")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _save_figure(output_path)


# Calculation provenance:
#   _save_figure -> local function in this file
#   plt.bar -> matplotlib.pyplot.bar (third-party package)
#   plt.figure -> matplotlib.pyplot.figure (third-party package)
#   plt.grid -> matplotlib.pyplot.grid (third-party package)
#   plt.text -> matplotlib.pyplot.text (third-party package)
#   plt.title -> matplotlib.pyplot.title (third-party package)
#   plt.xticks -> matplotlib.pyplot.xticks (third-party package)
#   plt.ylabel -> matplotlib.pyplot.ylabel (third-party package)
def plot_metric_bar(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric: str = "RMSE",
    title: str = "Model comparison",
):
    df = metrics_df.dropna(subset=[metric]).sort_values(metric).copy()
    df["DisplayName"] = df["Model"].map(DISPLAY_NAMES).fillna(df["Model"])

    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["DisplayName"], df[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, df[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save_figure(output_path)


# Calculation provenance:
#   _plot_dates -> local function in this file
#   _save_figure -> local function in this file
#   pd.to_datetime -> pandas.to_datetime (third-party package)
#   plt.subplots -> matplotlib.pyplot.subplots (third-party package)
def plot_residuals(
    pred_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Residual Analysis",
    event_line: str | None = None,
    event_label: str = "Event",
    target_col: str | None = None,
):
    """Plot weekly residual summaries and the row-level residual distribution."""
    if target_col is None or target_col not in pred_df.columns:
        raise KeyError("A valid target_col is required for residual analysis.")
    required = {"date", target_col, "Hybrid_InvMSE"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise KeyError(f"Residual plot is missing required columns: {sorted(missing)}")

    # The histogram includes every scored prediction, even if a malformed
    # display date is encountered. Date validity only limits the weekly panel.
    frame = pred_df.assign(date=_plot_dates(pred_df))[
        ["date", target_col, "Hybrid_InvMSE"]
    ].dropna(subset=[target_col, "Hybrid_InvMSE"]).copy()
    frame["Residual"] = frame["Hybrid_InvMSE"] - frame[target_col]
    weekly = frame.dropna(subset=["date"]).groupby("date")["Residual"].agg(
        median="median",
        mean="mean",
        p05=lambda values: values.quantile(0.05),
        p95=lambda values: values.quantile(0.95),
    ).reset_index().sort_values("date")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].fill_between(
        weekly["date"],
        weekly["p05"],
        weekly["p95"],
        alpha=0.2,
        label="5th–95th percentile",
    )
    axes[0].plot(weekly["date"], weekly["median"], linewidth=1.8, label="Weekly median")
    axes[0].plot(weekly["date"], weekly["mean"], linewidth=1.2, linestyle="--", label="Weekly mean")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    if event_line:
        axes[0].axvline(
            pd.to_datetime(event_line),
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=event_label,
        )
    axes[0].set_ylabel("Residual (Predicted − Observed)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].hist(frame["Residual"], bins=50, edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.25)

    _save_figure(output_path)


# Calculation provenance:
#   _save_figure -> local function in this file
#   plt.barh -> matplotlib.pyplot.barh (third-party package)
#   plt.figure -> matplotlib.pyplot.figure (third-party package)
#   plt.grid -> matplotlib.pyplot.grid (third-party package)
#   plt.text -> matplotlib.pyplot.text (third-party package)
#   plt.title -> matplotlib.pyplot.title (third-party package)
#   plt.xlabel -> matplotlib.pyplot.xlabel (third-party package)
def plot_state_choropleth(
    state_df: pd.DataFrame,
    value_col: str,
    output_path: str | Path,
    title: str,
    geo_col: str = "REGION",
    top_n: int = 15,
):
    """Create a ranked state bar chart for the highest mean predictions."""
    if geo_col not in state_df.columns:
        raise KeyError(f"Geographic column not found: {geo_col}")
    if value_col not in state_df.columns:
        raise KeyError(f"Value column not found: {value_col}")

    state_abbrev = {
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
        "Wisconsin": "WI", "Wyoming": "WY",
    }

    plot_df = state_df[[geo_col, value_col]].dropna().copy()
    plot_df = plot_df[~plot_df[geo_col].astype(str).str.lower().isin({"united states", "national"})]
    plot_df["State"] = plot_df[geo_col].map(state_abbrev).fillna(plot_df[geo_col])
    plot_df = plot_df.sort_values(value_col, ascending=False).head(top_n).sort_values(value_col)

    plt.figure(figsize=(8, 6.5))
    bars = plt.barh(plot_df["State"], plot_df[value_col])
    plt.xlabel("Mean predicted surveillance outcome")
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, plot_df[value_col]):
        plt.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=8)
    _save_figure(output_path)
