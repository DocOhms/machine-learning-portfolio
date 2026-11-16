"""Data loading and output helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert common CDC suppression/missing-value markers to numeric NaN."""
    return pd.to_numeric(
        series.replace({"X": np.nan, "x": np.nan, "": np.nan}),
        errors="coerce",
    )


def epi_week_start(year: pd.Series, week: pd.Series) -> pd.Series:
    """Return an ISO-week Monday date proxy without collapsing week 53.

    YEAR/WEEK numeric ordering remains the authoritative split key. Invalid ISO
    combinations are returned as NaT rather than silently remapped.
    """
    vals: list[pd.Timestamp | pd.NaT] = []
    for y, w in zip(year, week):
        try:
            vals.append(pd.Timestamp.fromisocalendar(int(y), int(w), 1))
        except (TypeError, ValueError):
            vals.append(pd.NaT)
    return pd.Series(vals, index=year.index)


def load_nssp(path: str | Path, target: str = "percent_visits_combined") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "week_end" not in df.columns:
        raise KeyError("NSSP file must contain a 'week_end' column.")
    if target not in df.columns:
        raise KeyError(f"NSSP target column not found: {target}")

    df["date"] = pd.to_datetime(df["week_end"], errors="coerce")
    iso = df["date"].dt.isocalendar()
    df["YEAR"] = iso.year.astype("Int64")
    df["WEEK"] = iso.week.astype("Int64")
    df = df.dropna(subset=["YEAR", "WEEK"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["WEEK"] = df["WEEK"].astype(int)
    df["epi_order"] = df["YEAR"] * 100 + df["WEEK"]

    for col in df.columns:
        if col.startswith("percent_visits_"):
            df[col] = clean_numeric(df[col])
    df[target] = clean_numeric(df[target])
    return df


def load_ilinet(path: str | Path, target: str = "%UNWEIGHTED ILI") -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, low_memory=False)
    required = {"YEAR", "WEEK", target}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"ILINet file is missing required columns: {sorted(missing)}")

    df["YEAR"] = clean_numeric(df["YEAR"]).astype("Int64")
    df["WEEK"] = clean_numeric(df["WEEK"]).astype("Int64")
    df = df.dropna(subset=["YEAR", "WEEK"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["WEEK"] = df["WEEK"].astype(int)
    df["epi_order"] = df["YEAR"] * 100 + df["WEEK"]
    df["date"] = epi_week_start(df["YEAR"], df["WEEK"])

    for col in df.columns:
        if col not in ["REGION TYPE", "REGION", "date"]:
            df[col] = clean_numeric(df[col])
    df[target] = clean_numeric(df[target])
    return df


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
