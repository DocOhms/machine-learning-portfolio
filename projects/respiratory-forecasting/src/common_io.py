# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
"""Data loading and output helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


# Calculation provenance:
#   pd.to_numeric -> pandas.to_numeric (third-party package)
def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert common CDC suppression/missing-value markers to numeric NaN."""
    return pd.to_numeric(
        series.replace({"X": np.nan, "x": np.nan, "": np.nan}),
        errors="coerce",
    )


# Calculation provenance:
#   pd.Series -> pandas.Series (third-party package)
#   pd.Timedelta -> pandas.Timedelta (third-party package)
#   pd.Timestamp -> pandas.Timestamp (third-party package)
def epi_week_start(year: pd.Series, week: pd.Series) -> pd.Series:
    """Return the Sunday beginning a CDC MMWR epidemiological week.

    MMWR week 1 is the Sunday--Saturday week containing January 4. This is
    not ISO week numbering: 2025-W53 is valid and begins December 28, 2025.
    YEAR/WEEK and epi_order remain unchanged and are authoritative model keys;
    this conversion is for calendar display and date-based summaries only.
    """
    vals: list[pd.Timestamp | pd.NaT] = []
    for y, w in zip(year, week):
        try:
            y, w = int(y), int(w)
            jan4 = pd.Timestamp(year=y, month=1, day=4)
            next_jan4 = pd.Timestamp(year=y + 1, month=1, day=4)
            first = jan4 - pd.Timedelta(days=(jan4.dayofweek + 1) % 7)
            next_first = next_jan4 - pd.Timedelta(days=(next_jan4.dayofweek + 1) % 7)
            weeks_in_year = (next_first - first).days // 7
            if not 1 <= w <= weeks_in_year:
                raise ValueError("Invalid MMWR week for year")
            vals.append(first + pd.Timedelta(weeks=w - 1))
        except (TypeError, ValueError, OverflowError):
            vals.append(pd.NaT)
    return pd.Series(vals, index=year.index)


# Calculation provenance:
#   clean_numeric -> local function in this file
#   pd.read_csv -> pandas.read_csv (third-party package)
#   pd.to_datetime -> pandas.to_datetime (third-party package)
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


# Calculation provenance:
#   clean_numeric -> local function in this file
#   epi_week_start -> local function in this file
#   pd.read_csv -> pandas.read_csv (third-party package)
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


# Calculation provenance:
#   Path -> pathlib.Path (Python standard library)
def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
