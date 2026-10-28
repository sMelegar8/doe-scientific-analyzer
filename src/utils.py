"""Utility functions for loading, validating, and coding DoE datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset and normalize column names.

    CSV files are read with automatic delimiter detection, which supports both
    comma-separated and semicolon-separated exports from laboratory software.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=None, engine="python")
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Input file must be a CSV or Excel file.")

    df.columns = df.columns.astype(str).str.strip()
    return df


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert text-like analytical values to numeric values.

    Values such as ``<0.1`` or ``>99`` are converted to their reported numeric
    limit. Censoring direction is intentionally ignored in this lightweight
    portfolio package; the original Streamlit app keeps a Tobit workflow for
    censored responses.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def validate_columns(df: pd.DataFrame, factors: Iterable[str], responses: Iterable[str]) -> None:
    """Raise a clear error when required factor or response columns are missing."""
    requested = list(factors) + list(responses)
    missing = [column for column in requested if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_analysis_frame(
    df: pd.DataFrame,
    factors: list[str],
    response: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Return a numeric analysis frame with coded factor columns.

    Each factor is coded to approximately ``[-1, 1]`` using the observed low and
    high values. Coded columns are named ``x_<factor>`` so formulas remain valid
    even when original column names contain spaces or units.
    """
    validate_columns(df, factors, [response])

    analysis_df = pd.DataFrame(index=df.index)
    coding: dict[str, dict[str, float]] = {}
    for factor in factors:
        numeric = clean_numeric_series(df[factor])
        low = float(numeric.min())
        high = float(numeric.max())
        center = (low + high) / 2.0
        half_range = (high - low) / 2.0 or 1.0
        coded_name = coded_column_name(factor)

        analysis_df[factor] = numeric
        analysis_df[coded_name] = (numeric - center) / half_range
        coding[factor] = {
            "low": low,
            "high": high,
            "center": center,
            "half_range": half_range,
        }

    analysis_df[response] = clean_numeric_series(df[response])
    analysis_df = analysis_df.dropna(subset=[response] + factors).copy()
    return analysis_df, coding


def coded_column_name(factor: str) -> str:
    """Return the deterministic coded-column name used for a factor."""
    safe = "".join(char if char.isalnum() else "_" for char in factor.strip())
    return f"x_{safe}"


def decode_factor_value(coded_value: float, factor: str, coding: dict[str, dict[str, float]]) -> float:
    """Convert a coded factor value back to the original engineering units."""
    return coded_value * coding[factor]["half_range"] + coding[factor]["center"]


def ensure_output_dir(path: str | Path) -> Path:
    """Create and return the output directory used for tables and plots."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
