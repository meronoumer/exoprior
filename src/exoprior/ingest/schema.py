"""
Raw catalog schema assertions.

These checks run immediately after fetch() returns. Their purpose is to detect
Archive schema changes early — if NASA renames or drops a column we depend on,
we want a loud error here rather than silent NaN propagation downstream.

Rules
-----
- REQUIRED_COLUMNS: must be present in the DataFrame; values may still be null.
- NEVER_NULL: must be present and fully populated.
- NUMERIC_COLUMNS: must be castable to float64 (NaN allowed).

None of these checks constitute data cleaning. They only assert the raw structure.
"""

from __future__ import annotations

import pandas as pd

# Columns we cannot proceed without at all.
NEVER_NULL: list[str] = [
    "pl_name",
    "hostname",
    "default_flag",
    "tran_flag",
]

# Columns that must exist; NULLs are expected and handled in cleaning.
REQUIRED_COLUMNS: list[str] = [
    "pl_name",
    "hostname",
    "pl_letter",
    "default_flag",
    "tran_flag",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_masse",
    "pl_masseerr1",
    "pl_masseerr2",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_orbsmax",
    "pl_trandep",
    "pl_trandeperr1",
    "pl_trandeperr2",
    "pl_tranmid",
    "pl_trandur",
    "pl_insol",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_eqt",
    "pl_eqterr1",
    "pl_eqterr2",
    "st_teff",
    "st_tefferr1",
    "st_tefferr2",
    "st_rad",
    "st_raderr1",
    "st_raderr2",
    "st_mass",
    "st_logg",
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_jmag",
    "sy_jmagerr1",
    "sy_jmagerr2",
    "sy_kmag",
    "sy_tmag",
    "st_spectype",
    "pl_refname",
    "st_refname",
    "rowupdate",
]

NUMERIC_COLUMNS: list[str] = [
    c for c in REQUIRED_COLUMNS
    if c not in ("pl_name", "hostname", "pl_letter", "st_spectype",
                 "pl_refname", "st_refname", "rowupdate")
]


class SchemaError(ValueError):
    """Raised when the raw DataFrame violates expected structure."""


def validate(df: pd.DataFrame) -> None:
    """Assert raw catalog structure. Raises SchemaError on any violation.

    Parameters
    ----------
    df:
        Raw DataFrame returned by tap_client.fetch().
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Required columns missing from raw catalog: {missing}\n"
            "NASA may have renamed or removed these columns. "
            "Re-run TAP_SCHEMA.columns query to find current names."
        )

    for col in NEVER_NULL:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise SchemaError(
                f"Column '{col}' must never be null but has {n_null} null values."
            )

    for col in NUMERIC_COLUMNS:
        try:
            pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise SchemaError(
                f"Column '{col}' expected numeric but could not be cast: {exc}"
            ) from exc

    unexpected_flags = df.loc[df["default_flag"] != 1, "pl_name"]
    if len(unexpected_flags) > 0:
        raise SchemaError(
            f"Rows with default_flag != 1 found ({len(unexpected_flags)} rows). "
            "The TAP filter should have excluded these."
        )

    unexpected_tran = df.loc[df["tran_flag"] != 1, "pl_name"]
    if len(unexpected_tran) > 0:
        raise SchemaError(
            f"Rows with tran_flag != 1 found ({len(unexpected_tran)} rows). "
            "The TAP filter should have excluded these."
        )
