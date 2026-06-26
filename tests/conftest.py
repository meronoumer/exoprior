"""Shared pytest fixtures for ExoPrior tests."""

from __future__ import annotations

import pandas as pd
import pytest

from exoprior.ingest.schema import REQUIRED_COLUMNS


def _make_row(overrides: dict | None = None) -> dict:
    """Return a minimal valid raw-catalog row with all required columns present."""
    base = {
        "pl_name": "TestPlanet b",
        "hostname": "TestStar",
        "pl_letter": "b",
        "default_flag": 1,
        "tran_flag": 1,
        "pl_rade": 2.0,
        "pl_radeerr1": 0.1,
        "pl_radeerr2": -0.1,
        "pl_masse": 10.0,
        "pl_masseerr1": 1.0,
        "pl_masseerr2": -1.0,
        "pl_bmasse": 10.0,
        "pl_bmasseerr1": 1.0,
        "pl_bmasseerr2": -1.0,
        "pl_orbper": 5.0,
        "pl_orbpererr1": 0.01,
        "pl_orbpererr2": -0.01,
        "pl_orbsmax": 0.05,
        "pl_trandep": 0.5,
        "pl_trandeperr1": 0.02,
        "pl_trandeperr2": -0.02,
        "pl_tranmid": 2459000.0,
        "pl_trandur": 2.0,
        "pl_insol": 10.0,
        "pl_insolerr1": 0.5,
        "pl_insolerr2": -0.5,
        "pl_eqt": 800.0,
        "pl_eqterr1": 50.0,
        "pl_eqterr2": -50.0,
        "st_teff": 5500.0,
        "st_tefferr1": 100.0,
        "st_tefferr2": -100.0,
        "st_rad": 1.0,
        "st_raderr1": 0.05,
        "st_raderr2": -0.05,
        "st_mass": 1.0,
        "st_logg": 4.4,
        "sy_dist": 100.0,
        "sy_disterr1": 2.0,
        "sy_disterr2": -2.0,
        "sy_jmag": 9.0,
        "sy_jmagerr1": 0.02,
        "sy_jmagerr2": -0.02,
        "sy_kmag": 8.5,
        "sy_tmag": 9.2,
        "st_spectype": "G2V",
        "pl_refname": "Ref2024",
        "st_refname": "Ref2024",
        "rowupdate": "2024-01-01",
    }
    if overrides:
        base.update(overrides)
    return base


@pytest.fixture
def minimal_raw_df() -> pd.DataFrame:
    """A 3-row synthetic catalog that passes all schema checks."""
    rows = [
        _make_row({"pl_name": "Alpha b", "hostname": "Alpha"}),
        _make_row({"pl_name": "Beta b", "hostname": "Beta", "sy_jmag": 11.0}),
        _make_row({"pl_name": "Gamma b", "hostname": "Gamma", "pl_masse": float("nan")}),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def raw_row_factory():
    """Return a factory for custom raw-catalog rows."""
    return _make_row
