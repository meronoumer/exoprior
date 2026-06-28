"""
NASA Exoplanet Archive TAP client.

Fetches the confirmed transiting planet catalog via ADQL from the public TAP endpoint,
saves the raw response as Parquet, and records the exact query used alongside the data.

Filter choices
--------------
default_flag = 1
    The `ps` table (Planetary Systems) contains one row per reference per planet.
    Multiple literature sources may report different parameter values for the same planet.
    `default_flag = 1` selects the single "default" parameter set that the Archive
    curators consider the best available for each planet. Without this filter, a planet
    like WASP-39 b would appear dozens of times with conflicting radii, masses, etc.
    We inherit the Archive's curation judgment and document that here rather than
    attempting our own aggregation.

tran_flag = 1
    The Transmission Spectroscopy Metric (TSM) and Emission Spectroscopy Metric (ESM)
    — the two primary feature inputs for our scoring model — are only physically
    meaningful for planets with confirmed transit detections. `tran_flag = 1` restricts
    the catalog to such planets. Non-transiting planets are out of scope for MVP.

Column provenance
-----------------
Column names were verified against TAP_SCHEMA.columns on 2026-06-26.
- Distance uses `sy_dist` (system-level; `st_dist` does not exist in this schema).
- J-band magnitude uses `sy_jmag` (`st_j` does not exist in this schema).
- Mass uses `pl_masse` (true mass); `pl_bmasse` (mass or m*sin(i)) is also fetched
  as a fallback column for planets where inclination is unknown.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns confirmed present in ps table via TAP_SCHEMA.columns query on 2026-06-26.
# Grouped by domain for readability.
QUERY_COLUMNS = [
    # Identity
    "pl_name",
    "hostname",
    "pl_letter",
    # Flags
    "default_flag",
    "tran_flag",
    # Planet geometry
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    # Planet mass — true mass where available; bmasse as fallback
    "pl_masse",
    "pl_masseerr1",
    "pl_masseerr2",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    # Orbital parameters
    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_orbsmax",
    # Transit observables
    "pl_trandep",
    "pl_trandeperr1",
    "pl_trandeperr2",
    "pl_tranmid",
    "pl_trandur",
    # Insolation and equilibrium temperature
    "pl_insol",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_eqt",
    "pl_eqterr1",
    "pl_eqterr2",
    # Stellar parameters
    "st_teff",
    "st_tefferr1",
    "st_tefferr2",
    "st_rad",
    "st_raderr1",
    "st_raderr2",
    "st_mass",
    "st_logg",
    # System-level photometry and astrometry
    # NOTE: sy_dist replaces the non-existent st_dist; sy_jmag replaces non-existent st_j
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_jmag",
    "sy_jmagerr1",
    "sy_jmagerr2",
    "sy_kmag",
    "sy_tmag",
    # Spectral type
    "st_spectype",
    # Reference
    "pl_refname",
    "st_refname",
    "rowupdate",
]

_ADQL = (
    "SELECT {cols} "
    "FROM ps "
    "WHERE default_flag = 1 AND tran_flag = 1"
).format(cols=",".join(QUERY_COLUMNS))


def fetch(
    raw_dir: Path,
    *,
    timeout: int = 120,
    force: bool = False,
) -> pd.DataFrame:
    """Download the transiting planet catalog from NASA TAP and persist it.

    Parameters
    ----------
    raw_dir:
        Directory where raw Parquet and query metadata are written.
    timeout:
        HTTP request timeout in seconds.
    force:
        If True, re-download even if a cached file exists.

    Returns
    -------
    DataFrame with one row per planet (default parameter set, transit-confirmed).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = raw_dir / "ps_transiting_default.parquet"
    meta_path = raw_dir / "ps_transiting_default_query.json"

    if parquet_path.exists() and not force:
        return pd.read_parquet(parquet_path)

    params = {
        "QUERY": _ADQL,
        "FORMAT": "csv",
        "lang": "ADQL",
        "REQUEST": "doQuery",
        "SERVICE": "TAP",
    }

    response = requests.get(TAP_ENDPOINT, params=params, timeout=timeout)
    response.raise_for_status()

    raw_csv = response.text
    df = pd.read_csv(pd.io.common.StringIO(raw_csv))

    df.to_parquet(parquet_path, index=False)

    meta = {
        "query": _ADQL,
        "endpoint": TAP_ENDPOINT,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "query_sha256": hashlib.sha256(_ADQL.encode()).hexdigest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return df


def _main() -> None:
    import argparse
    import sys

    from exoprior.ingest.schema import SchemaError, validate

    parser = argparse.ArgumentParser(
        prog="python -m exoprior.ingest.tap_client",
        description="Fetch the NASA TAP transiting-planet catalog and save it to data/raw/.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory for raw Parquet and query metadata (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a cached file already exists",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    print(f"Fetching catalog from {TAP_ENDPOINT} …")
    df = fetch(raw_dir, force=args.force)

    print("Validating schema …")
    try:
        validate(df)
    except SchemaError as exc:
        print(f"\nSchema validation FAILED:\n{exc}", file=sys.stderr)
        sys.exit(1)

    parquet_path = raw_dir / "ps_transiting_default.parquet"
    meta_path = raw_dir / "ps_transiting_default_query.json"

    print(f"\nOutput files:")
    print(f"  Parquet : {parquet_path.resolve()}")
    print(f"  Metadata: {meta_path.resolve()}")

    print(f"\nCatalog shape: {len(df)} rows × {len(df.columns)} columns")

    null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    any_missing = null_pct[null_pct > 0]
    if any_missing.empty:
        print("\nMissingness: none")
    else:
        print(f"\nMissingness (columns with >0% null, descending):")
        for col, pct in any_missing.items():
            print(f"  {col:<30} {pct:5.1f}%")


if __name__ == "__main__":
    _main()
