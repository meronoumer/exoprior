"""
Tests for ingest/tap_client.py and ingest/schema.py.

All tests use synthetic fixtures. No live network calls are made.
TRAPPIST-1e and 55 Cnc e are NOT used as reference values here; their
archive parameters change across literature updates and are not pinned.
"""

from __future__ import annotations

import json
import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from exoprior.ingest import schema as sc
from exoprior.ingest import tap_client
from exoprior.ingest.schema import SchemaError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_csv_response(df: pd.DataFrame) -> MagicMock:
    """Wrap a DataFrame as a mock requests.Response returning CSV text."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = df.to_csv(index=False)
    return mock_resp


# ---------------------------------------------------------------------------
# tap_client tests
# ---------------------------------------------------------------------------

class TestTapClientQuery:
    def test_adql_contains_default_flag_filter(self):
        assert "default_flag = 1" in tap_client._ADQL

    def test_adql_contains_tran_flag_filter(self):
        assert "tran_flag = 1" in tap_client._ADQL

    def test_adql_table_is_ps(self):
        assert "FROM ps" in tap_client._ADQL

    def test_adql_uses_sy_dist_not_st_dist(self):
        assert "sy_dist" in tap_client._ADQL
        assert "st_dist" not in tap_client._ADQL

    def test_adql_uses_sy_jmag_not_st_j(self):
        assert "sy_jmag" in tap_client._ADQL
        assert "st_j" not in tap_client._ADQL

    def test_all_query_columns_in_adql(self):
        for col in tap_client.QUERY_COLUMNS:
            assert col in tap_client._ADQL, f"Column {col!r} missing from ADQL"


class TestTapClientFetch:
    def test_fetch_writes_parquet(self, minimal_raw_df, tmp_path):
        with patch("requests.get", return_value=_df_to_csv_response(minimal_raw_df)):
            result = tap_client.fetch(tmp_path)

        parquet = tmp_path / "ps_transiting_default.parquet"
        assert parquet.exists()
        reloaded = pd.read_parquet(parquet)
        assert len(reloaded) == len(minimal_raw_df)

    def test_fetch_writes_query_metadata(self, minimal_raw_df, tmp_path):
        with patch("requests.get", return_value=_df_to_csv_response(minimal_raw_df)):
            tap_client.fetch(tmp_path)

        meta_path = tmp_path / "ps_transiting_default_query.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "query" in meta
        assert "fetched_utc" in meta
        assert "query_sha256" in meta
        assert meta["row_count"] == len(minimal_raw_df)

    def test_fetch_metadata_query_matches_adql(self, minimal_raw_df, tmp_path):
        with patch("requests.get", return_value=_df_to_csv_response(minimal_raw_df)):
            tap_client.fetch(tmp_path)

        meta = json.loads((tmp_path / "ps_transiting_default_query.json").read_text())
        assert meta["query"] == tap_client._ADQL

    def test_fetch_uses_cache_on_second_call(self, minimal_raw_df, tmp_path):
        with patch("requests.get", return_value=_df_to_csv_response(minimal_raw_df)) as mock_get:
            tap_client.fetch(tmp_path)
            tap_client.fetch(tmp_path)
        assert mock_get.call_count == 1

    def test_fetch_force_bypasses_cache(self, minimal_raw_df, tmp_path):
        with patch("requests.get", return_value=_df_to_csv_response(minimal_raw_df)) as mock_get:
            tap_client.fetch(tmp_path)
            tap_client.fetch(tmp_path, force=True)
        assert mock_get.call_count == 2

    def test_fetch_raises_on_http_error(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="503"):
                tap_client.fetch(tmp_path)

    def test_fetch_returns_dataframe(self, minimal_raw_df, tmp_path):
        with patch("requests.get", return_value=_df_to_csv_response(minimal_raw_df)):
            result = tap_client.fetch(tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(minimal_raw_df)


# ---------------------------------------------------------------------------
# schema tests
# ---------------------------------------------------------------------------

class TestSchemaValidate:
    def test_valid_df_passes(self, minimal_raw_df):
        sc.validate(minimal_raw_df)  # must not raise

    def test_missing_column_raises(self, minimal_raw_df):
        df = minimal_raw_df.drop(columns=["pl_rade"])
        with pytest.raises(SchemaError, match="pl_rade"):
            sc.validate(df)

    def test_missing_pl_name_raises(self, minimal_raw_df):
        df = minimal_raw_df.drop(columns=["pl_name"])
        with pytest.raises(SchemaError):
            sc.validate(df)

    def test_null_pl_name_raises(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[0, "pl_name"] = None
        with pytest.raises(SchemaError, match="pl_name"):
            sc.validate(df)

    def test_null_hostname_raises(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[1, "hostname"] = None
        with pytest.raises(SchemaError, match="hostname"):
            sc.validate(df)

    def test_nullable_numeric_column_passes(self, minimal_raw_df):
        # pl_masse is allowed to be null (not in NEVER_NULL)
        df = minimal_raw_df.copy()
        df["pl_masse"] = float("nan")
        sc.validate(df)  # must not raise

    def test_non_numeric_in_numeric_column_raises(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df["st_teff"] = "not_a_number"
        with pytest.raises(SchemaError, match="st_teff"):
            sc.validate(df)

    def test_default_flag_violation_raises(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[0, "default_flag"] = 0
        with pytest.raises(SchemaError, match="default_flag"):
            sc.validate(df)

    def test_tran_flag_violation_raises(self, minimal_raw_df):
        df = minimal_raw_df.copy()
        df.loc[0, "tran_flag"] = 0
        with pytest.raises(SchemaError, match="tran_flag"):
            sc.validate(df)

    def test_error_message_mentions_schema_query(self, minimal_raw_df):
        df = minimal_raw_df.drop(columns=["sy_dist"])
        with pytest.raises(SchemaError, match="TAP_SCHEMA"):
            sc.validate(df)


class TestSchemaConstants:
    def test_never_null_is_subset_of_required(self):
        assert set(sc.NEVER_NULL).issubset(set(sc.REQUIRED_COLUMNS))

    def test_numeric_columns_excludes_string_columns(self):
        string_cols = {"pl_name", "hostname", "pl_letter", "st_spectype",
                       "pl_refname", "st_refname", "rowupdate"}
        for col in sc.NUMERIC_COLUMNS:
            assert col not in string_cols, f"{col!r} should not be in NUMERIC_COLUMNS"

    def test_sy_dist_in_required_not_st_dist(self):
        assert "sy_dist" in sc.REQUIRED_COLUMNS
        assert "st_dist" not in sc.REQUIRED_COLUMNS

    def test_sy_jmag_in_required_not_st_j(self):
        assert "sy_jmag" in sc.REQUIRED_COLUMNS
        assert "st_j" not in sc.REQUIRED_COLUMNS
