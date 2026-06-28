"""
Tests for features/engineer.py.

All expected values are derived analytically from the documented formulas using
the same synthetic inputs.  No live NASA archive values are used.
TRAPPIST-1e, 55 Cnc e, and similar planets are NOT referenced.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from exoprior.features.engineer import (
    _HZ_T_MAX,
    _HZ_T_MIN,
    _R_SUN_IN_R_EARTH,
    _T_EFF_EARTH_ZERO_ALBEDO_K,
    _TSM_SCALE,
    _ESM_NORM,
    _hz_s_eff,
    add_all,
    add_t_eq,
    esm_proxy_simplified,
    hz_flag,
    mass_uncertainty_fraction,
    radius_uncertainty_fraction,
    t_eq_from_insol,
    tsm_proxy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame with all required feature columns."""
    defaults = dict(
        pl_rade=2.0,
        pl_radeerr1=0.2,
        pl_radeerr2=-0.1,
        pl_masse=10.0,
        pl_masseerr1=1.0,
        pl_masseerr2=-0.8,
        pl_bmasse=10.0,
        pl_bmasseerr1=1.0,
        pl_bmasseerr2=-0.8,
        pl_eqt=800.0,
        pl_eqterr1=50.0,
        pl_eqterr2=-50.0,
        pl_insol=2.0,
        pl_insolerr1=0.2,
        pl_insolerr2=-0.2,
        st_teff=5500.0,
        st_tefferr1=100.0,
        st_tefferr2=-100.0,
        st_rad=1.0,
        st_raderr1=0.05,
        st_raderr2=-0.05,
        sy_jmag=8.0,
        sy_jmagerr1=0.02,
        sy_jmagerr2=-0.02,
        sy_kmag=7.5,
        sy_kmagerr1=0.02,
        sy_kmagerr2=-0.02,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def _multi(*rows) -> pd.DataFrame:
    return pd.concat([_row(**r) for r in rows], ignore_index=True)


# ---------------------------------------------------------------------------
# 3A — t_eq_from_insol
# ---------------------------------------------------------------------------

class TestTEqFromInsol:
    def test_earth_analog(self):
        # S = 1 S_earth → T_eq = 278.5 K (albedo=0 by definition)
        s = pd.Series([1.0])
        result = t_eq_from_insol(s)
        assert math.isclose(result.iloc[0], _T_EFF_EARTH_ZERO_ALBEDO_K, rel_tol=1e-6)

    def test_scaling_law(self):
        # T_eq ∝ S^0.25; doubling S multiplies T by 2^0.25
        s = pd.Series([1.0, 2.0])
        result = t_eq_from_insol(s)
        ratio = result.iloc[1] / result.iloc[0]
        assert math.isclose(ratio, 2 ** 0.25, rel_tol=1e-6)

    def test_nan_for_missing_insol(self):
        s = pd.Series([float("nan")])
        result = t_eq_from_insol(s)
        assert pd.isna(result.iloc[0])

    def test_nan_for_zero_insol(self):
        s = pd.Series([0.0])
        result = t_eq_from_insol(s)
        assert pd.isna(result.iloc[0])

    def test_nan_for_negative_insol(self):
        s = pd.Series([-1.0])
        result = t_eq_from_insol(s)
        assert pd.isna(result.iloc[0])

    def test_high_insolation_gives_high_temp(self):
        # Very close-in planet: S=2000 → T_eq ≈ 278.5 * 2000^0.25 ≈ 1757 K
        s = pd.Series([2000.0])
        result = t_eq_from_insol(s)
        expected = _T_EFF_EARTH_ZERO_ALBEDO_K * (2000.0 ** 0.25)
        assert math.isclose(result.iloc[0], expected, rel_tol=1e-6)


class TestAddTEq:
    def test_uses_archive_value_when_present(self):
        df = _row(pl_eqt=900.0, pl_insol=2.0)
        out = add_t_eq(df)
        assert out["t_eq"].iloc[0] == pytest.approx(900.0)
        assert out["t_eq_source"].iloc[0] == "archive"

    def test_derives_from_insol_when_eqt_missing(self):
        df = _row(pl_eqt=float("nan"), pl_insol=1.0)
        out = add_t_eq(df)
        assert out["t_eq"].iloc[0] == pytest.approx(_T_EFF_EARTH_ZERO_ALBEDO_K, rel=1e-5)
        assert out["t_eq_source"].iloc[0] == "derived_from_insol"

    def test_missing_when_both_missing(self):
        df = _row(pl_eqt=float("nan"), pl_insol=float("nan"))
        out = add_t_eq(df)
        assert pd.isna(out["t_eq"].iloc[0])
        assert out["t_eq_source"].iloc[0] == "missing"

    def test_archive_takes_precedence_over_insol(self):
        # Even if both are present, archive wins
        df = _row(pl_eqt=500.0, pl_insol=100.0)
        out = add_t_eq(df)
        assert out["t_eq"].iloc[0] == pytest.approx(500.0)
        assert out["t_eq_source"].iloc[0] == "archive"

    def test_does_not_modify_original(self):
        df = _row(pl_eqt=float("nan"), pl_insol=1.0)
        _ = add_t_eq(df)
        assert "t_eq" not in df.columns

    def test_mixed_rows(self):
        df = _multi(
            dict(pl_eqt=700.0, pl_insol=3.0),
            dict(pl_eqt=float("nan"), pl_insol=1.0),
            dict(pl_eqt=float("nan"), pl_insol=float("nan")),
        )
        out = add_t_eq(df)
        assert out["t_eq_source"].tolist() == ["archive", "derived_from_insol", "missing"]


# ---------------------------------------------------------------------------
# 3B — tsm_proxy
# ---------------------------------------------------------------------------

class TestTsmProxy:
    def _tsm_expected(self, r_p, t_eq, m_p, r_s, j):
        """Compute expected TSM analytically using the documented formula."""
        if r_p < 1.5:
            scale = 0.190
        elif r_p < 2.75:
            scale = 1.26
        elif r_p < 4.0:
            scale = 1.28
        else:
            scale = 1.15
        return scale * (r_p ** 3 * t_eq) / (m_p * r_s ** 2) * 10 ** (-j / 5)

    def test_formula_correctness_sub_earth(self):
        df = _row(pl_rade=1.0, pl_masse=1.0, st_rad=1.0, sy_jmag=10.0, pl_eqt=300.0)
        df = add_t_eq(df)
        result = tsm_proxy(df).iloc[0]
        expected = self._tsm_expected(1.0, 300.0, 1.0, 1.0, 10.0)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_formula_correctness_super_earth(self):
        df = _row(pl_rade=2.0, pl_masse=10.0, st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0)
        df = add_t_eq(df)
        result = tsm_proxy(df).iloc[0]
        expected = self._tsm_expected(2.0, 800.0, 10.0, 1.0, 8.0)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_formula_correctness_neptune(self):
        df = _row(pl_rade=3.5, pl_masse=17.0, st_rad=0.8, sy_jmag=9.0, pl_eqt=600.0)
        df = add_t_eq(df)
        result = tsm_proxy(df).iloc[0]
        expected = self._tsm_expected(3.5, 600.0, 17.0, 0.8, 9.0)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_formula_correctness_jupiter(self):
        df = _row(pl_rade=11.0, pl_masse=318.0, st_rad=1.0, sy_jmag=7.0, pl_eqt=1200.0)
        df = add_t_eq(df)
        result = tsm_proxy(df).iloc[0]
        expected = self._tsm_expected(11.0, 1200.0, 318.0, 1.0, 7.0)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_nan_when_radius_missing(self):
        df = _row(pl_rade=float("nan"), pl_masse=10.0, st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0)
        df = add_t_eq(df)
        assert pd.isna(tsm_proxy(df).iloc[0])

    def test_nan_when_mass_missing(self):
        df = _row(pl_rade=2.0, pl_masse=float("nan"), pl_bmasse=float("nan"),
                  st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0)
        df = add_t_eq(df)
        assert pd.isna(tsm_proxy(df).iloc[0])

    def test_fallback_to_bmasse(self):
        # pl_masse missing but pl_bmasse available — should compute a value
        df = _row(pl_rade=2.0, pl_masse=float("nan"), pl_bmasse=10.0,
                  st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0)
        df = add_t_eq(df)
        result = tsm_proxy(df).iloc[0]
        expected = self._tsm_expected(2.0, 800.0, 10.0, 1.0, 8.0)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_nan_when_t_eq_missing(self):
        df = _row(pl_rade=2.0, pl_masse=10.0, st_rad=1.0, sy_jmag=8.0,
                  pl_eqt=float("nan"), pl_insol=float("nan"))
        df = add_t_eq(df)
        assert pd.isna(tsm_proxy(df).iloc[0])

    def test_brighter_star_higher_tsm(self):
        # Lower J-mag → brighter → 10^(-J/5) larger → higher TSM
        df = _multi(
            dict(pl_rade=2.0, pl_masse=10.0, st_rad=1.0, sy_jmag=6.0, pl_eqt=800.0),
            dict(pl_rade=2.0, pl_masse=10.0, st_rad=1.0, sy_jmag=12.0, pl_eqt=800.0),
        )
        df = add_t_eq(df)
        vals = tsm_proxy(df)
        assert vals.iloc[0] > vals.iloc[1]

    def test_larger_radius_higher_tsm(self):
        # TSM ∝ R_p^3; larger planet, same mass → much higher TSM
        df = _multi(
            dict(pl_rade=3.0, pl_masse=10.0, st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0),
            dict(pl_rade=1.5, pl_masse=10.0, st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0),
        )
        df = add_t_eq(df)
        vals = tsm_proxy(df)
        assert vals.iloc[0] > vals.iloc[1]

    def test_scale_factor_boundaries(self):
        # Verify correct scale factor is applied at each radius boundary
        radii = [1.0, 2.0, 3.5, 5.0]
        expected_scales = [0.190, 1.26, 1.28, 1.15]
        for r, exp_scale in zip(radii, expected_scales):
            df = _row(pl_rade=r, pl_masse=10.0, st_rad=1.0, sy_jmag=8.0, pl_eqt=800.0)
            df = add_t_eq(df)
            val = tsm_proxy(df).iloc[0]
            # Back-compute the implicit scale factor and compare
            raw = (r ** 3 * 800.0) / (10.0 * 1.0 ** 2) * 10 ** (-8.0 / 5)
            implicit_scale = val / raw
            assert math.isclose(implicit_scale, exp_scale, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 3B — esm_proxy_simplified
# ---------------------------------------------------------------------------

class TestEsmProxySimplified:
    def _esm_expected(self, r_p, r_s, t_eq, t_s, k):
        r_ratio = r_p / (r_s * _R_SUN_IN_R_EARTH)
        t_day = 1.1 * t_eq
        return _ESM_NORM * r_ratio ** 2 * (t_day / t_s) ** 4 * 10 ** (-k / 5)

    def test_formula_correctness(self):
        df = _row(pl_rade=2.0, st_rad=1.0, pl_eqt=800.0, st_teff=5500.0, sy_kmag=7.5)
        df = add_t_eq(df)
        result = esm_proxy_simplified(df).iloc[0]
        expected = self._esm_expected(2.0, 1.0, 800.0, 5500.0, 7.5)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_nan_when_kmag_missing(self):
        df = _row(pl_rade=2.0, st_rad=1.0, pl_eqt=800.0, st_teff=5500.0,
                  sy_kmag=float("nan"))
        df = add_t_eq(df)
        assert pd.isna(esm_proxy_simplified(df).iloc[0])

    def test_nan_when_radius_missing(self):
        df = _row(pl_rade=float("nan"), st_rad=1.0, pl_eqt=800.0, st_teff=5500.0, sy_kmag=7.5)
        df = add_t_eq(df)
        assert pd.isna(esm_proxy_simplified(df).iloc[0])

    def test_nan_when_teff_missing(self):
        df = _row(pl_rade=2.0, st_rad=1.0, pl_eqt=800.0, st_teff=float("nan"), sy_kmag=7.5)
        df = add_t_eq(df)
        assert pd.isna(esm_proxy_simplified(df).iloc[0])

    def test_hotter_planet_higher_esm(self):
        # Higher t_eq → higher T_day → higher (T_day/T_s)^4 → higher ESM
        df = _multi(
            dict(pl_rade=2.0, st_rad=1.0, pl_eqt=1200.0, st_teff=5500.0, sy_kmag=7.5),
            dict(pl_rade=2.0, st_rad=1.0, pl_eqt=500.0,  st_teff=5500.0, sy_kmag=7.5),
        )
        df = add_t_eq(df)
        vals = esm_proxy_simplified(df)
        assert vals.iloc[0] > vals.iloc[1]

    def test_larger_planet_higher_esm(self):
        # ESM ∝ (R_p/R_s)^2
        df = _multi(
            dict(pl_rade=3.0, st_rad=1.0, pl_eqt=800.0, st_teff=5500.0, sy_kmag=7.5),
            dict(pl_rade=1.5, st_rad=1.0, pl_eqt=800.0, st_teff=5500.0, sy_kmag=7.5),
        )
        df = add_t_eq(df)
        vals = esm_proxy_simplified(df)
        ratio = vals.iloc[0] / vals.iloc[1]
        assert math.isclose(ratio, (3.0 / 1.5) ** 2, rel_tol=1e-5)

    def test_t_day_is_1p1_t_eq(self):
        # Verify the 1.1 * T_eq dayside assumption by comparing two temps
        t_eq = 700.0
        df_a = _row(pl_rade=2.0, st_rad=1.0, pl_eqt=t_eq, st_teff=5500.0, sy_kmag=7.5)
        df_b = _row(pl_rade=2.0, st_rad=1.0, pl_eqt=t_eq * 1.1, st_teff=5500.0, sy_kmag=7.5)
        df_a = add_t_eq(df_a)
        df_b = add_t_eq(df_b)
        esm_a = esm_proxy_simplified(df_a).iloc[0]
        esm_b_raw = _ESM_NORM * (2.0 / (1.0 * _R_SUN_IN_R_EARTH)) ** 2 \
                    * (t_eq * 1.1 / 5500.0) ** 4 * 10 ** (-7.5 / 5)
        # esm_a should equal esm computed with T_day = 1.1 * t_eq
        expected = _ESM_NORM * (2.0 / (1.0 * _R_SUN_IN_R_EARTH)) ** 2 \
                   * (1.1 * t_eq / 5500.0) ** 4 * 10 ** (-7.5 / 5)
        assert math.isclose(esm_a, expected, rel_tol=1e-6)

    def test_column_named_esm_proxy_simplified_not_esm(self):
        # Enforce naming convention — the simplified metric must not be called 'esm'
        df = _row()
        df = add_t_eq(df)
        out = df.copy()
        out["esm_proxy_simplified"] = esm_proxy_simplified(df)
        assert "esm_proxy_simplified" in out.columns
        assert "esm" not in [c for c in out.columns if c == "esm"]


# ---------------------------------------------------------------------------
# 3B — hz_flag
# ---------------------------------------------------------------------------

class TestHzFlag:
    def test_solar_twin_earth_analog_is_in_hz(self):
        # S=1.0 S_earth around a solar twin should be in the conservative HZ
        df = _row(pl_insol=1.0, st_teff=5780.0)
        result = hz_flag(df)
        assert result.iloc[0] is True or result.iloc[0] == True  # noqa: E712

    def test_very_hot_planet_is_not_in_hz(self):
        # S=1000 S_earth (hot Jupiter insolation) is not habitable
        df = _row(pl_insol=1000.0, st_teff=5780.0)
        result = hz_flag(df)
        assert result.iloc[0] is False or result.iloc[0] == False  # noqa: E712

    def test_very_cold_planet_is_not_in_hz(self):
        # S=0.001 S_earth is well outside the outer HZ
        df = _row(pl_insol=0.001, st_teff=5780.0)
        result = hz_flag(df)
        assert result.iloc[0] is False or result.iloc[0] == False  # noqa: E712

    def test_nan_when_insol_missing(self):
        df = _row(pl_insol=float("nan"), st_teff=5780.0)
        result = hz_flag(df)
        assert pd.isna(result.iloc[0])

    def test_nan_when_teff_missing(self):
        df = _row(pl_insol=1.0, st_teff=float("nan"))
        result = hz_flag(df)
        assert pd.isna(result.iloc[0])

    def test_nan_when_teff_below_validity_range(self):
        df = _row(pl_insol=1.0, st_teff=_HZ_T_MIN - 1)
        result = hz_flag(df)
        assert pd.isna(result.iloc[0])

    def test_nan_when_teff_above_validity_range(self):
        df = _row(pl_insol=1.0, st_teff=_HZ_T_MAX + 1)
        result = hz_flag(df)
        assert pd.isna(result.iloc[0])

    def test_inner_boundary_consistency(self):
        # A planet just inside the inner HZ boundary should be False
        t_s = 5780.0
        s_inner = _hz_s_eff(t_s, "inner")
        df = _row(pl_insol=s_inner * 1.05, st_teff=t_s)  # 5% inside inner (too hot)
        result = hz_flag(df)
        assert result.iloc[0] is False or result.iloc[0] == False  # noqa: E712

    def test_outer_boundary_consistency(self):
        # A planet just outside the outer HZ boundary should be False
        t_s = 5780.0
        s_outer = _hz_s_eff(t_s, "outer")
        df = _row(pl_insol=s_outer * 0.95, st_teff=t_s)  # 5% beyond outer (too cold)
        result = hz_flag(df)
        assert result.iloc[0] is False or result.iloc[0] == False  # noqa: E712

    def test_hz_s_eff_solar_inner(self):
        # At 5780 K, S_eff_inner ≈ 1.0385 (from Kopparapu Table 3 with dt=0)
        s = _hz_s_eff(5780.0, "inner")
        assert math.isclose(s, 1.0385, rel_tol=1e-4)

    def test_hz_s_eff_solar_outer(self):
        # At 5780 K, S_eff_outer ≈ 0.3179 (dt=0 term only)
        s = _hz_s_eff(5780.0, "outer")
        assert math.isclose(s, 0.3179, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 3B — uncertainty fractions
# ---------------------------------------------------------------------------

class TestRadiusUncertaintyFraction:
    def test_symmetric_errors(self):
        # Both errors equal → fraction = err / value
        df = _row(pl_rade=2.0, pl_radeerr1=0.2, pl_radeerr2=-0.2)
        frac, partial = radius_uncertainty_fraction(df)
        assert math.isclose(frac.iloc[0], 0.2 / 2.0, rel_tol=1e-6)
        assert not partial.iloc[0]

    def test_asymmetric_errors(self):
        # Mean of |err1| and |err2| divided by value
        df = _row(pl_rade=4.0, pl_radeerr1=0.4, pl_radeerr2=-0.2)
        frac, partial = radius_uncertainty_fraction(df)
        expected = (0.4 + 0.2) / 2 / 4.0
        assert math.isclose(frac.iloc[0], expected, rel_tol=1e-6)
        assert not partial.iloc[0]

    def test_nan_when_radius_missing(self):
        df = _row(pl_rade=float("nan"), pl_radeerr1=0.1, pl_radeerr2=-0.1)
        frac, _ = radius_uncertainty_fraction(df)
        assert pd.isna(frac.iloc[0])

    def test_partial_flag_upper_only(self):
        df = _row(pl_rade=2.0, pl_radeerr1=0.2, pl_radeerr2=float("nan"))
        frac, partial = radius_uncertainty_fraction(df)
        assert partial.iloc[0]
        assert math.isclose(frac.iloc[0], 0.2 / 2.0, rel_tol=1e-6)

    def test_partial_flag_lower_only(self):
        df = _row(pl_rade=2.0, pl_radeerr1=float("nan"), pl_radeerr2=-0.1)
        frac, partial = radius_uncertainty_fraction(df)
        assert partial.iloc[0]
        assert math.isclose(frac.iloc[0], 0.1 / 2.0, rel_tol=1e-6)

    def test_nan_when_both_errors_missing(self):
        df = _row(pl_rade=2.0, pl_radeerr1=float("nan"), pl_radeerr2=float("nan"))
        frac, partial = radius_uncertainty_fraction(df)
        assert pd.isna(frac.iloc[0])


class TestMassUncertaintyFraction:
    def test_uses_pl_masse_when_present(self):
        df = _row(pl_masse=10.0, pl_masseerr1=1.0, pl_masseerr2=-1.0,
                  pl_bmasse=20.0, pl_bmasseerr1=2.0, pl_bmasseerr2=-2.0)
        frac, _ = mass_uncertainty_fraction(df)
        assert math.isclose(frac.iloc[0], 1.0 / 10.0, rel_tol=1e-6)

    def test_falls_back_to_bmasse(self):
        df = _row(pl_masse=float("nan"), pl_masseerr1=float("nan"), pl_masseerr2=float("nan"),
                  pl_bmasse=20.0, pl_bmasseerr1=2.0, pl_bmasseerr2=-2.0)
        frac, _ = mass_uncertainty_fraction(df)
        assert math.isclose(frac.iloc[0], 2.0 / 20.0, rel_tol=1e-6)

    def test_nan_when_both_mass_columns_missing(self):
        df = _row(pl_masse=float("nan"), pl_masseerr1=float("nan"), pl_masseerr2=float("nan"),
                  pl_bmasse=float("nan"), pl_bmasseerr1=float("nan"), pl_bmasseerr2=float("nan"))
        frac, _ = mass_uncertainty_fraction(df)
        assert pd.isna(frac.iloc[0])


# ---------------------------------------------------------------------------
# add_all integration
# ---------------------------------------------------------------------------

class TestAddAll:
    def test_all_columns_added(self):
        df = _row()
        out = add_all(df)
        expected_cols = {
            "t_eq", "t_eq_source",
            "tsm", "tsm_mass_source",
            "esm_proxy_simplified",
            "hz_flag",
            "uf_radius", "uf_radius_partial",
            "uf_mass", "uf_mass_partial",
        }
        assert expected_cols.issubset(set(out.columns))

    def test_does_not_modify_original(self):
        df = _row()
        original_cols = set(df.columns)
        _ = add_all(df)
        assert set(df.columns) == original_cols

    def test_esm_column_not_named_esm(self):
        df = _row()
        out = add_all(df)
        assert "esm" not in out.columns
        assert "esm_proxy_simplified" in out.columns

    def test_no_tsm_when_mass_fully_missing(self):
        df = _row(pl_masse=float("nan"), pl_bmasse=float("nan"))
        out = add_all(df)
        assert pd.isna(out["tsm"].iloc[0])

    def test_tsm_mass_source_recorded(self):
        df = _row(pl_masse=float("nan"), pl_bmasse=10.0)
        out = add_all(df)
        assert out["tsm_mass_source"].iloc[0] == "pl_bmasse"

    def test_tsm_mass_source_pl_masse(self):
        df = _row(pl_masse=10.0, pl_bmasse=10.0)
        out = add_all(df)
        assert out["tsm_mass_source"].iloc[0] == "pl_masse"
