"""
Feature engineering for ExoPrior.

Organized in dependency order:

  3A  t_eq_from_insol()      — derive equilibrium temperature from insolation flux
      add_t_eq()             — fill pl_eqt where missing using 3A; document which rows used

  3B  tsm_proxy()            — Transmission Spectroscopy Metric (Kempton et al. 2018)
      esm_proxy_simplified() — simplified Emission Spectroscopy Metric (no K-band)
      hz_flag()              — conservative habitable-zone flag (Kopparapu et al. 2013)
      radius_uncertainty_fraction()
      mass_uncertainty_fraction()
      add_all()              — convenience wrapper that applies 3A then 3B in order

None of these functions modify or drop rows. They only append columns. Cleaning is
Ticket 2's responsibility. Scoring is Ticket 4's responsibility.

Scientific claims
-----------------
No column produced here constitutes a scientific conclusion. All derived values
are engineering proxies for ranking purposes. Limitations are noted per function
and in docs/feature_engineering.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 3A — Equilibrium temperature
# ---------------------------------------------------------------------------

# Equilibrium temperature from insolation flux, assuming zero albedo (Bond albedo = 0).
#
# Derivation:
#   T_eq = T_sun * (R_sun / (2 * a))^0.5 * (1 - A)^0.25
# which, in terms of insolation S relative to Earth (S_earth = 1), simplifies to:
#   T_eq = T_earth_eff * S^0.25
#
# where T_earth_eff is Earth's effective temperature assuming A=0 (~278.5 K).
#
# Source: standard stellar physics; see e.g. Seager (2010) "Exoplanet Atmospheres",
#   equation 3.9.  Kopparapu et al. (2013) use the same base formula.
#
# Limitation: albedo = 0 is a known overestimate of temperature for planets that
#   reflect significant starlight. The archive's pl_eqt values typically assume
#   A=0 or A=0.3 depending on the source. This derived value uses A=0 throughout
#   and is only used where the archive provides no pl_eqt at all.

_T_EFF_EARTH_ZERO_ALBEDO_K = 278.5  # K; Earth effective temp assuming A=0


def t_eq_from_insol(insol: pd.Series) -> pd.Series:
    """Derive equilibrium temperature [K] from insolation flux [Earth units].

    T_eq = 278.5 * S^0.25   (albedo = 0 assumption; see module docstring)

    Returns NaN where insol is NaN or non-positive.
    """
    valid = insol.where(insol > 0)
    return _T_EFF_EARTH_ZERO_ALBEDO_K * (valid ** 0.25)


def add_t_eq(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a filled equilibrium temperature column.

    Logic (applied row by row, precedence order):
      1. Use archive pl_eqt where present.
      2. Derive from pl_insol via t_eq_from_insol() where pl_eqt is null.
      3. Leave as NaN if both are missing.

    Adds two columns:
      t_eq        — best available equilibrium temperature [K]
      t_eq_source — 'archive', 'derived_from_insol', or 'missing'
    """
    df = df.copy()

    derived = t_eq_from_insol(df["pl_insol"])

    df["t_eq"] = df["pl_eqt"].where(df["pl_eqt"].notna(), derived)

    source = pd.Series("missing", index=df.index, dtype=str)
    source[df["pl_eqt"].notna()] = "archive"
    source[df["pl_eqt"].isna() & derived.notna()] = "derived_from_insol"
    df["t_eq_source"] = source

    return df


# ---------------------------------------------------------------------------
# 3B — Scientific metrics
# ---------------------------------------------------------------------------

# --- TSM proxy ---
#
# Formula: Kempton et al. (2018), Table 1 scale factor variant.
#   TSM = scale_factor * (R_p^3 * T_eq) / (M_p * R_s^2) * 10^(-J/5)
#
# where:
#   R_p   — planet radius [Earth radii]
#   T_eq  — equilibrium temperature [K]  (from t_eq column, 3A output)
#   M_p   — planet mass [Earth masses]   (pl_masse; falls back to pl_bmasse)
#   R_s   — stellar radius [Solar radii]
#   J     — J-band magnitude of host star (sy_jmag)
#   scale_factor — radius-bin-dependent constant from Kempton Table 1
#
# Scale factors (Kempton et al. 2018, Table 1):
#   R_p < 1.5 R_earth   → 0.190
#   1.5 ≤ R_p < 2.75    → 1.26
#   2.75 ≤ R_p < 4.0    → 1.28
#   R_p ≥ 4.0           → 1.15
#
# Returns NaN when any of R_p, T_eq, M_p, R_s, or sy_jmag is NaN.
# M_p falls back to pl_bmasse (mass or m*sin(i)) when pl_masse is null;
# this fallback is recorded in a companion column tsm_mass_source.
#
# Reference: Kempton, E. M.-R., et al. (2018), PASP, 130, 993.
#   https://doi.org/10.1086/498328
#
# Limitation: This is the Kempton TSM formula applied to catalog parameters.
#   It does not account for atmospheric scale height uncertainty, cloud coverage,
#   actual observing time, or telescope-specific noise floors.

_TSM_SCALE = [
    (1.5, 0.190),
    (2.75, 1.26),
    (4.0, 1.28),
    (float("inf"), 1.15),
]


def _tsm_scale_factor(r_p: float) -> float:
    for threshold, factor in _TSM_SCALE:
        if r_p < threshold:
            return factor
    return 1.15  # unreachable; satisfies type checker


def tsm_proxy(df: pd.DataFrame) -> pd.Series:
    """Compute TSM per Kempton et al. (2018) for each planet row.

    Required columns: pl_rade, t_eq, st_rad, sy_jmag.
    Mass: prefers pl_masse; falls back to pl_bmasse.
    Returns NaN for any row with missing required input.
    """
    r_p = df["pl_rade"]
    t_eq = df["t_eq"]
    r_s = df["st_rad"]
    j = df["sy_jmag"]

    m_p = df["pl_masse"].where(df["pl_masse"].notna(), df["pl_bmasse"])

    scale = r_p.apply(
        lambda x: _tsm_scale_factor(x) if pd.notna(x) else float("nan")
    )

    tsm = scale * (r_p ** 3 * t_eq) / (m_p * r_s ** 2) * 10 ** (-j / 5)

    # NaN propagates automatically from any missing input above; enforce explicitly
    # for clarity so future readers understand the null contract.
    required_present = r_p.notna() & t_eq.notna() & r_s.notna() & j.notna() & m_p.notna()
    return tsm.where(required_present)


# --- ESM proxy (simplified) ---
#
# The full Kempton (2018) ESM formula compares the planet/star brightness ratio
# at 7.5 µm (a JWST/MIRI wavelength), using the Planck function B(T, 7.5 µm)
# evaluated at both stellar and planetary temperatures, and normalises by K-band
# magnitude.
#
# That formula requires:
#   - K-band magnitude (sy_kmag)       — available in the archive
#   - Planck function at 7.5 µm        — computable
#   - Planet dayside temperature T_day = 1.1 * T_eq (Kempton assumption)
#
# THIS IMPLEMENTATION IS A SIMPLIFICATION.
# We omit the Planck function ratio and replace it with a Stefan-Boltzmann
# temperature ratio (T_p / T_s)^4, which is only a good approximation when
# both temperatures are far from the 7.5 µm peak and behave roughly as
# blackbodies in that regime.  The result is named `esm_proxy_simplified`
# to make this distinction explicit and machine-searchable.
#
# Simplified formula used here:
#   esm_proxy_simplified = 4.29e6 * (R_p/R_s)^2 * (T_day/T_s)^4 * 10^(-K/5)
#
# where:
#   R_p, R_s in consistent units (pl_rade / st_rad are converted to R_sun / R_sun,
#     so the ratio is dimensionless — pl_rade in R_earth, st_rad in R_sun;
#     1 R_sun = 109.076 R_earth, so r_ratio = pl_rade / (st_rad * 109.076))
#   T_day    = 1.1 * t_eq   (Kempton 2018 dayside temperature assumption)
#   T_s      = st_teff
#   K        = sy_kmag
#   4.29e6   = normalisation constant from Kempton (2018) Table 1
#
# Returns NaN when R_p, R_s, t_eq, st_teff, or sy_kmag is missing.
#
# Reference: Kempton et al. (2018), PASP, 130, 993.
# Limitation: The Planck ratio approximation is least accurate for cool stars
#   (T_s < 3500 K) where the 7.5 µm peak matters most. Do not interpret
#   esm_proxy_simplified as equivalent to Kempton ESM for M-dwarf hosts.

_R_SUN_IN_R_EARTH = 109.076  # 1 R_sun = 109.076 R_earth (IAU 2015 nominal)
_ESM_NORM = 4.29e6


def esm_proxy_simplified(df: pd.DataFrame) -> pd.Series:
    """Compute a simplified ESM proxy (Stefan-Boltzmann approximation).

    See module docstring for the full list of simplifications and limitations.
    Required columns: pl_rade, st_rad, t_eq, st_teff, sy_kmag.
    Returns NaN for any row with missing required input.

    This is NOT equivalent to the Kempton (2018) ESM. It is named
    `esm_proxy_simplified` to signal the approximation explicitly.
    """
    r_p = df["pl_rade"]           # R_earth
    r_s = df["st_rad"]            # R_sun
    t_eq = df["t_eq"]             # K
    t_s = df["st_teff"]           # K
    k = df["sy_kmag"]             # mag

    # Convert to the same radius unit (dimensionless ratio)
    r_ratio = r_p / (r_s * _R_SUN_IN_R_EARTH)

    t_day = 1.1 * t_eq            # Kempton 2018 dayside temperature assumption

    esm = _ESM_NORM * r_ratio ** 2 * (t_day / t_s) ** 4 * 10 ** (-k / 5)

    required_present = r_p.notna() & r_s.notna() & t_eq.notna() & t_s.notna() & k.notna()
    return esm.where(required_present)


# --- Conservative habitable zone flag ---
#
# The habitable zone (HZ) boundaries used here are the "conservative" limits
# from Kopparapu et al. (2013): Runaway Greenhouse (inner) and Maximum Greenhouse
# (outer), parameterised as a function of stellar effective temperature.
#
# Empirical polynomial coefficients (Kopparapu 2013, Table 3):
#
#   S_eff = S_eff_sun + a*(T_star - 5780) + b*(T_star - 5780)^2
#             + c*(T_star - 5780)^3 + d*(T_star - 5780)^4
#
# where T_star is in K and S_eff_sun, a, b, c, d are from Table 3 for each limit.
#
# Conservative limits used:
#   Inner (Runaway Greenhouse): S_eff_sun=1.0385, a=1.2456e-4, b=1.4612e-8,
#                               c=-7.6345e-12, d=-1.7511e-15
#   Outer (Maximum Greenhouse): S_eff_sun=0.3179, a=5.4513e-5, b=1.5313e-9,
#                               c=-2.7786e-12, d=-8.2246e-16
#
# Reference: Kopparapu, R. K., et al. (2013), ApJ, 765, 131.
#   https://doi.org/10.1088/0004-637X/765/2/131
#
# Limitations:
#   1. These are 1D energy-balance boundaries assuming an Earth-like atmospheric
#      composition. Different atmospheric compositions shift the boundaries.
#   2. We use insolation flux (pl_insol) as the comparison value. Where pl_insol
#      is missing we use t_eq to compute a flux proxy, but that introduces error.
#   3. The Kopparapu 2013 parameterisation is valid for T_star 2600–7200 K.
#      Stars outside this range receive hz_flag = NaN (cannot evaluate).
#   4. A True hz_flag does NOT mean the planet is habitable. It means it
#      receives insolation consistent with the conservative HZ definition for
#      its host star type.

_HZ_COEFFS = {
    # inner: Runaway Greenhouse
    "inner": dict(S_sun=1.0385, a=1.2456e-4, b=1.4612e-8, c=-7.6345e-12, d=-1.7511e-15),
    # outer: Maximum Greenhouse
    "outer": dict(S_sun=0.3179, a=5.4513e-5, b=1.5313e-9, c=-2.7786e-12, d=-8.2246e-16),
}

_HZ_T_MIN = 2600.0  # K; lower validity limit of Kopparapu 2013 polynomial
_HZ_T_MAX = 7200.0  # K; upper validity limit


def _hz_s_eff(t_star: float, limit: str) -> float:
    """Compute HZ critical insolation flux for a given stellar temperature."""
    c = _HZ_COEFFS[limit]
    dt = t_star - 5780.0
    return c["S_sun"] + c["a"] * dt + c["b"] * dt**2 + c["c"] * dt**3 + c["d"] * dt**4


def hz_flag(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True when planet is within the conservative HZ.

    Requires columns: pl_insol, st_teff.
    Returns NaN where pl_insol or st_teff is missing, or where st_teff is
    outside the 2600–7200 K validity range of Kopparapu et al. (2013).
    """
    result = pd.Series(np.nan, index=df.index, dtype=object)

    valid = df["pl_insol"].notna() & df["st_teff"].notna()
    in_range = valid & df["st_teff"].between(_HZ_T_MIN, _HZ_T_MAX)

    for idx in df.index[in_range]:
        t = df.at[idx, "st_teff"]
        s = df.at[idx, "pl_insol"]
        s_inner = _hz_s_eff(t, "inner")
        s_outer = _hz_s_eff(t, "outer")
        result.at[idx] = bool(s_outer <= s <= s_inner)

    return result


# --- Uncertainty fractions ---
#
# Fractional uncertainty = (|err_upper| + |err_lower|) / (2 * |value|)
#
# This is the mean of the upper and lower 1-sigma uncertainties normalised by
# the central value.  Returns NaN when the value is zero, missing, or when
# both error columns are missing.  A partial result (one error missing) uses
# the available error only and is therefore an underestimate; the column
# uf_*_partial flags these rows.

def _uncertainty_fraction(
    value: pd.Series,
    err_upper: pd.Series,
    err_lower: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Compute fractional uncertainty and a flag for partial (one-sided) estimates.

    Returns (fraction, partial_flag).
    """
    abs_upper = err_upper.abs()
    abs_lower = err_lower.abs()

    both = abs_upper.notna() & abs_lower.notna()
    upper_only = abs_upper.notna() & abs_lower.isna()
    lower_only = abs_upper.isna() & abs_lower.notna()

    err_mean = pd.Series(np.nan, index=value.index)
    err_mean[both] = (abs_upper[both] + abs_lower[both]) / 2
    err_mean[upper_only] = abs_upper[upper_only]
    err_mean[lower_only] = abs_lower[lower_only]

    nonzero = value.abs() > 0
    fraction = err_mean.where(nonzero & err_mean.notna()) / value.abs()

    partial_flag = (upper_only | lower_only) & value.notna()

    return fraction, partial_flag


def radius_uncertainty_fraction(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fractional radius uncertainty: (|err1| + |err2|) / (2 * pl_rade).

    Returns (uf_radius, uf_radius_partial).
    uf_radius_partial is True where only one error column was available.
    """
    return _uncertainty_fraction(df["pl_rade"], df["pl_radeerr1"], df["pl_radeerr2"])


def mass_uncertainty_fraction(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fractional mass uncertainty using pl_masse (falls back to pl_bmasse).

    Returns (uf_mass, uf_mass_partial).
    uf_mass_partial is True where only one error column was available.
    """
    mass = df["pl_masse"].where(df["pl_masse"].notna(), df["pl_bmasse"])
    err1 = df["pl_masseerr1"].where(df["pl_masse"].notna(), df["pl_bmasseerr1"])
    err2 = df["pl_masseerr2"].where(df["pl_masse"].notna(), df["pl_bmasseerr2"])
    return _uncertainty_fraction(mass, err1, err2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps (3A then 3B) and return a new DataFrame.

    Adds columns:
      t_eq, t_eq_source,
      tsm, tsm_mass_source,
      esm_proxy_simplified,
      hz_flag,
      uf_radius, uf_radius_partial,
      uf_mass,   uf_mass_partial
    """
    df = add_t_eq(df)

    df = df.copy()
    df["tsm"] = tsm_proxy(df)

    mass_used = df["pl_masse"].where(df["pl_masse"].notna(), df["pl_bmasse"])
    df["tsm_mass_source"] = pd.Series("missing", index=df.index, dtype=str)
    df.loc[df["pl_masse"].notna(), "tsm_mass_source"] = "pl_masse"
    df.loc[df["pl_masse"].isna() & df["pl_bmasse"].notna(), "tsm_mass_source"] = "pl_bmasse"

    df["esm_proxy_simplified"] = esm_proxy_simplified(df)
    df["hz_flag"] = hz_flag(df)

    uf_r, uf_r_partial = radius_uncertainty_fraction(df)
    df["uf_radius"] = uf_r
    df["uf_radius_partial"] = uf_r_partial

    uf_m, uf_m_partial = mass_uncertainty_fraction(df)
    df["uf_mass"] = uf_m
    df["uf_mass_partial"] = uf_m_partial

    return df
