# Feature Engineering

All features in this document are derived in `src/exoprior/features/engineer.py`.
None of these derived values constitute scientific conclusions.
They are engineering proxies used as inputs to the scoring model.

---

## 3A — Equilibrium Temperature (`t_eq`)

### Purpose
Many planets in the archive lack `pl_eqt`. This step fills the gap so that
downstream metrics (TSM, ESM, HZ flag) have a temperature input wherever possible.

### Columns added
| Column | Type | Description |
|---|---|---|
| `t_eq` | float [K] | Best available equilibrium temperature |
| `t_eq_source` | str | `'archive'`, `'derived_from_insol'`, or `'missing'` |

### Precedence
1. Use archive `pl_eqt` where present.
2. Derive from `pl_insol` using the formula below where `pl_eqt` is null.
3. Leave as NaN if both are absent.

### Derivation formula
```
T_eq = 278.5 × S^0.25   [K]
```
where `S` is insolation in Earth flux units (`pl_insol`), and 278.5 K is
Earth's effective temperature assuming zero Bond albedo (A = 0).

Source: standard stellar physics; e.g. Seager (2010), *Exoplanet Atmospheres*,
eq. 3.9. Kopparapu et al. (2013) use the same base formula.

### Limitations
- **Albedo = 0 assumption.** This overestimates temperature for reflective planets.
  The archive's own `pl_eqt` values use varying albedo assumptions per reference;
  this derivation consistently assumes A = 0.
- Only used as a fallback. If `pl_eqt` exists, it is always preferred.
- The `t_eq_source` column tracks which rows used the derivation so the effect
  can be assessed at the scoring or sensitivity stage.

---

## 3B — Transmission Spectroscopy Metric (`tsm`)

### Purpose
The TSM ranks planets by their expected signal-to-noise ratio in transmission
spectroscopy observations, as a proxy for atmospheric characterisation value.

### Formula
Kempton et al. (2018), PASP 130, 993 — Table 1:

```
TSM = scale_factor × (R_p³ × T_eq) / (M_p × R_s²) × 10^(−J/5)
```

| Symbol | Source column | Unit |
|---|---|---|
| `R_p` | `pl_rade` | R⊕ |
| `T_eq` | `t_eq` (3A output) | K |
| `M_p` | `pl_masse`, fallback `pl_bmasse` | M⊕ |
| `R_s` | `st_rad` | R☉ |
| `J` | `sy_jmag` | mag |

**Scale factors** (Kempton Table 1, radius bins):

| R_p range [R⊕] | scale_factor |
|---|---|
| < 1.5 | 0.190 |
| 1.5 – 2.75 | 1.26 |
| 2.75 – 4.0 | 1.28 |
| ≥ 4.0 | 1.15 |

### Mass fallback
`tsm_mass_source` records whether `pl_masse` or `pl_bmasse` was used.
`pl_bmasse` is mass or M·sin(i) and introduces inclination uncertainty into TSM
for non-edge-on systems; this is noted but not corrected in the MVP.

### Null contract
TSM is NaN if any of R_p, T_eq, M_p, R_s, J is NaN.

### Limitations
- Does not account for atmospheric scale height uncertainty, cloud coverage,
  stellar activity, or telescope-specific noise properties.
- Applies to all planets uniformly regardless of whether transmission spectra
  have already been obtained.
- M·sin(i) fallback underestimates true mass for non-edge-on orbits.

---

## 3B — Emission Spectroscopy Metric — Simplified (`esm_proxy_simplified`)

### ⚠ This is NOT the Kempton (2018) ESM

The Kempton (2018) ESM uses the ratio of the Planck function B(T, 7.5 µm)
evaluated at stellar and planetary temperatures, normalised by K-band magnitude.
That ratio is wavelength-dependent and differs significantly from a simple
Stefan-Boltzmann temperature ratio, especially for cool (M-dwarf) host stars
where 7.5 µm approaches the emission peak.

**This implementation substitutes (T_day/T_s)⁴ for the Planck ratio.**
It is named `esm_proxy_simplified` throughout the codebase and documentation
to make this approximation explicit and machine-searchable. Do not interpret
results as equivalent to the Kempton ESM.

### Formula used
```
esm_proxy_simplified = 4.29×10⁶ × (R_p / (R_s × 109.076))² × (T_day / T_s)⁴ × 10^(−K/5)
```

where:
- `T_day = 1.1 × T_eq`  (Kempton 2018 dayside temperature assumption)
- `109.076` converts R_s from R☉ to R⊕ to make the ratio dimensionless (IAU 2015)
- `4.29×10⁶` is the Kempton normalisation constant

| Symbol | Source column | Unit |
|---|---|---|
| `R_p` | `pl_rade` | R⊕ |
| `R_s` | `st_rad` | R☉ |
| `T_eq` | `t_eq` | K |
| `T_s` | `st_teff` | K |
| `K` | `sy_kmag` | mag |

### Null contract
NaN if any of R_p, R_s, T_eq, T_s, K is NaN.

### Limitations
- **Planck ratio approximation is poor for cool stars (T_s < 3500 K).** For M-dwarf
  hosts, do not use `esm_proxy_simplified` to compare against actual emission
  spectroscopy estimates from the literature.
- The true Kempton ESM should be computed in a future ticket once K-band Planck
  function evaluation is implemented.

---

## 3B — Conservative Habitable Zone Flag (`hz_flag`)

### Purpose
Flags planets that receive insolation within the conservative habitable zone
boundaries for their host star type. Used as one binary component in scoring.

### Boundaries
Conservative HZ from Kopparapu et al. (2013), ApJ 765, 131.
- **Inner edge**: Runaway Greenhouse
- **Outer edge**: Maximum Greenhouse

Parameterised polynomial (Table 3):
```
S_eff = S_sun + a×ΔT + b×ΔT² + c×ΔT³ + d×ΔT⁴
where ΔT = T_star − 5780 [K]
```

Coefficients:
| Limit | S_sun | a | b | c | d |
|---|---|---|---|---|---|
| Inner (RG) | 1.0385 | 1.2456×10⁻⁴ | 1.4612×10⁻⁸ | −7.6345×10⁻¹² | −1.7511×10⁻¹⁵ |
| Outer (MG) | 0.3179 | 5.4513×10⁻⁵ | 1.5313×10⁻⁹ | −2.7786×10⁻¹² | −8.2246×10⁻¹⁶ |

### Comparison value
`pl_insol` (insolation flux in Earth units) is compared against S_eff_inner
and S_eff_outer evaluated at `st_teff`.

### Null contract
Returns NaN when `pl_insol` or `st_teff` is missing, or when `st_teff` is
outside the polynomial's validity range of 2600–7200 K.

### Limitations
1. These are 1D energy-balance boundaries assuming Earth-like atmospheric
   composition. Planets with different atmospheres (e.g. CO₂-dominated) have
   different HZ boundaries.
2. A `True` flag means the planet receives the right *amount* of light.
   It says nothing about atmospheric retention, surface pressure, or water.
3. Validity range 2600–7200 K excludes some extreme stellar types; those rows
   receive `hz_flag = NaN`.
4. The Kopparapu (2013) HZ applies to single-star systems. Circumbinary planets
   (`cb_flag = 1`) may have complex insolation histories not captured here.

---

## 3B — Uncertainty Fractions (`uf_radius`, `uf_mass`)

### Purpose
Quantify measurement precision per planet. Used in the scoring model's
uncertainty-priority component (prefer well-constrained targets).

### Formula
```
fractional_uncertainty = (|err_upper| + |err_lower|) / (2 × |value|)
```

This is the mean 1-sigma uncertainty normalised by the central value.

### Columns

| Column | Derived from | Description |
|---|---|---|
| `uf_radius` | `pl_rade`, `pl_radeerr1`, `pl_radeerr2` | Fractional radius uncertainty |
| `uf_radius_partial` | — | True if only one error column was available |
| `uf_mass` | `pl_masse` / `pl_bmasse` + errors | Fractional mass uncertainty |
| `uf_mass_partial` | — | True if only one error column was available |

### Null contract
- NaN when the central value is missing or zero.
- NaN when both error columns are missing.
- Partial (one-sided) estimates are flagged; they underestimate true uncertainty.

### Mass source
`uf_mass` uses `pl_masse` and its errors by preference; falls back to `pl_bmasse`
with its errors when `pl_masse` is null. The same fallback logic as TSM.
