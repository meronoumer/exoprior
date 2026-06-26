# Data Dictionary

Columns used by ExoPrior, verified against `TAP_SCHEMA.columns` on 2026-06-26.
Source table: `ps` (NASA Exoplanet Archive Planetary Systems).

All column names, descriptions, units, and datatypes were confirmed by querying:
```
SELECT column_name, description, unit, datatype
FROM TAP_SCHEMA.columns
WHERE table_name = 'ps'
```

---

## Filter columns

| Column | Type | Description | Nullable |
|---|---|---|---|
| `default_flag` | int | Default Parameter Set (1 = selected default) | No |
| `tran_flag` | int | Detected by Transits (1 = transit confirmed) | No |

**Why `default_flag = 1`:** The `ps` table contains one row per literature reference per planet. A planet may have dozens of parameter sets from different publications with conflicting values. Setting `default_flag = 1` selects the single parameter set that NASA Archive curators designate as the best available. This ensures one canonical row per planet and avoids duplicates. We inherit the Archive's curation judgment and make no attempt to aggregate across sources.

**Why `tran_flag = 1`:** The Transmission Spectroscopy Metric (TSM) and Emission Spectroscopy Metric (ESM) — the core scoring inputs — require a transit depth measurement and are physically undefined for non-transiting planets. Restricting to transiting planets is a hard prerequisite, not a preference.

---

## Identity columns

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `pl_name` | char | — | Planet Name (e.g. "WASP-39 b") | Never |
| `hostname` | char | — | Host star name | Never |
| `pl_letter` | char | — | Planet letter (b, c, d, …) | Yes |

---

## Planet radius

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `pl_rade` | double | R⊕ | Planet Radius | Yes |
| `pl_radeerr1` | double | R⊕ | Upper uncertainty | Yes |
| `pl_radeerr2` | double | R⊕ | Lower uncertainty (negative) | Yes |

---

## Planet mass

Two mass columns are fetched. `pl_masse` is true mass (requires measured inclination). `pl_bmasse` is mass or m·sin(i), used as a fallback where inclination is unavailable.

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `pl_masse` | double | M⊕ | True planet mass | Yes |
| `pl_masseerr1` | double | M⊕ | Upper uncertainty | Yes |
| `pl_masseerr2` | double | M⊕ | Lower uncertainty (negative) | Yes |
| `pl_bmasse` | double | M⊕ | Mass or M·sin(i) | Yes |
| `pl_bmasseerr1` | double | M⊕ | Upper uncertainty | Yes |
| `pl_bmasseerr2` | double | M⊕ | Lower uncertainty (negative) | Yes |

---

## Orbital parameters

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `pl_orbper` | double | day | Orbital period | Yes |
| `pl_orbpererr1` | double | day | Upper uncertainty | Yes |
| `pl_orbpererr2` | double | day | Lower uncertainty (negative) | Yes |
| `pl_orbsmax` | double | au | Semi-major axis | Yes |

---

## Transit observables

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `pl_trandep` | double | % | Transit depth | Yes |
| `pl_trandeperr1` | double | % | Upper uncertainty | Yes |
| `pl_trandeperr2` | double | % | Lower uncertainty (negative) | Yes |
| `pl_tranmid` | double | day (BJD) | Transit midpoint | Yes |
| `pl_trandur` | double | hour | Transit duration | Yes |

---

## Insolation and equilibrium temperature

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `pl_insol` | double | S⊕ | Insolation flux relative to Earth | Yes |
| `pl_insolerr1` | double | S⊕ | Upper uncertainty | Yes |
| `pl_insolerr2` | double | S⊕ | Lower uncertainty (negative) | Yes |
| `pl_eqt` | double | K | Equilibrium temperature | Yes |
| `pl_eqterr1` | double | K | Upper uncertainty | Yes |
| `pl_eqterr2` | double | K | Lower uncertainty (negative) | Yes |

---

## Stellar parameters

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `st_teff` | double | K | Stellar effective temperature | Yes |
| `st_tefferr1` | double | K | Upper uncertainty | Yes |
| `st_tefferr2` | double | K | Lower uncertainty (negative) | Yes |
| `st_rad` | double | R☉ | Stellar radius | Yes |
| `st_raderr1` | double | R☉ | Upper uncertainty | Yes |
| `st_raderr2` | double | R☉ | Lower uncertainty (negative) | Yes |
| `st_mass` | double | M☉ | Stellar mass | Yes |
| `st_logg` | double | log(cm/s²) | Stellar surface gravity | Yes |
| `st_spectype` | char | — | Spectral type | Yes |

---

## System-level columns

> **Column name corrections (verified 2026-06-26):**
> - `st_dist` **does not exist** in the current `ps` schema. The correct column is `sy_dist`.
> - `st_j` **does not exist** in the current `ps` schema. The correct column is `sy_jmag`.
> Any prior documentation or code using `st_dist` or `st_j` must be updated.

| Column | Type | Unit | Description | Nullable |
|---|---|---|---|---|
| `sy_dist` | double | pc | Distance to system | Yes |
| `sy_disterr1` | double | pc | Upper uncertainty | Yes |
| `sy_disterr2` | double | pc | Lower uncertainty (negative) | Yes |
| `sy_jmag` | double | mag | J (2MASS) magnitude | Yes |
| `sy_jmagerr1` | double | mag | Upper uncertainty | Yes |
| `sy_jmagerr2` | double | mag | Lower uncertainty (negative) | Yes |
| `sy_kmag` | double | mag | Ks (2MASS) magnitude | Yes |
| `sy_tmag` | double | mag | TESS magnitude | Yes |

---

## Reference and provenance columns

| Column | Type | Description | Nullable |
|---|---|---|---|
| `pl_refname` | char | Planetary parameter reference | Yes |
| `st_refname` | char | Stellar parameter reference | Yes |
| `rowupdate` | char | Date of last archive update | Yes |

---

## Columns explicitly not fetched

| Column | Reason not fetched |
|---|---|
| `pl_radj`, `pl_massj` | Redundant with `pl_rade`, `pl_masse` (just unit conversion) |
| `*_str` columns | Human-formatted strings; numeric columns used instead |
| `*_lim` columns | Limit flags deferred to Ticket 2 (cleaning) |
| `pl_controv_flag` | Controversial-planet filtering deferred to Ticket 2 |
| `disc_*` columns | Discovery metadata not used in scoring MVP |
| `sy_vmag`, `sy_bmag`, etc. | Not used in TSM/ESM; J/K preferred per Kempton (2018) |
