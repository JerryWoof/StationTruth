# StationTruth data contract

Everything runs on free infrastructure: GitHub Actions collects and verifies,
GitHub Release assets store the history, GitHub Pages serves the site.

```
 aviationweather.gov METAR cache ──► collect-obs (hourly) ──┐
 ECMWF open data (IFS 0.25°)    ──► collect-fcst (3-hourly) ─┼─► Release "data-YYYY-MM" assets
 NOAA GFS on AWS (0.25°)        ──►                          ┘            │
                                                                         ▼
                                       verify + correct + sitegen (hourly) ──► GitHub Pages
```

All times are UTC. Parquet files use timezone-aware `datetime64[ns, UTC]`
columns. Units: °C, knots, degrees, hPa, metres, feet, percent.

## Storage: GitHub Releases

One prerelease per month, tag `data-YYYY-MM` (month of the file's date), with
flat asset names:

| Asset | Content |
|---|---|
| `obs-YYYY-MM-DD.parquet` | observations for that UTC day (appended hourly) |
| `fcst-ecmwf-YYYYMMDDHH.parquet` | one ECMWF run, interpolated to stations |
| `fcst-gfs-YYYYMMDDHH.parquet` | one GFS run, interpolated to stations |
| `stations.parquet` | station list (also in each month, overwritten) |

`stationtruth.storage` uploads and downloads these with the GitHub REST API
(token from `GITHUB_TOKEN` / `GH_TOKEN`; repo from `GITHUB_REPOSITORY`,
default `JerryWoof/StationTruth`). Locally, every command can work on a plain
directory holding files with exactly these names (`--data-dir`).

## `stations.parquet`

| column | type | notes |
|---|---|---|
| `station` | str | ICAO id |
| `name` | str | |
| `country` | str | ISO 3166-1 alpha-2 or empty |
| `lat`, `lon` | float64 | degrees, lon in [-180, 180) |
| `elev_m` | float32 | station elevation |

Source: aviationweather.gov `stations.cache.json.gz`, limited to stations that
report METARs.

## `obs-YYYY-MM-DD.parquet`

One row per decoded METAR/SPECI (deduplicated on `station`, `time`). Decoded
with MetarParse from the raw text.

| column | type | notes |
|---|---|---|
| `station` | str | |
| `time` | datetime UTC | observation time |
| `type` | str | METAR / SPECI |
| `t`, `td` | float32 | °C (US `T` remark group used when present: 0.1 °C) |
| `wdir` | float32 | degrees, NaN when variable/calm/missing |
| `wspd`, `gust` | float32 | kt, gust NaN when not reported |
| `vis_m` | float32 | prevailing visibility (CAVOK/9999 -> 10000) |
| `ceil_ft` | float32 | NaN when no ceiling |
| `qnh` | float32 | hPa |
| `slp` | float32 | hPa, US `SLPppp` remark when present |
| `cat` | str | VFR/MVFR/IFR/LIFR or empty |

## `fcst-<model>-YYYYMMDDHH.parquet`

Model fields bilinearly interpolated to each station. Steps 0–72 h every 3 h,
runs 00Z and 12Z.

| column | type | notes |
|---|---|---|
| `station` | str | |
| `run` | datetime UTC | model initialisation time |
| `valid` | datetime UTC | `run + lead_h` |
| `lead_h` | int16 | |
| `t`, `td` | float32 | °C at 2 m |
| `wspd` | float32 | kt at 10 m, from u/v |
| `wdir` | float32 | degrees (meteorological, direction wind comes FROM) |
| `gust` | float32 | kt (ECMWF `10fg`, GFS `GUST:surface`); NaN if missing |
| `mslp` | float32 | hPa (ECMWF `msl`, GFS `PRMSL`) |
| `tcc` | float32 | total cloud cover % |
| `vis_m` | float32 | GFS `VIS:surface`; NaN for ECMWF |

No height correction is applied: the difference between model orography and
station elevation is part of what the bias correction learns.

## Verification rules (`stationtruth.verify`)

* A forecast row is matched with the observation at the same station closest
  to `valid`, within ±30 minutes; routine METARs win ties over SPECIs.
* Variables verified: `t`, `td`, `wspd`, `gust` (only when both report a gust),
  `mslp` vs `slp` (only where SLP is reported), `wdir` (circular error, only when
  observed wspd >= 5 kt).
* Error = forecast − observed. Scores: count `n`, `bias` (mean error), `mae`,
  `rmse`.

## Bias correction (`stationtruth.correct`)

The correction for run R of a raw model (`ecmwf`, `gfs`) uses only verified
pairs that were available at R: valid time ≤ R, run time **strictly earlier
than R**, valid time within the previous 14 days (no look-ahead). Errors are
forecast − observed. It is a mean error estimated at three nested levels, each
shrunk toward the next broader one with a fixed pseudo-count k
("the level's own pairs plus k imaginary pairs at the broader mean"):

| level | group | estimate |
|---|---|---|
| region prior | region × lead bucket × valid-hour bucket, all stations | `b_r = S_r / (n_r + 20)` |
| station | station, all lead and hour buckets | `b_s = (S_s + 5·b_r) / (n_s + 5)` |
| cell | station × lead bucket × valid-hour bucket | `b_c = (S_c + 10·b_s) / (n_c + 10)` |

`S`, `n` are the sum of errors and the number of pairs of the group in the
window, separately per model and variable (`t`, `td`, `wspd`, `gust`, `mslp`).
Lead buckets: 0–24, 27–48, 51–72 h; valid-hour buckets: 00–05, 06–11, 12–17,
18–23 UTC; regions as in `scores.json` (one global prior when no region map is
passed to `stationtruth.correct.build_models`).
The corrected forecast is `raw − b_c`; `wspd` and `gust` are clipped at 0 and
the corrected `td` is capped at the corrected `t`. Only when no pair at all is
available (empty station and empty region bucket) is the raw forecast kept.
`wdir` is not corrected. Corrected models are named `ecmwf_bc`, `gfs_bc`;
they are verified exactly like raw ones, so the scoreboard shows whether
correction actually helps. `meta.json` `correction.min_pairs` is therefore 1.

The scheme and its constants were chosen with `scripts/evaluate_correction.py`,
an out-of-sample evaluation that learns every candidate exactly as above (only
pairs available at each run) and scores the corrected runs. On 8 days of data
(7 ECMWF and 14 GFS runs) it reduced RMSE versus raw for every model and
variable, in every lead bucket and at every amount of history, and beat the
previous rule (cell mean, applied only with ≥ 8 pairs), plain shrinkage toward
zero, median-based (robust) estimates and a multiplicative wind correction.

`blend` is built for every run time R of either model (the blend run is named
by R). For each model, the component is that model's corrected forecast from
its **latest run at or before R**, used only if that run is at most 12 h older
than R and covers the valid time within 72 h of its own initialisation. The
components are matched on station and valid time (so their leads differ when one
run is older), `lead_h = valid − R` (0–72 h), and the blend is their mean
(circular mean for `wdir`). Where only one component is available, the blend is
that component. No run later than R is ever used and each component's
correction is itself look-ahead free, so the blend is too.

Note that `gust` is verified only when the observation reports a gust, so its
scores describe a conditional (gusty-conditions) sample, not all hours. The
gust correction is learned from that same sample, so it describes gusty hours
and may overstate gusts in calm hours, which no score can reveal.

## Site data (`site/data/`, generated, not committed)

### `meta.json`

```json
{
  "generated": "2026-10-12T16:00:00Z",
  "first_obs": "2026-10-05T00:00:00Z",
  "obs_count": 812345,
  "station_count": 5100,
  "runs": {"ecmwf": ["2026101200", "..."], "gfs": ["..."]},
  "latest_run": {"ecmwf": "2026101200", "gfs": "2026101206"},
  "models": ["ecmwf", "gfs", "ecmwf_bc", "gfs_bc", "blend"],
  "pairs": 1234567
}
```

### `scores.json`

```json
{
  "windows": {
    "7d": {
      "global": {
        "ecmwf": {
          "t":    {"by_lead": [{"lead_h": 0, "n": 4800, "bias": 0.3, "mae": 1.4, "rmse": 1.9}, ...],
                   "all": {"n": 120000, "bias": 0.2, "mae": 1.6, "rmse": 2.2}},
          "td": {...}, "wspd": {...}, "gust": {...}, "mslp": {...}, "wdir": {...}
        },
        "gfs": {...}, "ecmwf_bc": {...}, "gfs_bc": {...}, "blend": {...}
      },
      "regions": {"Europe": {<same shape as global>}, "North America": {...}, ...}
    },
    "30d": {...}
  }
}
```

Regions: Europe, North America, Central America & Caribbean, South America,
Africa, Middle East, Asia, Oceania, Polar (|lat| > 66.5).

### `latest.json`

Array, one entry per station with a recent observation (≤ 3 h old):

```json
{
  "id": "LKPR", "name": "Prague/Havel Arpt", "country": "CZ",
  "lat": 50.101, "lon": 14.26,
  "time": "2026-10-12T15:30:00Z",
  "obs": {"t": 12.0, "td": 6.0, "wspd": 6, "gust": null, "wdir": 260, "mslp": null},
  "fc": {
    "ecmwf": {"run": "2026101200", "lead_h": 15, "t": 13.1, "td": 5.2, "wspd": 9.1, "gust": 15.0, "wdir": 250, "mslp": 1021.3},
    "gfs": {...}, "ecmwf_bc": {...}, "gfs_bc": {...}, "blend": {...}
  },
  "bias7": {"ecmwf": {"t": 0.8, "wspd": 2.1}, "gfs": {...}},   // 7-day mean error, null if < 8 pairs
  "best": "ecmwf_bc"   // lowest 7-day temperature RMSE among models with n >= 8, or null
}
```

`fc.*` is the forecast for the observation's hour from the most recent run
that covers it (interpolated linearly in time between 3-hourly steps).

### `stations/<ID>.json`

One file per station with an observation in the last 7 days (~8 KB each,
~45 MB for ~5,600 stations). Only the variables the station panel shows are
included: `t`, `td`, `wspd`, `mslp` (obs `mslp` is the observed SLP).

```json
{
  "id": "LKPR", "name": "...", "country": "CZ", "lat": 50.101, "lon": 14.26, "elev_m": 364,
  "scale": 10,
  "obs": {"start": "2026-10-05T19:00:00Z", "step": 60, "t": [140, -20, null, 10, ...], "td": [...], "wspd": [...]},
  "past": {"ecmwf": {"start": "2026-10-05T21:00:00Z", "step": 180, "t": [...], ...}, ...},
  "future": {"ecmwf": {"run": "2026101212", "start": "2026-10-12T12:00:00Z", "step": 180, "t": [...], ...}, "gfs": {...}, "ecmwf_bc": {...}, "gfs_bc": {...}, "blend": {...}},
  "scores": {"7d": {"ecmwf": {"t": [160, 0.8, 1.2, 1.6], "td": [...], "wspd": [...], "mslp": [...]}, ...}, "30d": {...}}
}
```

* **Series blocks** (`obs`, `past.<model>`, `future.<model>`) are regular time
  grids: value *i* is valid at `start + i × step` (`step` in minutes). Slots
  without data are `null` — readers must not join lines across them.
* **Values** are integers in units of `1 / scale` (tenths), **delta-coded**: the
  first non-null number is the value itself, each later one the difference to
  the previous non-null value. Decode with a running sum that skips nulls,
  then divide by `scale`: `[140, -20, null, 10]` → `[14.0, 12.0, null, 13.0]`.
* A variable with no value in a block is left out of that block (e.g. `mslp`
  where the station reports no SLP); a block without any value is left out.
* `obs`: hourly (`step` 60), the report nearest each full hour (±30 min, METAR
  wins ties), last 7 days. Leading/trailing hours without data are trimmed.
* `past.<model>`: the 24 h lead forecast (nearest lead in 21–27 h) verifying in
  the last 7 days, normally 3-hourly.
* `future.<model>`: the model's latest run (`run`), every step (0–72 h).
* `scores.<window>.<model>.<var>`: `[n, bias, mae, rmse]` over all lead times,
  1 decimal, `null` when `n` is 0.

Compact JSON, no `NaN`.
