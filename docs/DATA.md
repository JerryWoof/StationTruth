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

For each model, station, variable, lead-time bucket (0–24, 27–48, 51–72 h) and
valid hour-of-day bucket (00–05, 06–11, 12–17, 18–23 UTC), the correction is
the mean error over the **previous 14 days of verified pairs whose run time is
earlier than the run being corrected** (no look-ahead). Applied only when at
least 8 pairs exist; otherwise the raw forecast is kept. Corrected models are
named `ecmwf_bc`, `gfs_bc`; `blend` is the mean of `ecmwf_bc` and `gfs_bc`
(falls back to whichever exists). Corrected forecasts are verified exactly like
raw ones, so the scoreboard shows whether correction actually helps.

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

```json
{
  "id": "LKPR", "name": "...", "country": "CZ", "lat": 50.1, "lon": 14.26, "elev_m": 364,
  "obs": {"time": [...], "t": [...], "td": [...], "wspd": [...], "gust": [...], "wdir": [...], "mslp": [...]},  // last 7 days, hourly
  "past": {"ecmwf": {"valid": [...], "t": [...], ...}, ...},   // 24 h lead forecasts verifying in the last 7 days
  "future": {"ecmwf": {"run": "2026101212", "valid": [...], "t": [...], ...}, "gfs": {...}, "ecmwf_bc": {...}, "gfs_bc": {...}, "blend": {...}},  // latest run, next 72 h
  "scores": {"7d": {"ecmwf": {"t": {"n": 160, "bias": 0.8, "mae": 1.2, "rmse": 1.6}, ...}, ...}, "30d": {...}}
}
```

Floats rounded to 1 decimal, missing values `null`, compact JSON.
