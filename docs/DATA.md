# StationTruth data contract

Everything runs on free infrastructure: GitHub Actions collects and verifies,
GitHub Release assets store the history, GitHub Pages serves the site.

```
 aviationweather.gov METAR cache ──► collect-obs (hourly) ──┐
 ECMWF open data (IFS, AIFS)    ──► collect-fcst (3-hourly) ─┼─► Release "data-YYYY-MM" assets
 NOAA on AWS (GFS, AIGFS)       ──►                          │            │
 DWD open data (ICON)           ──►                          ┘            │
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
| `fcst-<model>-YYYYMMDDHH.parquet` | one run of `<model>` (`ecmwf`, `gfs`, `icon`, `aifs`, `aigfs`), interpolated to stations |
| `stations.parquet` | station list (also in each month, overwritten) |

`stationtruth.storage` uploads and downloads these with the GitHub REST API
(token from `GITHUB_TOKEN` / `GH_TOKEN`; repo from `GITHUB_REPOSITORY`,
default `JerryWoof/StationTruth`). Locally, every command can work on a plain
directory holding files with exactly these names (`--data-dir`).

## Models (`stationtruth.registry`)

`src/stationtruth/registry.py` is the single source of truth for the model
list: download sources, CLI choices (`--model all`), the verified model names,
the corrections and the blend all derive from it. `site/common.js` keeps a
matching copy (labels, colours; checked by `tests/test_registry.py`).

| id | model | kind | grid | steps used | fields | open-data retention |
|---|---|---|---|---|---|---|
| `ecmwf` | ECMWF IFS (HRES) | physics | 0.25° | 0–72 h / 3 h | all but `vis_m` | ~4 days |
| `gfs` | NOAA GFS | physics | 0.25° | 0–72 h / 3 h | all | years (AWS) |
| `icon` | DWD ICON global | physics | ~13 km icosahedral (R3B7) | 0–72 h / 3 h | all but `vis_m` | ~24 h (latest run per hour) |
| `aifs` | ECMWF AIFS Single | AI | 0.25° | 0–72 h / 6 h | `t`, `td`, wind, `mslp`, `tcc` | ~3 days |
| `aigfs` | NOAA AIGFS (GraphCast-based) | AI | 0.25° | 0–72 h / 6 h | `t`, `td`, wind, `mslp` | months (AWS) |

Runs 00Z and 12Z for every model. Sources:

* **ECMWF IFS / AIFS**: `https://data.ecmwf.int/forecasts/YYYYMMDD/HHz/{ifs|aifs-single}/0p25/oper/YYYYMMDDHH0000-{step}h-oper-fc.grib2`
  with a JSON-lines `.index`; only the needed messages (`2t`, `2d`, `10u`, `10v`,
  `msl`, `tcc`, IFS also `10fg`) are fetched by HTTP range requests (~58 MB per AIFS run).
  AIFS has no `10fg`, so its `gust` is NaN.
* **GFS**: `noaa-gfs-bdp-pds` on AWS, `pgrb2.0p25` with wgrib2 `.idx`.
* **AIGFS**: `https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/aigfs.YYYYMMDD/HH/model/atmos/grib2/aigfs.tHHz.sfc.fFFF.grib2`
  (+ `.idx`): `TMP`/`DPT` 2 m, `UGRD`/`VGRD` 10 m, `PRMSL` (~48 MB per run). No
  gust, cloud or visibility. (The older experimental `graphcastgfs.*` prefixes in
  the same bucket stopped in May 2026.)
* **ICON**: `https://opendata.dwd.de/weather/nwp/icon/grib/HH/{param}/icon_global_icosahedral_single-level_YYYYMMDDHH_SSS_{PARAM}.grib2.bz2`,
  one bz2 file per parameter and step (`T_2M`, `TD_2M`, `U_10M`, `V_10M`,
  `VMAX_10M`, `PMSL`, `CLCT`; ~3.7 MB each, ~645 MB and ~1 min per run). Global
  ICON is published only on the native unstructured grid: the cell-centre
  coordinates (`clat`/`clon` time-invariant files of the run) give each station the
  inverse-distance-weighted mean of its 3 nearest cells (≤ ~12 km away). The
  station mapping is cached per grid UUID and station list in
  `$STATIONTRUTH_CACHE` (default `~/.cache/stationtruth`). `VMAX_10M` is the
  maximum gust of the hour before the step (none at step 0).

Verification, correction and scoring treat every model alike. The AI models'
6-hourly output means they are scored only at 6-hourly lead times.

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

Model fields interpolated to each station (bilinear on regular grids, 3-nearest
inverse-distance weights on ICON's unstructured grid). Steps 0–72 h every 3 h
(AIFS, AIGFS: every 6 h), runs 00Z and 12Z. Fields a model does not provide are NaN.

| column | type | notes |
|---|---|---|
| `station` | str | |
| `run` | datetime UTC | model initialisation time |
| `valid` | datetime UTC | `run + lead_h` |
| `lead_h` | int16 | |
| `t`, `td` | float32 | °C at 2 m |
| `wspd` | float32 | kt at 10 m, from u/v |
| `wdir` | float32 | degrees (meteorological, direction wind comes FROM) |
| `gust` | float32 | kt (ECMWF `10fg`, GFS `GUST:surface`, ICON `VMAX_10M`); NaN for AIFS, AIGFS and at step 0 of ECMWF/ICON |
| `mslp` | float32 | hPa (ECMWF/AIFS `msl`, GFS/AIGFS `PRMSL`, ICON `PMSL`) |
| `tcc` | float32 | total cloud cover % |
| `vis_m` | float32 | GFS `VIS:surface`; NaN for all other models |

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

The correction for run R of a raw model (every registry model) uses only verified
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
`wdir` is not corrected. Corrected models are named `<id>_bc` (`ecmwf_bc`, `gfs_bc`, `icon_bc`, `aifs_bc`, `aigfs_bc`);
they are verified exactly like raw ones, so the scoreboard shows whether
correction actually helps. `meta.json` `correction.min_pairs` is therefore 1.

The scheme and its constants were chosen with `scripts/evaluate_correction.py`,
an out-of-sample evaluation that learns every candidate exactly as above (only
pairs available at each run) and scores the corrected runs. On 8 days of data
(7 ECMWF and 14 GFS runs) it reduced RMSE versus raw for every model and
variable, in every lead bucket and at every amount of history, and beat the
previous rule (cell mean, applied only with ≥ 8 pairs), plain shrinkage toward
zero, median-based (robust) estimates and a multiplicative wind correction.

`blend` is the mean of **all** corrected models. It is built for every run
time R of any model (the blend run is named by R). For each model, the component is that model's corrected forecast from
its **latest run at or before R**, used only if that run is at most 12 h older
than R and covers the valid time within 72 h of its own initialisation. The
components are matched on station and valid time (so their leads differ when one
run is older), `lead_h = valid − R` (0–72 h), and the blend is their mean
(circular mean for `wdir`), per row and variable over the components that have
a value: a model without the variable (AIFS/AIGFS `gust`) or without that lead
(the AI models' 3-hourly gaps) does not contribute, so at e.g. +3 h the blend is
the mean of the 3-hourly models only. Where only one component is available, the
blend is that component. No run later than R is ever used and each component's
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
  "runs": {"ecmwf": ["2026101200", "..."], "gfs": ["..."], "icon": [], "aifs": [], "aigfs": []},
  "latest_run": {"ecmwf": "2026101200", "gfs": "2026101206"},
  "models": ["ecmwf", "gfs", "icon", "aifs", "aigfs", "ecmwf_bc", "gfs_bc", "icon_bc", "aifs_bc", "aigfs_bc", "blend"],
  "pairs": 1234567
}
```

`models` lists the models that have data in this build (registry order: raw,
corrected, blend); the site shows only those. `scores.json` keeps every registry
model (with `n = 0` blocks when it has no data).

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

### `compare.json`

Head-to-head comparisons (`stationtruth.compare`, written by the site build;
the scoreboard's "Head to head" section). Model ids are those of `meta.json`
`models`, so new models are compared automatically.

**Common sample.** For models A and B, a pair enters the comparison only when
both have a verified forecast for the same station, the same valid time and the
same lead time (hence the same initialisation time; stricter than "same lead
bucket"), for that variable. Both errors then refer to the same observation.
Variables: `t`, `td`, `wspd`, `gust`, `mslp` (verification rules as above);
leads 0–72 h; windows by valid time as in `scores.json`.

**Uncertainty: block bootstrap.** Errors are correlated in space (one weather
system spoils a region's forecasts) and time, so pairs are not independent. The
pairs are reduced to sums per **24 h period (counted back from the build time)
× region × lead bucket**: `n`, Σe_A², Σe_B², Σ|e_A|, Σ|e_B|. Each of 2,000
bootstrap replicates resamples these blocks, never rows:

* per region, the region's days with common data are resampled with a
  *moving-block* bootstrap: runs of `L = round(days^(1/3))` consecutive days
  (1 day for ≤ 3 days, 2 for 4–15, 3 for 16–42), so day-to-day persistence of
  the errors is partly kept;
* globally, every region is resampled independently (stratified by region; the
  stations without a known region form one more stratum) and the regions are
  summed. Regions are thousands of km apart, beyond the synoptic decorrelation
  length, which gives ~10× more blocks than resampling global days.

ΔRMSE = RMSE_A − RMSE_B and ΔMAE = MAE_A − MAE_B are computed from the summed
blocks of each replicate; the 95% interval is the 2.5–97.5 percentile range.
A difference is *significant* when the interval excludes 0. No interval is
given with fewer than 3 days or 30 pairs. Remaining approximation: day blocks
(even in runs) do not capture all persistence, so intervals may be slightly
narrow; `lag1` reports the lag-1 autocorrelation of the global daily ΔMSE
(windows with ≥ 10 days) to judge this. The random stream is fixed per pair,
variable and window: a rebuild on the same data gives the same intervals, also
when other models are added. Errors are kept as int16 hundredths
internally (RMSE changes < 1e-4).

**Station wins.** `frac_a_better`: among stations with ≥ 10 common pairs, the
share where A has the lower RMSE (ties count half).

**All-models common sample** (`common`): per variable, the pairs verified by
*every* included model, and each model's scores on it — an apples-to-apples
ranking. A model is included when it has ≥ 25% of the largest model's pair
count in the window (a model collected only for a day would otherwise shrink
everyone's sample to that day); the others are listed in `excluded`.

```json
{
  "generated": "2026-10-12T18:17:00Z",
  "models": ["ecmwf", "gfs", "ecmwf_bc", "gfs_bc", "blend"],
  "vars": ["t", "td", "wspd", "gust", "mslp"],
  "buckets": ["0-24", "27-48", "51-72", "all"],
  "regions": ["Europe", "..."],
  "fields": ["n", "stations", "rmse_a", "rmse_b", "d_rmse", "d_rmse_lo", "d_rmse_hi",
             "mae_a", "mae_b", "d_mae", "d_mae_lo", "d_mae_hi", "frac_a_better", "days"],
  "common_fields": ["n", "bias", "mae", "rmse"],
  "method": {"resamples": 2000, "ci": 0.95, "seed": 20261012, "...": "..."},
  "windows": {
    "7d": {
      "from": "...", "to": "...", "days": 7,
      "global": {
        "ecmwf|gfs": {"t": [[290000, 5400, 1.99, 2.27, -0.27, -0.33, -0.23, ...], [...], [...], [...]], "td": [...]},
        "...": {}
      },
      "regions": {"Europe": {"ecmwf|gfs": {...}}, "...": {}},
      "common": {
        "t": {"models": ["ecmwf", "..."], "excluded": [],
              "global": {"ecmwf": [[n, bias, mae, rmse], [...], [...], [...]], "...": []},
              "regions": {"Europe": {"ecmwf": [...]}}}
      },
      "lag1": {"ecmwf|gfs": {"t": 0.4}}
    },
    "30d": {"...": "..."}
  },
  "timings": {"result": 2.0}
}
```

* Each pair is stored once, as `"A|B"` with A before B in `models` order; every
  difference is **A − B** (negative: A more accurate). For B vs A negate the
  differences, swap `*_a`/`*_b`, swap and negate the interval ends, and use
  `1 − frac_a_better`.
* A pair/variable entry is a list with one row (`fields`) per bucket in
  `buckets` order (`all` = 0–72 h); a bucket without common pairs is `null`.
  Interval ends and `frac_a_better` may be `null` (too little data). `days` =
  days with common data (global: in any region).
* Pairs, variables and regions without any common pair are left out.

### `health.json`

Pipeline health (`stationtruth.health`), written at the end of every site build
and again by `python -m stationtruth.health --data-dir data --out site/data`
(run after the TAF and analysis steps so their status is final; a re-run keeps
the build details of the same site build). Shown on `status.html`.

```json
{
  "generated": "2026-10-12T18:17:00Z",
  "status": "ok",                         // ok | degraded | down = worst reason
  "reasons": [{"level": "degraded", "component": "runs:gfs", "msg": "..."}],  // levels: down, degraded, info
  "obs": {
    "last_obs": "...", "age_min": 61, "last_collect": "...",
    "stations_3h": 5246, "stations_24h": 5556, "median_reports": 8979,
    "hours": {"start": "...", "step": 60, "reports": [48 ints], "stations": [48 ints],
              "state": ["ok" | "low" | "gap" | "partial", ...]},
    "gaps": [{"from": "...", "to": "...", "hours": 2, "kind": "gap" | "low"}],
    "files": [{"date": "2026-10-12", "rows": 156622, "stored": "..." | null}]
  },
  "runs": {
    "days": 7, "cycle_hours": [0, 12], "listing": true,
    "models": {"gfs": {"latest": "2026101212", "latest_age_h": 6.0, "stored": 14, "expected": 14,
                       "missing": [], "extra": [], "median_late_h": 5.3, "grace_h": 10,
                       "cycles": [{"run": "2026101212", "status": "ok", "stored": "...", "late_h": 5.3, "rows": 193275},
                                  {"run": "...", "status": "missing" | "pending" | "before"}]}}
  },
  "tafs": {"last_issued": "...", "age_h": 0.1, "files": [{"date": "...", "rows": 229831, "stored": null}]},
  "analysis": {"status": "ok" | "stale" | "not_built", "generated": "...", "valid": "...", "age_min": 5},
  "taf_scores": {"status": "ok" | "not_built", "generated": "...", "...": "..."},
  "build": {"generated": "...", "seconds": 14.8, "timings": {...}, "peak_rss_mb": 1496,
            "errors": {"compare": "RuntimeError: ..."}, "models": [...], "pairs": 1919195}
}
```

* Observation hours are counted by observation time over the 48 h ending with
  the current hour. An hour is `partial` until 70 min after its end (reports
  arrive late and are collected hourly); a complete hour is `gap` with no
  report and `low` below half the 48 h median. Gaps ending in the last 6 h
  make the status `degraded`, older ones are `info`. No observation for 3 h:
  `down`.
* Runs: every model with a `fcst-<model>-*.parquet` in the last 14 days;
  expected cycles 00Z and 12Z of the last 7 days, *due* `grace_h` after
  initialisation (ECMWF 12 h, GFS 10 h, others 12 h). Cycles before the
  model's first stored run are `before`. Missing latest due run(s):
  `degraded`; older missing runs: `info`; no run of any model for 36 h: `down`.
* `stored` / `late_h` come from the asset listing in `data/.sync.json`
  (GitHub `updated_at`, or the source mtime for a local store), never from
  local file mtimes; without a listing they are `null`. Backfilled runs show
  long delays.
* TAFs: newest `issued` older than 8 h: `degraded`. Analysis older than 3 h:
  `degraded`; not built yet (the hourly job builds it after the scores): `info`.
  A failed optional build step (`build.errors`, also in `meta.json`
  `optional_errors`): `degraded`. The status page additionally reports `down`
  when `health.json` itself is more than 3 h old.
