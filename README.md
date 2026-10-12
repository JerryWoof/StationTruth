# StationTruth

[![CI](https://github.com/JerryWoof/StationTruth/actions/workflows/ci.yml/badge.svg)](https://github.com/JerryWoof/StationTruth/actions/workflows/ci.yml)
[![Site](https://github.com/JerryWoof/StationTruth/actions/workflows/pages.yml/badge.svg)](https://jerrywoof.github.io/StationTruth/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**How good are weather models, really?**

StationTruth checks the world's two main global forecast models, **ECMWF IFS**
and **NOAA GFS**, against what ~5,000 airports actually measured, every hour.
It learns each model's local habits (*"GFS runs 2 °C too warm at night in this
valley"*) and uses them to make bias-corrected forecasts, then scores those too,
so you can see whether the correction actually helps.

**[🌍 Live map](https://jerrywoof.github.io/StationTruth/)** ·
**[🗺️ Analysis](https://jerrywoof.github.io/StationTruth/analysis.html)** ·
**[📊 Scoreboard](https://jerrywoof.github.io/StationTruth/scores.html)** ·
**[✈️ TAFs vs reality](https://jerrywoof.github.io/StationTruth/tafs.html)** ·
**[📖 Methodology](https://jerrywoof.github.io/StationTruth/about.html)**

## What you can see

- **Error now**: every airport coloured by how wrong each model is right now
  (temperature, dew point, wind, pressure).
- **7-day bias** and **best model here**: which model to trust where.
- **Station charts**: observed weather vs. 24 h forecasts for the last week, plus
  the next 72 h from every model, raw and corrected.
- **Scoreboard**: bias, MAE and RMSE by lead time, model and region.
- **Analysis**: an hourly temperature and wind map that starts from ECMWF and is
  pulled toward what the airports measured (optimal interpolation). At stations
  it never saw, it is about a third more accurate than the model alone.
- **TAFs vs reality**: the official airport forecasts scored against what the
  METARs then showed: hit rate, POD/FAR/CSI for IFR-or-worse, by lead time.

## How it works

```
aviationweather.gov METARs ─► collect-obs (hourly) ───┐
ECMWF open data (0.25°)    ─► collect-fcst (3-hourly) ─┼─► GitHub Release assets (Parquet)
NOAA GFS on AWS (0.25°)    ─►                          ┘              │
                                                                      ▼
                         verify → bias-correct → blend → site (hourly, GitHub Pages)
```

1. **Observations**: METARs from aviationweather.gov, decoded with
   [MetarParse](https://github.com/JerryWoof/MetarParse).
2. **Forecasts**: 00Z and 12Z runs, 0–72 h every 3 h. Only the needed GRIB2
   messages are downloaded (HTTP range requests on the index files), decoded
   with eccodes and bilinearly interpolated to each airport.
3. **Verification**: each forecast is matched with the nearest observation within
   ±30 min. Scores: bias, MAE, RMSE (circular error for wind direction).
4. **Bias correction**: each model's typical error over the previous 14 days,
   estimated in layers (region → station → station at that lead time and time of
   day) with shrinkage, so it is robust from the first days of data. It uses
   **only observations available before the forecast was issued** (no look-ahead,
   enforced by tests). Chosen by out-of-sample comparison with alternatives
   ([scripts/evaluate_correction.py](scripts/evaluate_correction.py)).
5. **Blend**: the average of both corrected models, using each model's latest
   available run.

Everything runs on free infrastructure: GitHub Actions, GitHub Releases for the
history (monthly `data-YYYY-MM` prereleases) and GitHub Pages. The full data
contract is in [docs/DATA.md](docs/DATA.md).

## Run it yourself

```bash
git clone https://github.com/JerryWoof/StationTruth && cd StationTruth
pip install -e ".[dev]"
stationtruth collect-obs --data-dir data            # latest observations
stationtruth collect-fcst --model all --runs 1 --data-dir data
stationtruth build-site --data-dir data --out site/data
cd site && python -m http.server                    # open http://localhost:8000
```

## Data and attribution

- Observations: [aviationweather.gov](https://aviationweather.gov) (NOAA/NWS).
- ECMWF forecasts: Source: [ECMWF](https://www.ecmwf.int), open data licensed
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data has been
  modified (interpolated to stations, bias-corrected).
- GFS forecasts: NOAA, public domain, via the
  [NOAA Open Data Dissemination](https://registry.opendata.aws/noaa-gfs-bdp-pds/) program on AWS.

Not for operational use. This is a verification experiment, not an official
forecast or flight briefing.

## License

MIT
