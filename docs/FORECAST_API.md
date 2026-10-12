# StationTruth forecast API

A free, static JSON feed with the StationTruth **blend forecast** for every
airport that reported a METAR in the last 3 hours (~5,000 worldwide): hourly for
the rest of the latest 72-hour run, with an 80 % uncertainty band learned from
that airport's own recent forecast errors.

* **No key, no sign-up, no rate-limit negotiation**: plain files on GitHub Pages.
* **CORS-friendly**: GitHub Pages sends `Access-Control-Allow-Origin: *`, so
  browsers can fetch the files from any origin (checked with
  `curl -I https://jerrywoof.github.io/StationTruth/data/meta.json`, which
  returns `access-control-allow-origin: *` and `cache-control: max-age=600`).
* Human-readable page: <https://jerrywoof.github.io/StationTruth/forecast.html#LKPR>

> **Not for operational use.** This is an experimental, automatically generated
> product without monitoring or guarantees. Do not use it for flight planning,
> safety-of-life or other operational decisions: use official forecasts (TAF,
> your national meteorological service).

## Endpoints

| URL | Content |
|---|---|
| `https://jerrywoof.github.io/StationTruth/data/forecast/index.json` | list of airports that have a forecast file |
| `https://jerrywoof.github.io/StationTruth/data/forecast/<ICAO>.json` | forecast for one airport (ICAO id in upper case, e.g. `EGLL`) |
| `https://jerrywoof.github.io/StationTruth/data/meta.json` | site build info (model runs available, build time) |

An airport without a recent observation has no file (HTTP 404); its file
reappears with its next report. Files are served gzip-compressed (~3 KB per
airport on the wire).

## Update frequency

The site is rebuilt every hour (at about HH:40 UTC). New model runs arrive
roughly 6–9 hours after their 00Z/12Z initialisation (GFS also 06Z/18Z when
collected), so the forecast *content* changes every 6–12 hours while the
`obs`, the skill numbers and the starting hour advance hourly. GitHub's CDN may
cache a file for up to 10 minutes. Please poll at most once an hour per airport;
use `generated` to skip unchanged data.

## `index.json`

```json
{
  "generated": "2026-10-12T18:17:00Z",
  "count": 5184,
  "latest_run": {"ecmwf": "2026101200", "gfs": "2026101212"},
  "stations": [
    {"id": "EGLL", "name": "London/Heathrow", "country": "GB", "lat": 51.477, "lon": -0.461},
    ...
  ]
}
```

## `<ICAO>.json`

```json
{
  "id": "LKPR", "name": "Prague/Havel Arpt", "country": "CZ",
  "lat": 50.101, "lon": 14.26, "elev_m": 364,
  "generated": "2026-10-12T18:17:00Z",
  "scale": 10,
  "obs": {"time": "2026-10-12T17:00:00Z", "t": 10.0, "td": 5.0, "wspd": 4.0, "wdir": 270.0, "gust": null, "mslp": null},
  "blend": {
    "run": "2026101212", "start": "2026-10-12T18:00:00Z", "step": 60,
    "t": [84, -3, -4, ...], "td": [...], "wspd": [...], "wdir": [...], "gust": [...],
    "mslp": [...], "tcc": [...], "rh": [...], "feels": [...],
    "t_lo": [...], "t_hi": [...], "td_lo": [...], "td_hi": [...],
    "wspd_lo": [...], "wspd_hi": [...], "gust_lo": [...], "gust_hi": [...],
    "mslp_lo": [...], "mslp_hi": [...]
  },
  "band": {
    "q": [0.1, 0.9], "window_days": 14, "min_sample": 30,
    "buckets": [[0, 12], [13, 24], [25, 36], [37, 48], [49, 60], [61, 72]],
    "t": {"src": ["station", "station", ...], "n": [67, 49, ...], "lo": [-1.9, ...], "hi": [1.3, ...]},
    "td": {...}, "wspd": {...}, "gust": {...}, "mslp": {...}
  },
  "models": {
    "ecmwf":    {"run": "2026101200", "start": "2026-10-12T18:00:00Z", "step": 180, "t": [...], ..., "tcc": [...]},
    "gfs":      {...},
    "aifs":     {...}, "aigfs": {...},
    "ecmwf_bc": {"run": "2026101200", "start": "...", "step": 180, "t": [...], ...},
    "gfs_bc":   {...}, "aifs_bc": {...}, "aigfs_bc": {...}
  },
  "skill": {
    "window_days": 14,
    "models": {"blend": {"t": [272, -0.09, 1.26, 1.62], "td": [...], "wspd": [...], "gust": null, "mslp": null, "wdir": [...]}, "ecmwf": {...}, ...},
    "summary": {
      "var": "t", "n": 272, "mae": 1.26,
      "better_pct": {"ecmwf": 3, "gfs": 7, "ecmwf_bc": -14, "gfs_bc": 9},
      "common_n": {"ecmwf": 120, "gfs": 272, "ecmwf_bc": 120, "gfs_bc": 272},
      "text": "Blend temperature error here: 1.3 °C typical (mean absolute error, last 14 days), 3% better than ECMWF and 7% better than GFS."
    }
  }
}
```

### Time series encoding

`blend` and each `models.<model>` block is a regular time grid, the same compact
encoding as the site's station files (docs/DATA.md):

* value *i* is valid at `start + i × step` (`step` in **minutes**: 60 for the
  blend, 180 or 360 for the models); all times UTC;
* values are integers in units of `1 / scale` (tenths) and **delta-coded**: the
  first non-null number is the value itself, every later one the difference to
  the previous non-null value. Decode with a running sum that skips `null`s,
  then divide by `scale`: `[140, -20, null, 10]` → `[14.0, 12.0, null, 13.0]`;
* `null` = no value (do not draw lines across it); a variable without any
  value is left out of the block.

### Variables and units

| key | unit | notes |
|---|---|---|
| `t`, `td` | °C | 2 m temperature, dew point |
| `wspd`, `gust` | kt | 10 m wind speed, gust (×1.852 = km/h, ×0.5144 = m/s) |
| `wdir` | degrees | direction the wind comes **from** (not corrected) |
| `mslp` | hPa | mean sea-level pressure |
| `tcc` | % | total cloud cover, raw model output (not verified or corrected); blend = mean of the component runs |
| `rh` | % | relative humidity from `t` and `td` (Magnus formula) |
| `feels` | °C | wind chill (t ≤ 10 °C, wind > 4.8 km/h), heat index (t ≥ 26.7 °C, RH ≥ 40 %), else `t` |
| `<var>_lo`, `<var>_hi` | as `<var>` | 80 % uncertainty band, see below |

### Models

* `blend`: the mean of all bias-corrected models' latest runs (each at most
  12 h older than the blend run), the most accurate product on the
  [scoreboard](https://jerrywoof.github.io/StationTruth/scores.html). The blend
  is hourly, linearly interpolated in time from the 3-hourly model steps (wind
  direction along the shorter arc). It covers the hours from the current hour
  to `run + 72 h` (usually 60–66 h ahead).
* `<model>` (e.g. `ecmwf`, `gfs`, `aifs`, `aigfs`): raw model output interpolated
  to the airport, the model's latest run. Blocks are 3-hourly
  (`step` 180) or 6-hourly (`step` 360) for models that only provide 6-hourly
  steps (the AI models `aifs`, `aigfs`).
* `<model>_bc` (e.g. `ecmwf_bc`): the same run after StationTruth's bias
  correction (learned from the last 14 days of verification; see docs/DATA.md).

The set of models follows the model registry (`src/stationtruth/registry.py`)
and the data actually collected; treat the keys of `models` and `skill.models`
as an open set.

### Uncertainty band

For each variable and lead bucket (0–12, 13–24, …, 61–72 h after the blend run)
StationTruth collects the blend's errors *e = forecast − observed* over the last
14 days and takes their empirical 10th and 90th percentiles. The band is
`[forecast − q90, forecast − q10]`, so in 80 % of past cases the observation fell
inside it. It uses the airport's own pairs when there are at least 30 in the
bucket (`band.<var>.src = "station"`), otherwise all airports of its region
(`"region"`, regions as on the scoreboard), otherwise all airports (`"global"`).
`band.<var>.lo/hi` are the offsets added to the forecast per bucket; between
bucket centres they are interpolated linearly. Wind bands are clipped at 0.

In an out-of-sample check on real data (forecasts built on 10 Oct 18Z, compared
with the next two days' METARs at all airports) 81.5 % of temperature and 78.8 %
of wind speed observations fell inside the band. The band describes *typical*
error at this airport; it is not an ensemble spread and does not widen on
unusually uncertain days. `gust` is verified only when a gust is reported, so
its band describes gusty hours; `mslp` needs a reported sea-level pressure (often
regional).

### Skill

`skill.models.<model>.<var>` = `[n, bias, mae, rmse]` over this airport's
verified forecasts of the last 14 days (all lead times; error = forecast −
observed; `null` when there are none). `skill.summary.better_pct.<model>` =
`100 × (1 − MAE_blend / MAE_model)` for temperature, computed only on the
forecasts both made (same run, same valid time; `common_n` pairs) and only with
at least 20 of them. `summary` is `null` while the airport has fewer than 20
verified blend forecasts.

## Examples

### curl

```sh
curl -s https://jerrywoof.github.io/StationTruth/data/forecast/EGLL.json \
  | jq '{run: .blend.run, start: .blend.start, summary: .skill.summary.text}'
```

### Python

```python
import json
import urllib.request
from datetime import datetime, timedelta

URL = "https://jerrywoof.github.io/StationTruth/data/forecast/{}.json"


def decode(block: dict, scale: int) -> dict[str, list]:
    """Grid block -> {var: [float | None]} (delta-coded tenths)."""
    out = {}
    for key, arr in block.items():
        if not isinstance(arr, list):
            continue
        acc, vals = 0, []
        for x in arr:
            if x is None:
                vals.append(None)
            else:
                acc += x
                vals.append(acc / scale)
        out[key] = vals
    return out


with urllib.request.urlopen(URL.format("LKPR")) as r:
    doc = json.load(r)
b = doc["blend"]
vals = decode(b, doc["scale"])
t0 = datetime.fromisoformat(b["start"].replace("Z", "+00:00"))
for i in range(0, 24, 3):
    t = t0 + timedelta(minutes=i * b["step"])
    print(
        f"{t:%a %H}Z  {vals['t'][i]:5.1f} °C  ({vals['t_lo'][i]:.1f} to {vals['t_hi'][i]:.1f})"
        f"  wind {vals['wdir'][i]:3.0f}° {vals['wspd'][i]:.0f} kt"
    )
print(doc["skill"]["summary"]["text"] if doc["skill"]["summary"] else "no track record yet")
```

### JavaScript (browser or Node 18+)

```js
const url = (id) => `https://jerrywoof.github.io/StationTruth/data/forecast/${id}.json`;

function decode(block, scale) {
  const t0 = Date.parse(block.start);
  const out = { time: [] };
  for (const [k, arr] of Object.entries(block)) {
    if (!Array.isArray(arr)) continue;
    let acc = 0;
    out[k] = arr.map((x) => (x == null ? null : (acc += x) / scale));
    out.time = arr.map((_, i) => new Date(t0 + i * block.step * 60000));
  }
  return out;
}

const doc = await (await fetch(url('KJFK'))).json();
const fc = decode(doc.blend, doc.scale);
console.log(doc.name, 'blend run', doc.blend.run);
fc.time.slice(0, 6).forEach((t, i) => console.log(t.toISOString(), fc.t[i], '°C', fc.wspd[i], 'kt'));
```

## Licence and attribution

* Forecast data contain modified **Copernicus/ECMWF** open data: *"Source:
  ECMWF"*, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  StationTruth interpolated, bias-corrected and blended it. NOAA **GFS** data
  are in the public domain. Observations come from NOAA's
  [aviationweather.gov](https://aviationweather.gov).
* The StationTruth files are offered under the same terms (CC BY 4.0). When you
  use them, credit **"StationTruth (https://github.com/JerryWoof/StationTruth),
  contains modified ECMWF data (CC BY 4.0) and NOAA GFS data"**.
* Provided as is, without warranty of any kind; availability and format may
  change (the format is documented here and in docs/DATA.md). Not for
  operational use.

## How the files are built

`python -m stationtruth.forecastgen --data-dir data --out site/data`
(`src/stationtruth/forecastgen.py`) runs in the hourly *Publish site* workflow
after `stationtruth build-site`. It reloads the archive, verifies and corrects
exactly as the rest of the site, and writes `forecast/<ID>.json` plus
`forecast/index.json` (~54 MB uncompressed, ~14 MB gzipped, for ~5,200 airports and four models; ~10 s). Files
of airports that went quiet are deleted.
