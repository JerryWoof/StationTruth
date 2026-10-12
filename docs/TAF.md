# TAF vs reality

StationTruth also verifies the official airport forecasts — TAFs (Terminal
Aerodrome Forecasts) — against what the same airport's METARs later reported.
Code: `stationtruth.tafs` (collection, hourly expansion) and
`stationtruth.tafverify` (matching, scores). Page: `site/tafs.html`.

```
 aviationweather.gov tafs.cache.xml.gz ──► tafs collect (hourly) ──► taf-YYYY-MM-DD.parquet
 aviationweather.gov Data API (history) ──► tafs backfill (once)  ──┘           │
                                                    obs-*.parquet ──► tafverify ──► site/data/taf_scores.json
```

## Collection

* `python -m stationtruth.tafs collect [--data-dir DIR]` downloads the TAF
  cache (all TAFs currently in force, ~3,000 airports, ~350 kB gzipped),
  decodes every TAF with MetarParse `parse_taf`, expands it hourly and merges
  it into `taf-YYYY-MM-DD.parquet` for the UTC day of its issue time. A TAF is
  identified by `(station, issued)`; collecting it again replaces it.
  TAFs are typically reissued every 3–6 hours, so an hourly run catches every
  scheduled TAF and nearly every amendment.
* `python -m stationtruth.tafs backfill --days N [--data-dir DIR]` recovers
  past TAFs from the Data API. `GET /api/data/taf?ids=...&format=raw&date=YYYYMMDD_HHMM`
  returns, per station, the latest TAF issued at or before that time (`hours`
  is not accepted; probed October 2026, history reaches back about one month).
  The backfill takes one snapshot per hour (`--step-h`) for each chunk of at
  most 400 stations (the stations currently in the TAF cache, or `--ids`),
  with a pause between requests (`--pause`, default 0.5 s, `--workers` 2) and
  the StationTruth User-Agent. 7 days ≈ 1,350 requests, ≈ 70 MB, ≈ 20 minutes.
  Only amendments that were superseded within the snapshot step are missed.

Storage: same backends as everything else (`--data-dir` directory or the
monthly `data-YYYY-MM` GitHub release); `storage.names_for_window` includes
TAF files issued from two days before the window start, so `stationtruth sync`
downloads them too.

## `taf-YYYY-MM-DD.parquet`

One row per TAF and forecast hour.

| column | type | notes |
|---|---|---|
| `station` | str | ICAO id |
| `issued` | datetime UTC | issue time |
| `hour` | datetime UTC | start of the forecast hour `[hour, hour+1h)` |
| `lead_h` | int16 | `floor(hour − issued)` in hours |
| `prev_cat` | str | VFR/MVFR/IFR/LIFR of the prevailing forecast, or empty |
| `prev_vis_m` | float32 | prevailing visibility, m (CAVOK / 9999 → 10000) |
| `prev_ceil_ft` | float32 | prevailing ceiling (lowest BKN/OVC/VV), NaN if none |
| `prev_wspd`, `prev_gust` | float32 | kt; gust NaN if none |
| `tempo_cat` | str | worst category among TEMPO/PROB/INTER groups active that hour, or empty |
| `tempo_prob` | int8 | largest probability of those groups (100 = plain TEMPO/INTER), 0 if none |
| `amended` | bool | `TAF AMD` |
| `raw` | str | TAF text (repeated per hour, dictionary-encoded in Parquet) |

About 60,000 rows (0.2 MB) per snapshot of the cache; a full day of TAFs is a
few hundred thousand rows, ~1–2 MB.

## Hourly expansion rules

* Hours are kept only when they start at or after the issue time and inside
  the validity period, so a TAF is never scored for an hour already under way
  when it was issued. TAFs that are NIL, cancelled (`CNL`) or longer than 36 h
  are skipped.
* **Prevailing** conditions for hour `H` are those in effect at `H`: the base
  forecast, then the `FM` and `BECMG` groups whose effective time is `≤ H`,
  applied in order of effective time.
  * `FM` replaces the whole forecast from its time.
  * `BECMG` changes only the elements it mentions and **takes effect at the end
    of its period** (`BECMG 1214/1216` applies from 16Z). The change may happen
    at any time inside the period, but the end is when the forecaster commits
    to the new conditions; using it avoids crediting or blaming the TAF for the
    transition window. (`CAVOK` = 10 km, no cloud, no weather; `NSW` clears
    weather.)
* **Temporary** groups — `TEMPO`, `PROB30`/`PROB40` (also `PROB40 TEMPO`) and
  `INTER` — do not change the prevailing state. A group is active in hour `H`
  when its period overlaps `[H, H+1h)`; it is overlaid on the prevailing state
  (unmentioned elements kept) and its category taken. `tempo_cat` is the worst
  of the active groups.
* Categories are MetarParse `derived.flight_category` (FAA: LIFR ceiling
  < 500 ft or vis < 1 SM; IFR < 1,000 ft or < 3 SM; MVFR ≤ 3,000 ft or ≤ 5 SM),
  identical to the observed `cat` column.

## Verification

Each TAF hour is compared with the observed category in two ways:

* `worst` (primary, used for station rankings): the worst category in any
  METAR or SPECI with `H ≤ time < H+1h`;
* `nominal`: the routine METAR nearest to `H` within 30 minutes.

Hours without a usable observation or forecast category are dropped. Every
TAF covering an hour counts at its own lead time; amended and superseded TAFs
are all included. Windows `7d` and `30d` end at the last complete observed
hour.

Scores (for the yes/no event **IFR or worse** = IFR or LIFR): `a` hits, `b`
false alarms, `c` misses, `d` correct negatives;
POD = a/(a+c), FAR = b/(a+b), CSI = a/(a+b+c), frequency bias = (a+b)/(a+c).
`ifr` uses the prevailing forecast only; `ifr_tempo` counts the forecast as
"yes" when the prevailing forecast **or** any active TEMPO/PROB/INTER group is
IFR or worse. Category hit rate = trace / n of the 4×4 table; Heidke skill
score = (hit rate − chance) / (1 − chance). Lead buckets: 0–6, 6–12, 12–24,
24–30 h (`[lo, hi)`, the last also holds the rare leads ≥ 30 h).

## `site/data/taf_scores.json`

Written by `python -m stationtruth.tafverify --data-dir DIR --out site/data`.

```json
{
  "generated": "2026-10-12T18:00:00Z",
  "first_issued": "2026-10-05T17:20:00Z",
  "last_hour": "2026-10-12T17:00:00Z",
  "taf_count": 123456,
  "station_count": 3010,
  "categories": ["VFR", "MVFR", "IFR", "LIFR"],
  "lead_buckets": ["0-6", "6-12", "12-24", "24-30"],
  "station_min_n": 48,
  "station_fields": ["id", "n", "hits", "a", "b", "c", "a_t", "b_t", "c_t"],
  "stations": {"EGLL": ["London/Heathrow", "GB", "Europe"]},
  "windows": {
    "7d": {
      "start": "2026-10-05T18:00:00Z", "end": "2026-10-12T18:00:00Z",
      "global": {"worst": BLOCK, "nominal": BLOCK},
      "regions": {"Europe": {"worst": BLOCK, "nominal": BLOCK}},
      "stations": [["EGLL", 640, 540, 20, 8, 9, 27, 30, 2]]
    },
    "30d": {}
  }
}
```

```
BLOCK = {"n": 51234, "hit_rate": 0.81, "hss": 0.52,
         "table": [[VFR→VFR, VFR→MVFR, VFR→IFR, VFR→LIFR], ...],   // rows forecast, cols observed
         "ifr":       {"a": 900, "b": 400, "c": 350, "d": 49584,
                       "pod": 0.72, "far": 0.308, "csi": 0.545, "bias": 1.04},
         "ifr_tempo": {...},
         "by_lead": [{"lead": "0-6", "n": ..., "hit_rate": ..., "hss": ...,
                      "ifr": {...}, "ifr_tempo": {...}}, ...]}
```

Top-level `stations` maps each listed airport to `[name, country, region]`
(region `null` when unknown). Each window's `stations` is a list of rows in
`station_fields` order, `worst` matching only: `n` verified hours, `hits`
hours in the right category, `a`/`b`/`c` IFR-or-worse hits, false alarms and
misses for the prevailing forecast, `a_t`/`b_t`/`c_t` the same counting
TEMPO/PROB/INTER groups. The page derives POD, FAR and CSI from them (rows
keep the payload small: ~3,000 airports × 2 windows).

Ratios are rounded to 3 decimals and `null` when undefined (zero
denominator). Window `stations` lists airports with ≥ 24 verified hours; the page
ranks those with ≥ `station_min_n` hours and ≥ 5 hits + misses + false alarms.
When no TAFs are stored yet, `windows` is `{}` and the page shows a
"collecting TAFs" notice.
