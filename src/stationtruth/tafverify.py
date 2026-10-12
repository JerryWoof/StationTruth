"""TAF vs reality: verify hourly-expanded TAFs (``stationtruth.tafs``) against METARs.

Matching
--------
Each TAF hour ``H`` (the clock hour ``[H, H+1h)``) at a station is compared
with two observed flight categories (the ``cat`` column of the obs files):

* ``worst`` (primary): the worst category among all METARs and SPECIs with
  ``H <= time < H+1h``. This is what a pilot planning to arrive during that
  hour would care about.
* ``nominal``: the routine METAR nearest to ``H`` within 30 minutes
  (``floor(time + 30 min) == H``; the earliest wins a tie), i.e. "the hourly
  report".

Hours without a usable observation (or with an empty forecast category) are
left out. Every TAF counts, so an hour is usually verified several times, once
per TAF covering it at different lead times (``lead_h = floor(H - issued)``).
A window ``Nd`` holds the hours ``end - N days <= H < end``, with ``end`` the
last complete observed hour.

Scores
------
* ``hit_rate``: share of hours where the prevailing TAF category equals the
  observed one; ``hss``: Heidke skill score of the 4x4 table (1 perfect, 0 no
  better than chance).
* ``table``: 4x4 contingency counts, ``table[f][o]`` with forecast ``f`` and
  observed ``o`` in the order VFR, MVFR, IFR, LIFR.
* ``ifr``: the yes/no event "IFR or worse" (IFR or LIFR), prevailing only:
  ``a`` hits, ``b`` false alarms, ``c`` misses, ``d`` correct negatives,
  ``pod = a/(a+c)``, ``far = b/(a+b)``, ``csi = a/(a+b+c)``,
  ``bias = (a+b)/(a+c)`` (frequency bias, 1 = forecast as often as observed);
  ratios are ``null`` when their denominator is 0.
* ``ifr_tempo``: the same, but the forecast says "yes" when either the
  prevailing conditions or any active TEMPO/PROB/INTER group is IFR or worse.
  Usually raises POD and FAR together.
* ``by_lead``: the same block per lead bucket ``0-6``, ``6-12``, ``12-24``,
  ``24-30`` hours (``[lo, hi)``, the last one holds every lead >= 24 h).

``site/data/taf_scores.json``
-----------------------------
::

    {
      "generated": "2026-10-12T18:00:00Z",
      "first_issued": "2026-10-05T17:20:00Z",   # oldest stored TAF, null if none
      "last_hour": "2026-10-12T17:00:00Z",      # last verified hour, null if none
      "taf_count": 123456, "station_count": 3010,
      "categories": ["VFR", "MVFR", "IFR", "LIFR"],
      "lead_buckets": ["0-6", "6-12", "12-24", "24-30"],
      "station_min_n": 48,                      # station rankings need this many hours
      "station_fields": ["id", "n", "hits", "a", "b", "c", "a_t", "b_t", "c_t"],
      "stations": {"EGLL": ["London/Heathrow", "GB", "Europe"], ...},  # name, country, region
      "windows": {
        "7d": {
          "start": "...", "end": "...",
          "global": {"worst": BLOCK, "nominal": BLOCK},
          "regions": {"Europe": {"worst": BLOCK, "nominal": BLOCK}, ...},
          "stations": [["EGLL", 640, 540, 20, 8, 9, 27, 30, 2], ...]  # rows of station_fields
        },
        "30d": {...}
      }
    }

    BLOCK = {"n": 51234, "hit_rate": 0.81, "hss": 0.52,
             "table": [[...4 ints...] x 4],
             "ifr": {"a": 900, "b": 400, "c": 350, "d": 49584,
                     "pod": 0.72, "far": 0.308, "csi": 0.545, "bias": 1.04},
             "ifr_tempo": {...same keys...},
             "by_lead": [{"lead": "0-6", "n": ..., "hit_rate": ..., "hss": ...,
                          "ifr": {...}, "ifr_tempo": {...}}, ...]}

Station rows use ``worst`` matching: ``n`` verified hours, ``hits`` hours in
the right category, ``a``/``b``/``c`` IFR-or-worse hits, false alarms and
misses for the prevailing forecast, ``*_t`` the same counting TEMPO/PROB/INTER
(POD/FAR/CSI are derived on the page). Stations with fewer than
``STATION_LIST_MIN_N`` verified hours are omitted. Ratios are rounded to 3
decimals.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .regions import REGIONS, region_index

CATS: tuple[str, ...] = ("VFR", "MVFR", "IFR", "LIFR")  # code 0..3, larger = worse
CAT_CODE = {c: i for i, c in enumerate(CATS)}
IFR_CODE = 2  # "IFR or worse" = code >= 2
LEAD_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0-6", 0, 6),
    ("6-12", 6, 12),
    ("12-24", 12, 24),
    ("24-30", 24, 10_000),
)
WINDOWS: dict[str, int] = {"7d": 7, "30d": 30}
MODES = ("worst", "nominal")
NOMINAL_TOL_MIN = 30
STATION_LIST_MIN_N = 24  # stations with fewer verified hours are not listed
STATION_RANK_MIN_N = 48  # the page ranks stations only above this

_TAF_RE = re.compile(r"^taf-(\d{4}-\d{2}-\d{2})\.parquet$")
_OBS_RE = re.compile(r"^obs-(\d{4}-\d{2}-\d{2})\.parquet$")
TAF_COLUMNS = ["station", "issued", "hour", "lead_h", "prev_cat", "tempo_cat"]

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def _iso(t) -> str | None:
    if t is None or pd.isna(t):
        return None
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _codes(s: pd.Series) -> np.ndarray:
    """Category strings -> int8 codes (VFR 0 .. LIFR 3), -1 for empty/unknown."""
    return s.map(CAT_CODE).fillna(-1).astype("int8").to_numpy()


# --------------------------------------------------------------------------- loading


def _files(data_dir: Path, rx: re.Pattern, start: datetime | None) -> list[Path]:
    out = []
    for p in sorted(Path(data_dir).glob("*.parquet")):
        m = rx.match(p.name)
        if not m:
            continue
        if start is not None and datetime.fromisoformat(m[1]).date() < start.date():
            continue
        out.append(p)
    return out


def load_tafs(data_dir: Path, start: datetime | None = None) -> pd.DataFrame:
    """TAF hours issued from ``start - 2 days`` (or everything)."""
    lo = None if start is None else start - timedelta(days=2)
    frames = [pd.read_parquet(p, columns=TAF_COLUMNS) for p in _files(data_dir, _TAF_RE, lo)]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in TAF_COLUMNS})
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["station", "issued", "hour"], keep="last").reset_index(drop=True)


def load_obs(data_dir: Path, start: datetime | None = None) -> pd.DataFrame:
    lo = None if start is None else start - timedelta(days=1)
    cols = ["station", "time", "type", "cat"]
    frames = [pd.read_parquet(p, columns=cols) for p in _files(data_dir, _OBS_RE, lo)]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.concat(frames, ignore_index=True).drop_duplicates(["station", "time"])


def load_stations(data_dir: Path) -> pd.DataFrame:
    p = Path(data_dir) / "stations.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["station", "name", "country", "lat", "lon"])
    return pd.read_parquet(p)


# --------------------------------------------------------------------------- matching


def observed_hours(obs: pd.DataFrame) -> pd.DataFrame:
    """Per ``(station, hour)``: ``worst`` and ``nominal`` observed category codes (-1 = none)."""
    if obs.empty:
        return pd.DataFrame(
            {
                "station": pd.Series(dtype=str),
                "hour": pd.Series(dtype="datetime64[ns, UTC]"),
                "worst": pd.Series(dtype="int8"),
                "nominal": pd.Series(dtype="int8"),
            }
        )
    o = obs[["station", "time", "type", "cat"]].copy()
    o["time"] = pd.to_datetime(o["time"], utc=True)
    o["code"] = _codes(o["cat"].fillna(""))
    o = o[o["code"] >= 0]

    o["hour"] = o["time"].dt.floor("h")
    worst = o.groupby(["station", "hour"], sort=False)["code"].max().rename("worst")

    m = o[o["type"] == "METAR"].copy()
    m["hour"] = (m["time"] + pd.Timedelta(minutes=NOMINAL_TOL_MIN)).dt.floor("h")
    m["dist"] = (m["time"] - m["hour"]).abs()
    m = m[m["dist"] <= pd.Timedelta(minutes=NOMINAL_TOL_MIN)]
    m = m.sort_values(["station", "hour", "dist", "time"])
    nominal = m.drop_duplicates(["station", "hour"]).set_index(["station", "hour"])["code"]
    out = pd.concat([worst, nominal.rename("nominal")], axis=1).reset_index()
    out["worst"] = out["worst"].fillna(-1).astype("int8")
    out["nominal"] = out["nominal"].fillna(-1).astype("int8")
    return out


def match(tafs: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """TAF hours joined with observed categories.

    Columns: ``station``, ``hour``, ``lead_h``, ``fc`` (prevailing code),
    ``tc`` (TEMPO code, -1 none), ``worst``, ``nominal`` (-1 when missing).
    Rows without a prevailing category or without any observation are dropped.
    """
    t = pd.DataFrame(
        {
            "station": tafs["station"].astype(str).to_numpy(),
            "hour": pd.to_datetime(tafs["hour"], utc=True).to_numpy(),
            "lead_h": tafs["lead_h"].astype("int16").to_numpy(),
            "fc": _codes(tafs["prev_cat"].fillna("")),
            "tc": _codes(tafs["tempo_cat"].fillna("")),
        }
    )
    t["hour"] = pd.to_datetime(t["hour"], utc=True)
    t = t[t["fc"] >= 0]
    oh = observed_hours(obs)
    oh["hour"] = pd.to_datetime(oh["hour"], utc=True)
    m = t.merge(oh, on=["station", "hour"], how="inner")
    m = m[(m["worst"] >= 0) | (m["nominal"] >= 0)]
    return m.reset_index(drop=True)


# --------------------------------------------------------------------------- scores


def _ratio(num: float, den: float, nd: int = 3) -> float | None:
    return round(num / den, nd) if den > 0 else None


def event_scores(a: int, b: int, c: int, d: int) -> dict:
    """POD, FAR, CSI and frequency bias from a 2x2 table (hits, false alarms, misses, c.n.)."""
    a, b, c, d = int(a), int(b), int(c), int(d)
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "pod": _ratio(a, a + c),
        "far": _ratio(b, a + b),
        "csi": _ratio(a, a + b + c),
        "bias": _ratio(a + b, a + c),
    }


def heidke(table: np.ndarray) -> float | None:
    """Heidke skill score of a square contingency table."""
    t = np.asarray(table, dtype="float64")
    n = t.sum()
    if n <= 0:
        return None
    pc = np.trace(t) / n
    pe = float((t.sum(axis=1) * t.sum(axis=0)).sum()) / (n * n)
    if pe >= 1:
        return None
    return round((pc - pe) / (1 - pe), 3)


def _counts(fc: np.ndarray, tc: np.ndarray, ob: np.ndarray, group: np.ndarray, n_groups: int):
    """Per group: 4x4 table (n_groups,4,4) and the TEMPO-aware 2x2 counts (n_groups,2,2)."""
    tab = np.bincount(group * 16 + fc * 4 + ob, minlength=n_groups * 16).reshape(n_groups, 4, 4)
    yes_t = ((fc >= IFR_CODE) | (tc >= IFR_CODE)).astype(np.int64)
    obs_yes = (ob >= IFR_CODE).astype(np.int64)
    ev = np.bincount(group * 4 + yes_t * 2 + obs_yes, minlength=n_groups * 4)
    return tab, ev.reshape(n_groups, 2, 2)


def _block_from(tab: np.ndarray, ev_t: np.ndarray) -> dict:
    """Score block from one 4x4 table and its TEMPO-aware 2x2 ([fc_yes][obs_yes])."""
    n = int(tab.sum())
    yes_f = slice(IFR_CODE, 4)
    no_f = slice(0, IFR_CODE)
    a = tab[yes_f, yes_f].sum()
    b = tab[yes_f, no_f].sum()
    c = tab[no_f, yes_f].sum()
    d = tab[no_f, no_f].sum()
    return {
        "n": n,
        "hit_rate": _ratio(np.trace(tab), n),
        "hss": heidke(tab),
        "table": tab.astype(int).tolist(),
        "ifr": event_scores(a, b, c, d),
        "ifr_tempo": event_scores(ev_t[1, 1], ev_t[1, 0], ev_t[0, 1], ev_t[0, 0]),
    }


def lead_bucket(lead_h: np.ndarray) -> np.ndarray:
    out = np.full(len(lead_h), -1, dtype=np.int64)
    for k, (_, lo, hi) in enumerate(LEAD_BUCKETS):
        out[(lead_h >= lo) & (lead_h < hi)] = k
    return out


def score_block(m: pd.DataFrame, mode: str) -> dict:
    """Full block (with ``by_lead``) for matched rows ``m`` against ``mode`` observations."""
    ob = m[mode].to_numpy().astype(np.int64)
    ok = ob >= 0
    fc = m["fc"].to_numpy().astype(np.int64)[ok]
    tc = m["tc"].to_numpy().astype(np.int64)[ok]
    ob = ob[ok]
    lb = lead_bucket(m["lead_h"].to_numpy()[ok])
    zero = np.zeros(len(fc), dtype=np.int64)
    tab, ev = _counts(fc, tc, ob, zero, 1)
    block = _block_from(tab[0], ev[0])
    sel = lb >= 0
    tab_l, ev_l = _counts(fc[sel], tc[sel], ob[sel], lb[sel], len(LEAD_BUCKETS))
    block["by_lead"] = []
    for k, (name, _, _) in enumerate(LEAD_BUCKETS):
        b = _block_from(tab_l[k], ev_l[k])
        b.pop("table")
        block["by_lead"].append({"lead": name, **b})
    return block


STATION_FIELDS: tuple[str, ...] = ("id", "n", "hits", "a", "b", "c", "a_t", "b_t", "c_t")


def station_scores(m: pd.DataFrame) -> list[list]:
    """Compact per-station rows (``worst`` matching), fields ``STATION_FIELDS``.

    ``hits`` is the number of hours in the right category; ``a``/``b``/``c``
    are IFR-or-worse hits, false alarms and misses (prevailing only) and
    ``*_t`` the same counting TEMPO/PROB/INTER groups.
    """
    ok = m["worst"].to_numpy() >= 0
    mm = m[ok]
    if mm.empty:
        return []
    ids, sidx = np.unique(mm["station"].to_numpy(), return_inverse=True)
    fc = mm["fc"].to_numpy().astype(np.int64)
    tc = mm["tc"].to_numpy().astype(np.int64)
    ob = mm["worst"].to_numpy().astype(np.int64)
    tab, ev = _counts(fc, tc, ob, sidx.astype(np.int64), len(ids))
    out = []
    yes, no = slice(IFR_CODE, 4), slice(0, IFR_CODE)
    for k, sid in enumerate(ids):
        t = tab[k]
        n = int(t.sum())
        if n < STATION_LIST_MIN_N:
            continue
        out.append(
            [
                str(sid),
                n,
                int(np.trace(t)),
                int(t[yes, yes].sum()),
                int(t[yes, no].sum()),
                int(t[no, yes].sum()),
                int(ev[k][1, 1]),
                int(ev[k][1, 0]),
                int(ev[k][0, 1]),
            ]
        )
    return out


def station_info(st: pd.DataFrame, ids: set[str], regions: dict[str, str]) -> dict[str, list]:
    """``{id: [name, country, region]}`` for the listed stations."""
    out: dict[str, list] = {}
    if len(st):
        sub = st[st["station"].isin(ids)].drop_duplicates("station")
        for sid, name, cc in zip(sub["station"], sub["name"], sub["country"], strict=True):
            out[str(sid)] = [str(name or ""), str(cc or ""), regions.get(sid)]
    for sid in sorted(ids - set(out)):
        out[sid] = ["", "", None]
    return out


def station_regions(st: pd.DataFrame) -> dict[str, str]:
    if st.empty:
        return {}
    idx = region_index(st["country"].tolist(), st["lat"].to_numpy(), st["lon"].to_numpy())
    return {s: REGIONS[i] for s, i in zip(st["station"], idx, strict=True) if i >= 0}


def compute(
    tafs: pd.DataFrame,
    obs: pd.DataFrame,
    stations: pd.DataFrame,
    now: datetime | None = None,
    windows: dict[str, int] | None = None,
) -> dict:
    """Build the ``taf_scores.json`` document (see module docstring)."""
    now = now or datetime.now(timezone.utc)
    windows = windows or WINDOWS
    doc: dict = {
        "generated": _iso(now),
        "first_issued": None,
        "last_hour": None,
        "taf_count": 0,
        "station_count": 0,
        "categories": list(CATS),
        "lead_buckets": [b[0] for b in LEAD_BUCKETS],
        "station_min_n": STATION_RANK_MIN_N,
        "station_fields": list(STATION_FIELDS),
        "stations": {},
        "windows": {},
    }
    if tafs.empty:
        return doc
    issued = pd.to_datetime(tafs["issued"], utc=True)
    doc["first_issued"] = _iso(issued.min())
    doc["taf_count"] = int(len(tafs[["station", "issued"]].drop_duplicates()))
    doc["station_count"] = int(tafs["station"].nunique())
    if obs.empty:
        return doc
    obs_end = pd.to_datetime(obs["time"], utc=True).max()
    end = min(obs_end.floor("h"), pd.Timestamp(now).floor("h"))
    m = match(tafs, obs)
    m = m[m["hour"] < end]
    if m.empty:
        return doc
    doc["last_hour"] = _iso(m["hour"].max())
    regions = station_regions(stations)
    reg = m["station"].map(regions)
    for wname, days in windows.items():
        start = end - pd.Timedelta(days=days)
        sel = (m["hour"] >= start).to_numpy()
        mw = m[sel]
        rw = reg[sel]
        if mw.empty:
            continue
        w = {
            "start": _iso(max(start, mw["hour"].min())),
            "end": _iso(end),
            "global": {mode: score_block(mw, mode) for mode in MODES},
            "regions": {},
        }
        for r in REGIONS:
            mr = mw[(rw == r).to_numpy()]
            if len(mr):
                w["regions"][r] = {mode: score_block(mr, mode) for mode in MODES}
        w["stations"] = station_scores(mw)
        doc["windows"][wname] = w
    listed = {row[0] for w in doc["windows"].values() for row in w["stations"]}
    doc["stations"] = station_info(stations, listed, regions)
    return doc


def build(data_dir: Path, out_dir: Path, now: datetime | None = None) -> dict:
    """Load data, compute and write ``out_dir/taf_scores.json``; returns a summary."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=max(WINDOWS.values()))
    t = time.monotonic()
    tafs = load_tafs(Path(data_dir), start)
    obs = load_obs(Path(data_dir), start)
    stations = load_stations(Path(data_dir))
    log(f"tafverify: {len(tafs)} TAF hours, {len(obs)} obs ({time.monotonic() - t:.1f}s)")
    doc = compute(tafs, obs, stations, now=now)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "taf_scores.json"
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    summary = {
        "taf_count": doc["taf_count"],
        "station_count": doc["station_count"],
        "first_issued": doc["first_issued"],
        "last_hour": doc["last_hour"],
        "bytes": len(text),
    }
    for wname, w in doc["windows"].items():
        g = w["global"]["worst"]
        summary[wname] = {
            "n": g["n"],
            "hit_rate": g["hit_rate"],
            "csi": g["ifr"]["csi"],
            "csi_tempo": g["ifr_tempo"]["csi"],
        }
    log(f"tafverify: wrote {path} ({len(text) / 1e3:.0f} kB, {time.monotonic() - t:.1f}s)")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m stationtruth.tafverify",
        description="Verify stored TAFs against METARs and write taf_scores.json.",
    )
    p.add_argument("--data-dir", required=True, help="directory with taf-/obs- parquet files")
    p.add_argument("--out", default="site/data")
    args = p.parse_args(argv)
    summary = build(Path(args.data_dir), Path(args.out))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
