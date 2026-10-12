"""Build the static site data (docs/DATA.md, "Site data").

``build(data_dir, out_dir, now)`` writes ``meta.json``, ``scores.json``,
``latest.json`` and ``stations/<ID>.json`` into ``out_dir``, then the optional
``compare.json`` (stationtruth.compare) and ``health.json``
(stationtruth.health); those two never fail the build, their errors go to
``meta.json`` ``optional_errors``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import compare, correct, health, verify
from .verify import FC_VARS, MODELS, NAT, Table, iso, run_id

log = logging.getLogger(__name__)

LATEST_MAX_AGE_MIN = 180
STATION_DAYS = 7
PAST_LEAD = 24
PAST_LEADS = (21, 27)
MIN_N = 8
#: station JSON variables (the ones the station panel charts and scores);
#: obs "mslp" is the observed SLP
OUT_VARS: tuple[str, ...] = ("t", "td", "wspd", "mslp")
#: station series are integers in units of 1/SCALE (tenths)
SCALE = 10


# ------------------------------------------------------------------ helpers


def _r(x, nd: int = 1):
    """Round a scalar for JSON; NaN/None -> None."""
    if x is None:
        return None
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, nd) + 0.0  # no "-0.0"


def _pylist(arr: np.ndarray, nd: int = 1) -> list:
    """Array -> list of rounded Python floats with None for NaN."""
    r = np.round(np.asarray(arr, dtype=np.float64), nd) + 0.0
    return [None if x != x else x for x in r.tolist()]


def _dump(obj, path: Path) -> int:
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(s, encoding="utf-8")
    tmp.replace(path)
    return len(s)


def _subset(T: Table, mask_or_idx) -> Table:
    return {k: v[mask_or_idx] for k, v in T.items()}


def _sorted_by(T: Table, *cols: str) -> Table:
    order = np.lexsort(tuple(T[c] for c in reversed(cols)))
    return _subset(T, order)


def _bounds(sorted_sid: np.ndarray, sids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.searchsorted(sorted_sid, sids, side="left"),
        np.searchsorted(sorted_sid, sids, side="right"),
    )


def _interp_dir(d0: np.ndarray, d1: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (d0 + w * verify.wrap_dir(d1 - d0)) % 360.0


def _peak_rss_mb() -> int:
    try:
        import resource
        import sys

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r / 2**20 if sys.platform == "darwin" else r / 1024)  # bytes vs KiB
    except Exception:  # pragma: no cover - non-Unix
        return -1


# ----------------------------------------------------------------- latest


def forecast_at(T: Table, sids: np.ndarray, times: np.ndarray, n_sid: int) -> pd.DataFrame:
    """Forecast of the most recent run covering (sid, time), linear in time.

    Returns a frame indexed by sid with ``run``, ``lead_h`` and FC_VARS.
    """
    tq = np.full(n_sid, NAT, dtype=np.int64)
    tq[sids] = times
    # cheap prefilter: runs at most 72 h + a bit before the oldest query time
    t_min = int(times.min()) if len(times) else 0
    pre = np.flatnonzero(T["run"] >= t_min - (verify.MAX_LEAD + 3) * 60)
    t = tq[T["sid"][pre]]
    valid = T["valid"][pre].astype(np.int64)
    ok = (t != NAT) & (T["run"][pre] <= t) & (np.abs(valid - t) <= 360)  # 6-hourly AI models
    rows = pre[ok]
    cols = ["sid", "run", "valid", "lead", *FC_VARS]
    df = pd.DataFrame({c: T[c][rows] for c in cols})
    df["tq"] = tq[df["sid"].to_numpy()]
    if df.empty:
        return pd.DataFrame(columns=["run", "lead_h", *FC_VARS])
    # drop_duplicates, not groupby().last(): the latter skips NaNs column by column
    # and would mix values from different rows
    df = df.sort_values(["sid", "run", "valid"])
    lower = df[df["valid"] <= df["tq"]].drop_duplicates(["sid", "run"], keep="last")
    upper = df[df["valid"] >= df["tq"]].drop_duplicates(["sid", "run"], keep="first")
    j = lower.merge(upper, on=["sid", "run"], suffixes=("0", "1"), how="inner")
    if j.empty:
        return pd.DataFrame(columns=["run", "lead_h", *FC_VARS])
    j = j.sort_values(["sid", "run"]).drop_duplicates("sid", keep="last").set_index("sid")
    span = (j["valid1"] - j["valid0"]).to_numpy(np.float64)
    w = np.where(span > 0, (j["tq0"] - j["valid0"]).to_numpy(np.float64) / np.maximum(span, 1), 0)
    res = pd.DataFrame(index=j.index)
    res["run"] = j["run"]
    res["lead_h"] = ((j["tq0"] - j["run"]) // 60).astype(int)
    for v in FC_VARS:
        a = j[f"{v}0"].to_numpy(np.float64)
        b = j[f"{v}1"].to_numpy(np.float64)
        x = _interp_dir(a, b, w) if v == "wdir" else a + w * (b - a)
        res[v] = np.where(w == 0, a, np.where(w == 1, b, x))  # exact step: other end may be NaN
    return res


def build_latest(ds: verify.Dataset, tables: dict[str, Table], station_sums: dict) -> list[dict]:
    obs = ds.obs
    if len(obs["sid"]) == 0:
        return []
    sid = obs["sid"]
    last = np.flatnonzero(np.r_[sid[1:] != sid[:-1], True])
    last = last[(ds.now - obs["time"][last]) <= LATEST_MAX_AGE_MIN]
    if len(last) == 0:
        return []
    sids = sid[last].astype(np.int64)
    times = obs["time"][last]
    fcs = {m: forecast_at(T, sids, times, ds.n_sid).to_dict("index") for m, T in tables.items()}
    s7 = station_sums.get("7d", {})
    st = ds.stations
    obs_vals = {
        v: _pylist(obs[verify.OBS_COL[v]][last])
        for v in ("t", "td", "wspd", "gust", "wdir", "mslp")
    }
    out = []
    for i, s in enumerate(sids):
        row = st.iloc[s]
        fc: dict = {}
        for m in MODELS:
            r = fcs.get(m, {}).get(s)
            if r is None:
                continue
            fc[m] = {
                "run": run_id(int(r["run"])),
                "lead_h": int(r["lead_h"]),
                **{v: _r(r[v]) for v in FC_VARS},
            }
        bias7: dict = {}
        best, best_rmse = None, np.inf
        for m in MODELS:
            sums = s7.get(m)
            if not sums:
                continue
            b = {}
            for v in ("t", "wspd"):
                S = sums[v]
                n = S.n[s]
                b[v] = _r(S.s[s] / n) if n >= MIN_N else None
            bias7[m] = b
            S = sums["t"]
            if S.n[s] >= MIN_N:
                rmse = np.sqrt(S.ss[s] / S.n[s])
                if rmse < best_rmse:
                    best, best_rmse = m, rmse
        out.append(
            {
                "id": ds.station_ids[s],
                "name": row["name"],
                "country": row["country"],
                "lat": _r(row["lat"], 3),
                "lon": _r(row["lon"], 3),
                "time": iso(int(times[i])),
                "obs": {v: obs_vals[v][i] for v in obs_vals},
                "fc": fc,
                "bias7": bias7,
                "best": best,
            }
        )
    return out


# ----------------------------------------------------------------- stations


def _past_rows(T: Table, active: np.ndarray, lo: int, hi: int) -> Table:
    """Lead-24 (or nearest lead in 21..27) forecast per (sid, valid) in (lo, hi]."""
    m = (
        (T["valid"] > lo)
        & (T["valid"] <= hi)
        & (T["lead"] >= PAST_LEADS[0])
        & (T["lead"] <= PAST_LEADS[1])
        & active[T["sid"]]
    )
    S = _subset(T, np.flatnonzero(m))
    S["dist"] = np.abs(S["lead"].astype(np.int16) - PAST_LEAD)
    S = _sorted_by(S, "sid", "valid", "dist", "lead")
    if len(S["sid"]):
        first = np.r_[True, (S["sid"][1:] != S["sid"][:-1]) | (S["valid"][1:] != S["valid"][:-1])]
        S = _subset(S, first)
    return S


def _tenths(a: np.ndarray) -> list:
    """Float array -> delta-coded ints in 1/SCALE units, None for NaN.

    The first number is the value itself, every later number the difference to
    the previous non-null value (lossless; decode with a running sum that skips
    nulls).
    """
    r = np.round(np.asarray(a, dtype=np.float64) * SCALE)
    out: list = []
    prev = 0
    for x in r.tolist():
        if x != x:
            out.append(None)
        else:
            x = int(x)
            out.append(x - prev)
            prev = x
    return out


def _grid(times: np.ndarray, S: Table, rows: slice, step: int | None = None) -> dict | None:
    """Regular time grid block ``{"start", "step", <var>: [...]}`` (docs/DATA.md).

    ``times`` are minutes; slots without a value are ``null`` so charts can break
    lines across gaps. Leading/trailing rows without any value are dropped, and so
    are variables without any value.
    ``step`` (minutes) defaults to the gcd of the time offsets (>= 60).
    """
    t = times[rows].astype(np.int64)
    vals = {v: np.asarray(S[v][rows], dtype=np.float64) for v in OUT_VARS}
    has = np.zeros(len(t), dtype=bool)
    for a in vals.values():
        has |= ~np.isnan(a)
    if not has.any():
        return None
    t0 = int(t[has].min())
    t1 = int(t[has].max())
    if step is None:
        step = int(np.gcd.reduce(t[has] - t0)) if t1 > t0 else 180
        step = max(step, 60)
    k = np.rint((t - t0) / step).astype(np.int64)
    n = int((t1 - t0) // step) + 1
    keep = (k >= 0) & (k < n)
    out: dict = {"start": iso(t0), "step": step}
    for v, a in vals.items():
        if np.isnan(a[keep]).all():
            continue  # a variable without any value is left out
        g = np.full(n, np.nan)
        g[k[keep]] = a[keep]
        out[v] = _tenths(g)
    return out


def write_stations(
    ds: verify.Dataset, tables: dict[str, Table], station_sums: dict, out_dir: Path
) -> tuple[int, int]:
    sdir = out_dir / "stations"
    sdir.mkdir(parents=True, exist_ok=True)
    obs = ds.obs
    now = ds.now
    lo = now - STATION_DAYS * 1440
    written: set[str] = set()
    nbytes = 0
    recent = obs["time"] > lo
    active_sids = np.unique(obs["sid"][recent]).astype(np.int64)
    if len(active_sids):
        active = np.zeros(ds.n_sid, dtype=bool)
        active[active_sids] = True
        # hourly obs: the report nearest each full hour (+-30 min, METAR wins ties)
        h0 = (lo // 60 + 1) * 60
        hours = np.arange(h0, now + 1, 60, dtype=np.int64)
        qs = np.repeat(active_sids, len(hours))
        qt = np.tile(hours, len(active_sids))
        idx = verify.match_nearest(obs["key"], obs["metar"], (qs << 32) | qt)
        ok = idx >= 0
        qs, qt, idx = qs[ok], qt[ok], idx[ok]
        H: Table = {"sid": qs, "time": qt}
        for v in OUT_VARS:
            H[v] = obs[verify.OBS_COL[v]][idx]

        # numpy tables sorted by sid: (run id or None, table, time column, step)
        groups: dict[str, tuple[str | None, Table, str, int | None]] = {}
        groups["obs"] = (None, H, "time", 60)
        for m in MODELS:
            T = tables.get(m)
            if T is None or len(T["sid"]) == 0:
                continue
            groups[f"past:{m}"] = (None, _past_rows(T, active, lo, now), "valid", None)
            latest_run = int(T["run"].max())
            F = _subset(T, np.flatnonzero((T["run"] == latest_run) & active[T["sid"]]))
            groups[f"future:{m}"] = (
                run_id(latest_run),
                _sorted_by(F, "sid", "lead"),
                "valid",
                None,
            )
        bounds = {g: _bounds(S["sid"], active_sids) for g, (_, S, _, _) in groups.items()}

        for i, s in enumerate(active_sids):
            d = _station_doc(ds, int(s), station_sums)
            for g, (rid, S, tcol, step) in groups.items():
                sa, sb = bounds[g]
                a, b = int(sa[i]), int(sb[i])
                if b <= a:
                    continue
                block = _grid(S[tcol], S, slice(a, b), step)
                if block is None:
                    continue
                if g == "obs":
                    d["obs"] = block
                    continue
                kind, m = g.split(":")
                d[kind][m] = {"run": rid, **block} if rid else block
            fname = _safe_name(d["id"])
            if fname is None:
                continue
            nbytes += _dump(d, sdir / f"{fname}.json")
            written.add(f"{fname}.json")
    for p in sdir.glob("*.json"):  # stations that went quiet
        if p.name not in written:
            p.unlink()
    return len(written), nbytes


def _station_doc(ds: verify.Dataset, s: int, station_sums: dict) -> dict:
    row = ds.stations.iloc[s]
    d = {
        "id": ds.station_ids[s],
        "name": row["name"],
        "country": row["country"],
        "lat": _r(row["lat"], 3),
        "lon": _r(row["lon"], 3),
        "elev_m": _r(row["elev_m"], 0),
        "scale": SCALE,
        "obs": {},
        "past": {},
        "future": {},
        "scores": {},
    }
    for w, by_model in station_sums.items():
        d["scores"][w] = {
            m: {
                v: list(verify.stat(S.n[s], S.s[s], S.sa[s], S.ss[s], nd=1).values())
                for v, S in by_var.items()
                if v in OUT_VARS
            }
            for m, by_var in by_model.items()
        }
    return d


def _safe_name(s: str) -> str | None:
    return s if s and s.isascii() and s.isalnum() else None


# ------------------------------------------------------------------- build

_FAILED = object()


def _optional(errors: dict, step: str, fn, *args):
    """Run an optional build step; on failure log it, note it in ``errors`` and go on."""
    try:
        return fn(*args)
    except Exception as exc:  # never let an optional page break the main site
        log.exception("optional step %s failed", step)
        errors.setdefault(step, f"{type(exc).__name__}: {exc}"[:300])
        return _FAILED


def _new_comparer(ds: verify.Dataset) -> compare.Comparer:
    return compare.Comparer(ds.obs, ds.region, ds.now)


def build(data_dir: Path, out_dir: Path, now: datetime | None = None) -> dict:
    """Verify, correct and write all site JSON. Returns a small summary."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    peak_mb: dict[str, int] = {}

    def lap(name: str, start: float) -> float:
        t = time.perf_counter()
        timings[name] = round(t - start, 2)
        peak_mb[name] = _peak_rss_mb()
        return t

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.replace(second=0, microsecond=0)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = t0
    ds = verify.load(data_dir, now)
    t = lap("load", t)
    for model in ds.fc:
        verify.match_table(ds.fc[model], ds.obs)
    t = lap("match", t)
    raw_pairs = sum(int((T["oi"] >= 0).sum()) for T in ds.fc.values())
    fcst_rows = {m: int(len(T["sid"])) for m, T in ds.fc.items()}
    keep_from = ds.now - max(verify.WINDOWS.values()) * 1440
    scorer = verify.Scorer(ds.obs, ds.region, ds.now)
    optional_errors: dict[str, str] = {}
    comparer = _optional(optional_errors, "compare", _new_comparer, ds)
    if comparer is _FAILED:
        comparer = None

    def on_table(m: str, T: Table) -> None:
        scorer.add(m, T)
        nonlocal comparer
        if (
            comparer is not None
            and _optional(optional_errors, "compare", comparer.add, m, T) is _FAILED
        ):
            comparer = None  # free what it collected; the main build goes on

    # scores are accumulated per model on complete tables; afterwards only the last
    # week (+ margin) is kept for latest.json and the station files
    trim_from = ds.now - (STATION_DAYS + 1) * 1440
    tables, applied = correct.build_models(
        ds.fc, ds.obs, keep_from, on_table, trim_from, region=ds.region
    )
    # build_models consumed ds.fc: the raw history is freed
    t = lap("correct+scores", t)
    # head-to-head comparisons; done now so the collected pairs are freed early
    (out_dir / "compare.json").unlink(missing_ok=True)  # never leave a stale one
    if comparer is not None:
        res = _optional(optional_errors, "compare", comparer.result, list(tables))
        comparer = None
        if isinstance(res, dict):
            _optional(optional_errors, "compare", _dump, res, out_dir / "compare.json")
        del res
    t = lap("compare", t)
    sc = scorer.result()
    latest = build_latest(ds, tables, sc.station)
    t = lap("latest", t)
    n_files, st_bytes = write_stations(ds, tables, sc.station, out_dir)
    t = lap("stations", t)

    latest_run = {m: run_id(int(T["run"].max())) for m, T in tables.items() if len(T["run"])}
    meta = {
        "generated": iso(ds.now),
        "first_obs": iso(ds.first_obs),
        "last_obs": iso(int(ds.obs["time"].max())) if len(ds.obs["time"]) else None,
        "obs_count": int(ds.obs_count_total),
        "station_count": int(len(np.unique(ds.obs["sid"]))),
        "stations_listed": int(len(ds.stations)),
        "station_files": n_files,
        "latest_count": len(latest),
        "runs": {m: [run_id(r) for r in ds.runs.get(m, [])] for m in verify.RAW_MODELS},
        "latest_run": {m: latest_run[m] for m in MODELS if m in latest_run},
        "models": [m for m in MODELS if m in tables] or list(MODELS),
        "pairs": int(raw_pairs),
        "status": "ok" if raw_pairs else "collecting",
        "correction": {
            "window_days": correct.WINDOW_DAYS,
            "min_pairs": correct.MIN_PAIRS,
            "applied_fraction": {
                m: {v: round(f, 3) for v, f in a.items()} for m, a in applied.items()
            },
        },
    }
    _dump(sc.scores, out_dir / "scores.json")
    _dump(latest, out_dir / "latest.json")
    meta["optional_errors"] = optional_errors
    _dump(meta, out_dir / "meta.json")
    lap("total", t0)
    build_info = {
        "seconds": timings.get("total"),
        "timings": timings,
        "peak_rss_mb": max(peak_mb.values(), default=None),
        "errors": optional_errors,
    }
    args = (data_dir, out_dir, now, meta, build_info)
    if _optional(optional_errors, "health", health.build, *args) is _FAILED:
        _dump(meta, out_dir / "meta.json")  # meta["optional_errors"] now names it
    summary = {
        "obs": int(len(ds.obs["sid"])),
        "fcst_rows": fcst_rows,
        "pairs": raw_pairs,
        "latest": len(latest),
        "station_files": n_files,
        "station_bytes": st_bytes,
        "timings": timings,
        "peak_rss_mb": peak_mb,
    }
    log.info("sitegen: %s", summary)
    return summary
