"""Per-airport forecast files: "the best forecast for your airport" (docs/FORECAST_API.md).

``build(data_dir, out_dir, now)`` writes ``forecast/<ID>.json`` for every station
with a recent observation (same rule as ``latest.json``) and ``forecast/index.json``
into ``out_dir``. Run it after ``stationtruth build-site``::

    python -m stationtruth.forecastgen --data-dir data --out site/data

Each file holds

* ``blend``: the blend forecast (docs/DATA.md, "Bias correction") of the station's
  latest blend run, **hourly** from the current hour to the end of the run
  (``run + 72 h``), linearly interpolated from the 3-hourly steps (wind direction
  along the shorter arc). Derived hourly fields: ``rh`` (relative humidity, %, Magnus
  formula over water) and ``feels`` (wind chill below 10 °C with wind > 4.8 km/h,
  heat index from 26.7 °C with RH >= 40 %, else the temperature). ``tcc`` is the
  mean of the raw cloud cover of the blend's component runs (cloud cover is not
  verified or corrected: METARs do not report it as a percentage).
* an **80 % uncertainty band** ``<var>_lo`` / ``<var>_hi`` for t, td, wspd, gust and
  mslp, see below.
* ``models``: every raw and bias-corrected model's latest run, 3-hourly, for
  comparison.
* ``skill``: the station's own verification over the last 14 days and a one-line
  summary ("blend error here ...").
* ``obs``: the latest observation.

Uncertainty band
----------------
For each variable and lead bucket (0-12, 13-24, ..., 61-72 h) the blend errors
``e = forecast - observed`` of the last ``UNC_DAYS`` (14) days are collected
(verified exactly as in docs/DATA.md, so ``gust`` is the gusty-hours sample and
``mslp`` needs a reported SLP). With the empirical quantiles ``q10``, ``q90`` of e
the band for the observed value is ``[forecast - q90, forecast - q10]``: in 80 % of
the past cases the observation fell inside it. The quantiles come from the
station's own pairs when it has at least ``MIN_SAMPLE`` (30) in that bucket, else
from its region's pairs (regions as in ``scores.json``), else from all stations
(no minimum). Between bucket centres the band offsets are interpolated linearly
in lead time. The band only covers the blend's typical error at this station; it
is not an ensemble spread and does not widen on unusually uncertain days.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import correct, sitegen, verify
from .regions import REGIONS
from .verify import NAT, Table, iso, run_id

log = logging.getLogger(__name__)

HORIZON_H = 72
#: model steps are 3-hourly; dense per-station arrays hold leads 0, 3, ..., 72
STEP_H = 3
N_STEPS = HORIZON_H // STEP_H + 1
#: days of verified blend pairs behind the band and the skill summary
UNC_DAYS = 14
#: data loaded: the band window + the correction's own 14-day window + leads
LOAD_DAYS = UNC_DAYS + correct.WINDOW_DAYS + 4
BAND_Q = (0.1, 0.9)
#: minimum pairs for a station (then region) quantile estimate
MIN_SAMPLE = 30
#: lead buckets (inclusive upper edges, hours)
LEAD_EDGES = (12, 24, 36, 48, 60, 72)
BAND_VARS: tuple[str, ...] = ("t", "td", "wspd", "gust", "mslp")
SERIES_VARS: tuple[str, ...] = ("t", "td", "wspd", "wdir", "gust", "mslp", "tcc")
SKILL_VARS: tuple[str, ...] = ("t", "td", "wspd", "gust", "mslp", "wdir")
SKILL_MIN_N = 20
SCALE = sitegen.SCALE
#: human-readable model names for the summary line
LABELS = {"ecmwf": "ECMWF", "gfs": "GFS", "ecmwf_bc": "corrected ECMWF", "gfs_bc": "corrected GFS"}


def _bucket_centres() -> np.ndarray:
    lo, out = 0, []
    for hi in LEAD_EDGES:
        out.append(float(np.mean(np.arange(lo + (-lo % STEP_H), hi + 1, STEP_H))))
        lo = hi + 1
    return np.array(out)


BUCKET_CENTRES = _bucket_centres()


def lead_bucket(lead: np.ndarray) -> np.ndarray:
    return np.minimum(np.searchsorted(np.array(LEAD_EDGES), lead, side="left"), len(LEAD_EDGES) - 1)


# ------------------------------------------------------------------ quantiles


def group_quantiles(
    group: np.ndarray, e: np.ndarray, n_groups: int, qs: Sequence[float] = BAND_Q
) -> tuple[np.ndarray, np.ndarray]:
    """(count per group, quantiles [n_groups, len(qs)]) of ``e`` by integer ``group``.

    Linear interpolation between order statistics (numpy's default method); NaN
    quantiles for empty groups.
    """
    n = np.bincount(group, minlength=n_groups)[:n_groups]
    out = np.full((n_groups, len(qs)), np.nan)
    if len(e) == 0:
        return n, out
    order = np.lexsort((e, group))
    es = e[order].astype(np.float64)
    start = np.zeros(n_groups, np.int64)
    start[1:] = np.cumsum(n)[:-1]
    has = n > 0
    for j, q in enumerate(qs):
        pos = (n[has] - 1) * q
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n[has] - 1)
        f = pos - i0
        a = es[start[has] + i0]
        b = es[start[has] + i1]
        out[has, j] = a + f * (b - a)
    return n, out


def band_offsets(
    T: Table, obs: Table, region: np.ndarray, n_sid: int, lo_t: int, hi_t: int
) -> dict[str, dict[str, np.ndarray]]:
    """Per var: error quantiles per (sid, lead bucket) with station/region/global fallback.

    Returns ``{var: {"q": [n_sid, nb, 2], "n": [n_sid, nb], "src": [n_sid, nb]}}``
    with src 0 = station, 1 = region, 2 = global, -1 = none. ``q`` are quantiles
    of forecast - observed over pairs valid in ``(lo_t, hi_t]``.
    """
    nb = len(LEAD_EDGES)
    nreg = len(REGIONS) + 1
    reg = region.astype(np.int64) + 1 if len(region) else np.zeros(n_sid, np.int64)
    out = {}
    for var in BAND_VARS:
        rows, e = verify.pair_errors(T, obs, var)
        keep = (T["valid"][rows] > lo_t) & (T["valid"][rows] <= hi_t)
        rows, e = rows[keep], e[keep]
        b = lead_bucket(T["lead"][rows])
        sid = T["sid"][rows].astype(np.int64)
        n_s, q_s = group_quantiles(sid * nb + b, e, n_sid * nb)
        n_r, q_r = group_quantiles(reg[sid] * nb + b, e, nreg * nb)
        n_g, q_g = group_quantiles(b, e, nb)
        n_s, q_s = n_s.reshape(n_sid, nb), q_s.reshape(n_sid, nb, -1)
        n_r, q_r = n_r.reshape(nreg, nb), q_r.reshape(nreg, nb, -1)
        q = np.full((n_sid, nb, len(BAND_Q)), np.nan)
        n = np.zeros((n_sid, nb), np.int64)
        src = np.full((n_sid, nb), -1, np.int8)
        # broadest first, narrower levels overwrite where they have enough pairs
        g_ok = n_g > 0
        q[:, g_ok] = q_g[g_ok]
        n[:, g_ok] = n_g[g_ok]
        src[:, g_ok] = 2
        rq, rn = q_r[reg[:n_sid]], n_r[reg[:n_sid]]
        r_ok = rn >= MIN_SAMPLE
        q[r_ok], n[r_ok], src[r_ok] = rq[r_ok], rn[r_ok], 1
        s_ok = n_s >= MIN_SAMPLE
        q[s_ok], n[s_ok], src[s_ok] = q_s[s_ok], n_s[s_ok], 0
        out[var] = {"q": q, "n": n, "src": src}
    return out


def interp_weights(lead_h: np.ndarray) -> np.ndarray:
    """Weights [..., n_buckets] to interpolate per-bucket values to ``lead_h``."""
    eye = np.eye(len(BUCKET_CENTRES))
    flat = np.asarray(lead_h, dtype=np.float64).ravel()
    w = np.stack([np.interp(flat, BUCKET_CENTRES, eye[k]) for k in range(len(eye))], axis=-1)
    return w.reshape(*np.shape(lead_h), len(eye))


# ------------------------------------------------------------------ dense runs


def latest_dense(T: Table, n_sid: int, variables: Sequence[str]) -> tuple[np.ndarray, dict]:
    """Each station's latest run as dense ``[n_sid, N_STEPS]`` arrays (lead / 3).

    Returns (run per sid, NAT if none; {var: array}).
    """
    run = np.full(n_sid, NAT, np.int64)
    if len(T["sid"]):
        np.maximum.at(run, T["sid"].astype(np.int64), T["run"].astype(np.int64))
    lead = T["lead"].astype(np.int64)
    rows = np.flatnonzero(
        (T["run"] == run[T["sid"]]) & (lead % STEP_H == 0) & (lead <= HORIZON_H) & (lead >= 0)
    )
    k = lead[rows] // STEP_H
    s = T["sid"][rows].astype(np.int64)
    out = {}
    for v in variables:
        a = np.full((n_sid, N_STEPS), np.nan, np.float32)
        if v in T:
            a[s, k] = T[v][rows]
        out[v] = a
    return run, out


def read_tcc(data_dir: Path, model: str, run: int, index: dict[str, int], n_sid: int):
    """Dense ``[n_sid, N_STEPS]`` raw total cloud cover of one run (NaN if missing)."""
    out = np.full((n_sid, N_STEPS), np.nan, np.float32)
    path = Path(data_dir) / f"fcst-{model}-{run_id(run)}.parquet"
    try:
        names = pq.read_schema(path).names
        if not {"station", "lead_h", "tcc"} <= set(names):
            return out
        df = pq.read_table(path, columns=["station", "lead_h", "tcc"]).to_pandas()
    except Exception as exc:  # missing/corrupt file: no cloud cover, not fatal
        log.warning("tcc: cannot read %s: %s", path, exc)
        return out
    sid = df["station"].astype(str).str.strip().str.upper().map(index)
    lead = pd.to_numeric(df["lead_h"], errors="coerce")
    tcc = pd.to_numeric(df["tcc"], errors="coerce").to_numpy(np.float32)
    ok = (
        sid.notna().to_numpy()
        & lead.notna().to_numpy()
        & (lead.to_numpy() % STEP_H == 0)
        & (lead.to_numpy() >= 0)
        & (lead.to_numpy() <= HORIZON_H)
    )
    tcc = np.where((tcc >= 0) & (tcc <= 100), tcc, np.nan)
    s = sid.to_numpy()[ok].astype(np.int64)
    k = (lead.to_numpy()[ok] // STEP_H).astype(np.int64)
    s_ok = s < n_sid
    out[s[s_ok], k[s_ok]] = tcc[ok][s_ok]
    return out


def shift_steps(a: np.ndarray, src_run: np.ndarray, dst_run: np.ndarray) -> np.ndarray:
    """Re-index dense rows from ``src_run`` leads to ``dst_run`` leads (per station)."""
    out = np.full_like(a, np.nan)
    ok = (src_run != NAT) & (dst_run != NAT)
    d = np.zeros(len(src_run), np.int64)
    d[ok] = (dst_run[ok] - src_run[ok]) // (STEP_H * 60)
    for k in range(N_STEPS):
        j = k + d  # source step for destination step k
        m = ok & (j >= 0) & (j < N_STEPS)
        out[m, k] = a[np.flatnonzero(m), j[m]]
    return out


# ------------------------------------------------------------------ hourly


def to_hourly(
    dense: np.ndarray, run: np.ndarray, start: int, n_hours: int, circular: bool = False
) -> np.ndarray:
    """3-hourly dense rows -> hourly values at ``start + i h`` (linear in time).

    NaN outside the run's 0..72 h or where a neighbouring step is missing
    (except exactly on a step).
    """
    n = dense.shape[0]
    h = start + 60 * np.arange(n_hours, dtype=np.int64)
    lead = (h[None, :] - run[:, None]) / 60.0  # hours
    lead[run == NAT] = np.nan
    inside = (lead >= 0) & (lead <= HORIZON_H)
    pos = np.where(inside, lead / STEP_H, 0.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, N_STEPS - 1)
    w = pos - i0
    rows = np.arange(n)[:, None]
    a = dense[rows, i0].astype(np.float64)
    b = dense[rows, i1].astype(np.float64)
    if circular:
        x = sitegen._interp_dir(a, b, w)
    else:
        x = a + w * (b - a)
    x = np.where(w == 0, a, x)
    x[~inside] = np.nan
    return x


def rel_humidity(t: np.ndarray, td: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from temperature and dew point (Magnus, over water)."""
    with np.errstate(invalid="ignore", over="ignore"):
        rh = 100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t))
    return np.minimum(rh, 100.0)


def feels_like(t: np.ndarray, rh: np.ndarray, wspd_kt: np.ndarray) -> np.ndarray:
    """Apparent temperature (°C): wind chill (cold, windy), heat index (hot, humid), else t."""
    with np.errstate(invalid="ignore"):
        v = wspd_kt * 1.852  # km/h
        vp = np.power(np.maximum(v, 0), 0.16)
        wc = 13.12 + 0.6215 * t - 11.37 * vp + 0.3965 * t * vp
        f = t * 9 / 5 + 32
        hi_f = (
            -42.379
            + 2.04901523 * f
            + 10.14333127 * rh
            - 0.22475541 * f * rh
            - 6.83783e-3 * f * f
            - 5.481717e-2 * rh * rh
            + 1.22874e-3 * f * f * rh
            + 8.5282e-4 * f * rh * rh
            - 1.99e-6 * f * f * rh * rh
        )
        hi = (hi_f - 32) * 5 / 9
        out = np.where((t <= 10) & (v > 4.8), np.minimum(wc, t), t)
        out = np.where((t >= 26.7) & (rh >= 40), np.maximum(hi, t), out)
    out[~np.isfinite(t)] = np.nan
    return out


# ------------------------------------------------------------------ skill


def station_skill(
    tables: dict[str, Table], obs: Table, n_sid: int, lo_t: int, hi_t: int
) -> tuple[dict[str, dict[str, verify.Sums]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Station verification over pairs valid in (lo_t, hi_t].

    Returns (model -> var -> Sums over sid, model -> paired temperature MAE sums).
    The paired sums compare the blend with each other model on the *same*
    forecasts only: pairs with equal (station, run, valid time), so models with a
    shorter history or fewer runs are not judged on a different sample. Per model:
    ``(sum |e| blend, sum |e| model, n)`` arrays over sid.
    """
    out: dict = {}
    keyed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m, T in tables.items():
        out[m] = {}
        for var in SKILL_VARS:
            rows, e = verify.pair_errors(T, obs, var)
            keep = (T["valid"][rows] > lo_t) & (T["valid"][rows] <= hi_t)
            rows, e = rows[keep], e[keep]
            out[m][var] = verify.Sums.of(T["sid"][rows].astype(np.int64), e, n_sid)
            if var == "t":
                run = T["run"][rows].astype(np.int64)
                lead = (T["valid"][rows].astype(np.int64) - run) // 60
                key = (T["sid"][rows].astype(np.int64) << 40) | (run << 8) | lead
                keyed[m] = (key, np.abs(e.astype(np.float64)), T["sid"][rows].astype(np.int64))
    paired: dict = {}
    if "blend" in keyed:
        kb, ab, _ = keyed["blend"]
        for m, (km, am, sm) in keyed.items():
            if m == "blend":
                continue
            _, ib, im = np.intersect1d(kb, km, assume_unique=False, return_indices=True)
            sid = sm[im]
            paired[m] = (
                np.bincount(sid, weights=ab[ib], minlength=n_sid)[:n_sid],
                np.bincount(sid, weights=am[im], minlength=n_sid)[:n_sid],
                np.bincount(sid, minlength=n_sid)[:n_sid],
            )
    return out, paired


def skill_summary(skill: dict, paired: dict, s: int) -> dict | None:
    """Headline: blend temperature MAE here and how much better it is than each model.

    ``better_pct[m]`` = 100 * (1 - MAE_blend / MAE_m) on the forecasts both made
    (see :func:`station_skill`), only with at least SKILL_MIN_N common pairs.
    """
    bl = skill.get("blend", {}).get("t")
    if bl is None or bl.n[s] < SKILL_MIN_N:
        return None
    mae = bl.sa[s] / bl.n[s]
    vs, common = {}, {}
    for m, (sab, sam, n) in paired.items():
        if n[s] < SKILL_MIN_N or sam[s] <= 0:
            continue
        vs[m] = round(100 * (1 - sab[s] / sam[s])) + 0
        common[m] = int(n[s])
    parts = []
    for m in ("ecmwf", "gfs"):
        if m in vs:
            p = vs[m]
            name = LABELS.get(m, m)
            parts.append(
                f"{abs(p)}% {'better' if p > 0 else 'worse'} than {name}"
                if p
                else f"as good as {name}"
            )
    text = (
        f"Blend temperature error here: {mae:.1f} °C typical "
        f"(mean absolute error, last {UNC_DAYS} days)"
    )
    text += f", {' and '.join(parts)}." if parts else "."
    return {
        "var": "t",
        "n": int(bl.n[s]),
        "mae": round(mae, 2),
        "better_pct": vs,
        "common_n": common,
        "text": text,
    }


# ------------------------------------------------------------------ encode


def block(start: int, step: int, arrays: dict[str, np.ndarray], **head) -> dict | None:
    """Grid block in the station-file encoding (docs/DATA.md): delta-coded tenths."""
    out: dict = {**head, "start": iso(start), "step": step}
    n = 0
    for v, a in arrays.items():
        if a is None or not np.isfinite(a).any():
            continue  # a variable without any value is left out
        last = int(np.flatnonzero(np.isfinite(a)).max())
        n = max(n, last + 1)
        out[v] = a
    if n == 0:
        return None
    for v in list(out):
        if isinstance(out[v], np.ndarray):
            out[v] = sitegen._tenths(out[v][:n])
    return out


def decode_block(blk: dict, scale: int = SCALE) -> dict[str, list]:
    """Inverse of :func:`block` (for tests and API users): var -> floats/None."""
    out = {}
    for k, a in blk.items():
        if not isinstance(a, list):
            continue
        acc, vals = 0, []
        for x in a:
            if x is None:
                vals.append(None)
            else:
                acc += x
                vals.append(acc / scale)
        out[k] = vals
    return out


def _sums_list(S: verify.Sums, s: int) -> list | None:
    if S.n[s] == 0:
        return None
    return list(verify.stat(S.n[s], S.s[s], S.sa[s], S.ss[s], nd=2).values())


# ------------------------------------------------------------------ build


def _floor_hour(m: int) -> int:
    return m // 60 * 60


def build(data_dir: Path, out_dir: Path, now: datetime | None = None) -> dict:
    """Write ``forecast/<ID>.json`` and ``forecast/index.json``; returns a summary."""
    t0 = time.perf_counter()
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.replace(second=0, microsecond=0)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    fdir = out_dir / "forecast"
    fdir.mkdir(parents=True, exist_ok=True)

    ds = verify.load(data_dir, now, horizon_days=LOAD_DAYS)
    now_m = ds.now
    obs = ds.obs
    n_sid = ds.n_sid
    runs_raw = {m: list(r) for m, r in ds.runs.items()}
    for T in ds.fc.values():
        verify.match_table(T, obs)
    lo_t = now_m - UNC_DAYS * 1440
    keep_from = lo_t - (HORIZON_H + 12) * 60
    tables, _ = correct.build_models(ds.fc, obs, keep_from=keep_from, region=ds.region)

    # stations with a recent observation (same rule as latest.json)
    written: list[dict] = []
    nbytes = 0
    active = np.zeros(0, np.int64)
    last_obs: dict[int, int] = {}
    if len(obs["sid"]):
        sid = obs["sid"]
        last = np.flatnonzero(np.r_[sid[1:] != sid[:-1], True])
        last = last[(now_m - obs["time"][last]) <= sitegen.LATEST_MAX_AGE_MIN]
        active = sid[last].astype(np.int64)
        last_obs = dict(zip(active.tolist(), last.tolist(), strict=True))

    blend = tables.get("blend")
    start = _floor_hour(now_m)
    n_hours = HORIZON_H + 1
    hourly: dict[str, np.ndarray] = {}
    band: dict = {}
    blend_run = np.full(n_sid, NAT, np.int64)
    if blend is not None and len(active):
        blend_run, bd = latest_dense(blend, n_sid, SERIES_VARS)
        # blend cloud cover: mean of the raw tcc of each component run
        index = {s: i for i, s in enumerate(ds.station_ids)}
        tcc_sum = np.zeros((n_sid, N_STEPS))
        tcc_n = np.zeros((n_sid, N_STEPS))
        cache: dict[tuple[str, int], np.ndarray] = {}
        for R in np.unique(blend_run[active][blend_run[active] != NAT]).tolist():
            st = active[blend_run[active] == R]
            for m, runs in runs_raw.items():
                ok = [r for r in runs if R - correct.BLEND_MAX_AGE_H * 60 <= r <= R]
                if not ok:
                    continue
                src = ok[-1]
                if (m, src) not in cache:
                    cache[(m, src)] = read_tcc(data_dir, m, src, index, n_sid)
                a = shift_steps(
                    cache[(m, src)][st],
                    np.full(len(st), src, np.int64),
                    np.full(len(st), R, np.int64),
                )
                has = np.isfinite(a)
                tcc_sum[st] += np.where(has, a, 0)
                tcc_n[st] += has
        with np.errstate(invalid="ignore"):
            bd["tcc"] = np.where(tcc_n > 0, tcc_sum / np.maximum(tcc_n, 1), np.nan)
        br = blend_run[active]
        for v in SERIES_VARS:
            hourly[v] = to_hourly(bd[v][active], br, start, n_hours, circular=v == "wdir")
        hourly["rh"] = rel_humidity(hourly["t"], hourly["td"])
        hourly["feels"] = feels_like(hourly["t"], hourly["rh"], hourly["wspd"])
        band = band_offsets(blend, obs, ds.region, n_sid, lo_t, now_m)
        h = start + 60 * np.arange(n_hours)
        lead_h = (h[None, :] - br[:, None]) / 60.0
        lead_h[br == NAT] = 0
        W = interp_weights(lead_h)  # [n_active, n_hours, nb]
        for v in BAND_VARS:
            q = band[v]["q"][active]  # [n_active, nb, 2]
            off_lo = np.einsum("shb,sb->sh", W, -q[:, :, 1])  # obs >= fc - q90
            off_hi = np.einsum("shb,sb->sh", W, -q[:, :, 0])
            hourly[f"{v}_lo"] = hourly[v] + off_lo
            hourly[f"{v}_hi"] = hourly[v] + off_hi
            if v in ("wspd", "gust"):
                hourly[f"{v}_lo"] = np.maximum(hourly[f"{v}_lo"], 0)

    # individual models (raw and corrected), latest run per station, 3-hourly
    model_dense: dict[str, tuple[np.ndarray, dict]] = {}
    raw_tcc: dict[str, np.ndarray] = {}
    index = {s: i for i, s in enumerate(ds.station_ids)}
    for m, T in tables.items():
        if m == "blend":
            continue
        run, d = latest_dense(T, n_sid, SERIES_VARS)
        raw = m.removesuffix("_bc")
        if raw not in raw_tcc and raw in runs_raw:
            # raw cloud cover of each station's latest raw run (corrected = same run)
            rruns = np.unique(run[active][run[active] != NAT]).tolist() if len(active) else []
            acc = np.full((n_sid, N_STEPS), np.nan, np.float32)
            for r in rruns:
                st = active[run[active] == r]
                acc[st] = read_tcc(data_dir, raw, r, index, n_sid)[st]
            raw_tcc[raw] = acc
        if raw in raw_tcc and m == raw:  # cloud cover is not corrected: raw blocks only
            d["tcc"] = raw_tcc[raw]
        model_dense[m] = (run, d)

    odd_empty = {
        m: ~np.any([np.isfinite(d[v][:, 1::2]).any(axis=1) for v in SERIES_VARS], axis=0)
        for m, (_, d) in model_dense.items()
    }
    skill, paired = station_skill(tables, obs, n_sid, lo_t, now_m)
    st = ds.stations
    obs_cols = {v: verify.OBS_COL[v] for v in ("t", "td", "wspd", "wdir", "gust", "mslp")}
    for i, s in enumerate(active.tolist()):
        fname = sitegen._safe_name(ds.station_ids[s])
        if fname is None:
            continue
        row = st.iloc[s]
        doc: dict = {
            "id": ds.station_ids[s],
            "name": row["name"],
            "country": row["country"],
            "lat": sitegen._r(row["lat"], 3),
            "lon": sitegen._r(row["lon"], 3),
            "elev_m": sitegen._r(row["elev_m"], 0),
            "generated": iso(now_m),
            "scale": SCALE,
        }
        oi = last_obs[s]
        doc["obs"] = {
            "time": iso(int(obs["time"][oi])),
            **{v: sitegen._r(obs[c][oi]) for v, c in obs_cols.items()},
        }
        if hourly and blend_run[s] != NAT:
            arrays = {k: a[i] for k, a in hourly.items()}
            blk = block(start, 60, arrays, run=run_id(int(blend_run[s])))
            if blk is not None:
                doc["blend"] = blk
                doc["band"] = _band_doc(band, s)
        models: dict = {}
        for m, (run, d) in model_dense.items():
            if run[s] == NAT:
                continue
            # 6-hourly models (no value at odd 3-h slots) get a 6-hourly block
            every = 2 if odd_empty[m][s] else 1
            k0 = max(0, (start - int(run[s])) // (STEP_H * 60))  # step at or before now
            k0 -= k0 % every
            blk = block(
                int(run[s]) + k0 * STEP_H * 60,
                every * STEP_H * 60,
                {v: d[v][s, k0::every] for v in SERIES_VARS},
                run=run_id(int(run[s])),
            )
            if blk is not None:
                models[m] = blk
        if "blend" not in doc and not models:
            continue
        doc["models"] = models
        doc["skill"] = {
            "window_days": UNC_DAYS,
            "models": {
                m: {v: _sums_list(S, s) for v, S in by_var.items()} for m, by_var in skill.items()
            },
            "summary": skill_summary(skill, paired, s),
        }
        nbytes += sitegen._dump(doc, fdir / f"{fname}.json")
        written.append(
            {
                "id": doc["id"],
                "name": doc["name"],
                "country": doc["country"],
                "lat": doc["lat"],
                "lon": doc["lon"],
            }
        )
    names = {f"{w['id']}.json" for w in written} | {"index.json"}
    for p in fdir.glob("*.json"):  # stations that went quiet
        if p.name not in names:
            p.unlink()
    written.sort(key=lambda w: w["id"])
    index_doc = {
        "generated": iso(now_m),
        "count": len(written),
        "latest_run": {m: run_id(r[-1]) for m, r in runs_raw.items() if r},
        "stations": written,
    }
    idx_bytes = sitegen._dump(index_doc, fdir / "index.json")
    summary = {
        "stations": len(written),
        "bytes": nbytes,
        "index_bytes": idx_bytes,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log.info("forecastgen: %s", summary)
    return summary


SRC_NAMES = ("station", "region", "global")


def _band_doc(band: dict, s: int) -> dict:
    lo = 0
    buckets = []
    for hi in LEAD_EDGES:
        buckets.append([lo, hi])
        lo = hi + 1
    out: dict = {
        "q": list(BAND_Q),
        "window_days": UNC_DAYS,
        "min_sample": MIN_SAMPLE,
        "buckets": buckets,
    }
    for v, b in band.items():
        q = b["q"][s]
        out[v] = {
            "src": [SRC_NAMES[k] if k >= 0 else None for k in b["src"][s].tolist()],
            "n": [int(x) for x in b["n"][s].tolist()],
            # offsets added to the forecast: [forecast + lo, forecast + hi]
            "lo": [sitegen._r(-x) for x in q[:, 1].tolist()],
            "hi": [sitegen._r(-x) for x in q[:, 0].tolist()],
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write per-airport forecast JSON (forecast/<ID>.json)."
    )
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--now", help="UTC time to build for, e.g. 2026-10-12T18:00 (default: now)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    now = None
    if a.now:
        now = datetime.fromisoformat(a.now.replace("Z", "+00:00"))
    summary = build(Path(a.data_dir), Path(a.out), now)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
