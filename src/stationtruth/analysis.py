"""Hourly gridded analysis of 2 m temperature and 10 m wind speed ("the real weather map").

Method, in plain words:

1. **Background.** The latest ECMWF open-data run that covers the analysis hour.
   Only the needed GRIB messages are downloaded (``2t``, ``10u``, ``10v`` at the
   two 3-hourly steps around the hour, plus the model orography ``z`` from step
   0; roughly 7 MB in all) and interpolated linearly in time.
2. **Observations.** For every station, the METAR closest to the analysis hour
   within +-30 minutes (routine reports win ties), as in verification.
3. **Innovations.** ``d = obs - background`` with the background interpolated
   bilinearly to the station. For temperature the observation is first moved to
   the model's orography height with a standard lapse rate (6.5 K/km), so a
   valley airport under a smoothed model mountain is compared like for like.
4. **Quality control.** Stations more than ``max_dz`` metres above or below the
   model orography are not used (the model cannot represent them), then
   innovations further than 4 robust standard deviations (1.4826 x MAD) from
   the median innovation are rejected as gross errors.
5. **Optimal interpolation.** Background errors are assumed to be correlated
   as ``exp(-r / L)`` (``r`` great-circle distance) with variance
   ``sigma_b**2``; observation errors (instrument + representativeness) are
   uncorrelated with variance ``sigma_o**2``. The weights solve
   ``(B + R) alpha = d`` once for all stations; the increment at any point is
   ``sum_k sigma_b**2 exp(-r_k / L) alpha_k``, evaluated only within
   ``cutoff`` (5 L) of a station. Far from stations it is exactly zero, so the
   analysis equals the model there.
6. **Honest validation.** Every station is withheld in turn together with all
   others within 10 km, the analysis is recomputed without them (closed form,
   no refit needed) and compared with what the station reported. These
   withheld-station errors are compared with the background's errors at the
   same stations.

Output (``--out``): Web-Mercator PNG images for Leaflet ``L.imageOverlay``
over ``[[-85, -180], [85, 180]]`` and ``analysis.json`` with metadata,
validation scores, colour scales and a 1 deg grid of values for read-outs.

Run as ``python -m stationtruth.analysis --data-dir data --out site/data/analysis``.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import http
from .grib import KT_PER_MPS, Grid, bilinear_weights, decode_message, download_messages
from .models import IndexEntry, parse_jsonl_index

EARTH_R_KM = 6371.0
LAT_MAX = 85.0  # Leaflet image bounds [[-LAT_MAX, -180], [LAT_MAX, 180]]
LAPSE_K_PER_M = 0.0065
G = 9.80665
MATCH_TOL_MIN = 30
CLUSTER_KM = 10.0  # stations withheld together in validation

# Colour scales shared with the site (MetarParse Live temperature and wind layers,
# StationTruth's diverging error scale). Values in the variable's unit.
TEMP_STOPS = [
    [-40, "#3d0d6b"],
    [-25, "#3b3fbf"],
    [-10, "#3f8fd9"],
    [0, "#8fd0ef"],
    [10, "#bfe39a"],
    [20, "#f7cf47"],
    [30, "#f2782f"],
    [40, "#cc2028"],
    [48, "#7a0b2b"],
]
WIND_STOPS = [
    [0, "#b7c4d1"],
    [5, "#8ec3e6"],
    [10, "#3f97d6"],
    [17, "#2a9d6c"],
    [25, "#e9b11c"],
    [34, "#ea5b25"],
    [48, "#c3123f"],
    [64, "#6d1a7a"],
]
# common.js DIV: x in [-1, 1] -> rgb
DIV_STOPS = [
    [-1.0, "#2166ac"],
    [-0.6, "#4393c3"],
    [-0.3, "#92c5de"],
    [-0.1, "#d1e5f0"],
    [0.0, "#ececec"],
    [0.1, "#fddbc7"],
    [0.3, "#f4a582"],
    [0.6, "#d6604d"],
    [1.0, "#b2182b"],
]


@dataclass(frozen=True)
class OIConfig:
    corr_length_km: float
    sigma_b: float  # background error standard deviation
    sigma_o: float  # observation (incl. representativeness) error standard deviation
    cutoff_lengths: float = 5.0
    qc_sd: float = 4.0
    max_dz_m: float = 500.0

    @property
    def cutoff_km(self) -> float:
        return self.corr_length_km * self.cutoff_lengths


@dataclass(frozen=True)
class VarSpec:
    name: str  # contract column: t / wspd
    label: str
    unit: str
    cfg: OIConfig
    stops: list
    inc_range: float  # increment colour scale saturates at +-inc_range
    inc_threshold: float  # |increment| below this is transparent
    lapse: bool  # move observations to model orography height
    valid_range: tuple[float, float]
    floor: float | None = None  # analysis clipped below (wind speed >= 0)


VARS: dict[str, VarSpec] = {
    "t": VarSpec(
        "t",
        "2 m temperature",
        "°C",
        OIConfig(corr_length_km=120.0, sigma_b=1.8, sigma_o=1.0),
        TEMP_STOPS,
        inc_range=5.0,
        inc_threshold=0.2,
        lapse=True,
        valid_range=(-90.0, 65.0),
    ),
    "wspd": VarSpec(
        "wspd",
        "10 m wind speed",
        "kt",
        OIConfig(corr_length_km=80.0, sigma_b=4.0, sigma_o=3.5),
        WIND_STOPS,
        inc_range=10.0,
        inc_threshold=0.5,
        lapse=False,
        valid_range=(0.0, 150.0),
        floor=0.0,
    ),
}


# --------------------------------------------------------------------------- grids


@dataclass
class LatLon:
    """Global regular grid: ``data[j, i]`` at ``lat0 + j*d``, ``-180 + i*d`` (lat ascending)."""

    data: np.ndarray
    lat0: float
    d: float

    @property
    def nlat(self) -> int:
        return self.data.shape[0]

    @property
    def nlon(self) -> int:
        return self.data.shape[1]

    @property
    def lats(self) -> np.ndarray:
        return self.lat0 + self.d * np.arange(self.nlat)

    @property
    def lons(self) -> np.ndarray:
        return -180.0 + self.d * np.arange(self.nlon)


def regularize(field: np.ndarray, grid: Grid) -> LatLon:
    """Reorder a decoded global GRIB field to latitude ascending, longitude from -180."""
    if not grid.is_global_lon:
        raise ValueError("analysis needs a global grid")
    if abs(grid.di - grid.dj) > 1e-6:
        raise ValueError("analysis needs equal lat/lon spacing")
    data = field if grid.j_positive else field[::-1]
    lat_first = grid.lat0 if grid.j_positive else grid.lat0 - grid.dj * (grid.nj - 1)
    if grid.i_negative:
        data = data[:, ::-1]
        lon_first = grid.lon0 - grid.di * (grid.ni - 1)
    else:
        lon_first = grid.lon0
    shift = int(round(((lon_first + 180.0) % 360.0) / grid.di)) % grid.ni
    data = np.roll(data, shift, axis=1)
    return LatLon(np.ascontiguousarray(data, dtype=np.float64), float(lat_first), grid.di)


def to_grib_grid(f: LatLon) -> Grid:
    return Grid(f.nlon, f.nlat, f.lat0, -180.0, f.d, f.d, False, True)


def at_points(f: LatLon, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a LatLon field to points (reuses grib.bilinear_weights)."""
    return bilinear_weights(to_grib_grid(f), lat, lon).apply(f.data)


# --------------------------------------------------------------------------- background


@dataclass
class Background:
    run: datetime
    steps: tuple[int, ...]
    weights: tuple[float, ...]
    fields: dict[str, LatLon]  # t (degC), wspd (kt), orog (m)
    bytes: int
    stream: str  # ECMWF open-data stream (oper, or scda for old 06/18Z runs)


def ecmwf_stem(base: str, run: datetime, step: int, stream: str = "oper") -> str:
    return f"{base}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{stream}/{run:%Y%m%d%H}0000-{step}h-{stream}-fc"


def select_params(entries: list[IndexEntry], params: Sequence[str]) -> dict[str, IndexEntry]:
    out: dict[str, IndexEntry] = {}
    for e in entries:
        a = e.attrs
        if a.get("levtype") != "sfc" or "levelist" in a or a.get("type", "fc") != "fc":
            continue
        p = a.get("param")
        if p in params and p not in out:
            out[p] = e
    return out


def bracket_steps(lead_h: float, step_every: int = 3) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Model steps around ``lead_h`` and their linear time-interpolation weights."""
    s0 = int(lead_h // step_every) * step_every
    frac = (lead_h - s0) / step_every
    if frac < 1e-9:
        return (s0,), (1.0,)
    return (s0, s0 + step_every), (1.0 - frac, frac)


def find_background(
    valid: datetime,
    *,
    base: str = "https://data.ecmwf.int/forecasts",
    lookback_h: int = 36,
    max_lead: int = 90,
    exists: Callable[[str], bool] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[datetime, str, tuple[int, ...], tuple[float, ...]] | None:
    """Newest run (00/06/12/18 UTC) whose steps around ``valid`` are published."""
    exists = exists or (lambda u: http.exists(u, retries=1, timeout=30))
    t = valid.replace(minute=0, second=0, microsecond=0)
    t -= timedelta(hours=t.hour % 6)
    stop = valid - timedelta(hours=lookback_h)
    while t >= stop:
        lead = (valid - t).total_seconds() / 3600.0
        if 0 <= lead <= max_lead:
            steps, weights = bracket_steps(lead)
            for stream in ("oper", "scda"):
                if all(exists(ecmwf_stem(base, t, s, stream) + ".index") for s in steps):
                    return t, stream, steps, weights
            log(f"  ECMWF {t:%Y%m%d%H} not available for +{lead:g} h")
        t -= timedelta(hours=6)
    return None


def fetch_background(
    valid: datetime,
    *,
    base: str = "https://data.ecmwf.int/forecasts",
    log: Callable[[str], None] = print,
) -> Background:
    found = find_background(valid, base=base, log=log)
    if found is None:
        raise RuntimeError(f"no ECMWF run covers {valid:%Y-%m-%d %H:%M}Z")
    run, stream, steps, weights = found
    jobs = [(s, ("2t", "10u", "10v")) for s in steps]
    if 0 in steps:
        jobs[steps.index(0)] = (0, ("2t", "10u", "10v", "z"))
    else:
        jobs.append((0, ("z",)))

    def fetch(job):
        step, params = job
        stem = ecmwf_stem(base, run, step, stream)
        idx = http.get(stem + ".index", timeout=60).decode("utf-8", "replace")
        sel = select_params(parse_jsonl_index(idx), params)
        missing = set(params) - set(sel)
        if missing:
            raise RuntimeError(f"{stem}: missing {sorted(missing)}")
        msgs, _, nbytes = download_messages(stem + ".grib2", sel, max_gap=64 * 1024)
        return step, msgs, nbytes + len(idx)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(fetch, jobs))
    nbytes = 0
    decoded: dict[tuple[int, str], LatLon] = {}
    for step, msgs, nb in results:
        nbytes += nb
        for p, msg in msgs.items():
            field, grid, _ = decode_message(msg)
            decoded[(step, p)] = regularize(field, grid)

    def blend(param: str) -> LatLon:
        f0 = decoded[(steps[0], param)]
        data = sum(w * decoded[(s, param)].data for s, w in zip(steps, weights, strict=True))
        return LatLon(np.asarray(data, dtype=np.float64), f0.lat0, f0.d)

    t = blend("2t")
    t.data -= 273.15
    u, v = blend("10u"), blend("10v")
    wspd = LatLon(np.hypot(u.data, v.data) * KT_PER_MPS, u.lat0, u.d)
    orog = decoded[(0, "z")]
    orog = LatLon(orog.data / G, orog.lat0, orog.d)
    log(
        f"  background ECMWF {run:%Y%m%d%H} {stream} steps {list(steps)} "
        f"weights {[round(w, 3) for w in weights]} ({nbytes / 1e6:.1f} MB)"
    )
    return Background(run, steps, weights, {"t": t, "wspd": wspd, "orog": orog}, nbytes, stream)


# --------------------------------------------------------------------------- observations


def load_obs(data_dir: Path, valid: datetime) -> pd.DataFrame:
    days = sorted({(valid + timedelta(minutes=m)).date() for m in (-60, 0, 60)})
    frames = []
    for d in days:
        p = data_dir / f"obs-{d:%Y-%m-%d}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["station", "time", "type", "t", "wspd"]))
    if not frames:
        return pd.DataFrame(columns=["station", "time", "type", "t", "wspd"])
    return pd.concat(frames, ignore_index=True)


def latest_obs_time(data_dir: Path) -> datetime | None:
    files = sorted(data_dir.glob("obs-*.parquet"))
    for p in reversed(files):
        df = pd.read_parquet(p, columns=["time"])
        if len(df):
            return df["time"].max().to_pydatetime()
    return None


def pick_valid_time(data_dir: Path) -> datetime:
    """The latest whole hour with its +-30 min window (mostly) observed.

    If the newest hour has far fewer reporting stations than the hour before
    (the hourly collection has not caught up yet), the previous hour is used.
    """
    last = latest_obs_time(data_dir)
    if last is None:
        raise RuntimeError(f"no observations in {data_dir}")
    t = last.replace(minute=0, second=0, microsecond=0)
    if last - t >= timedelta(minutes=MATCH_TOL_MIN):
        return t
    obs = load_obs(data_dir, t)

    def count(h: datetime) -> int:
        lo, hi = h - timedelta(minutes=MATCH_TOL_MIN), h + timedelta(minutes=MATCH_TOL_MIN)
        m = (obs["time"] >= lo) & (obs["time"] <= hi) & obs["t"].notna()
        return int(obs.loc[m, "station"].nunique())

    prev = t - timedelta(hours=1)
    return t if count(t) >= 0.8 * count(prev) else prev


def select_obs(
    obs: pd.DataFrame, stations: pd.DataFrame, valid: datetime, var: str, valid_range=None
) -> pd.DataFrame:
    """One value per station: closest report to ``valid`` within +-30 min (METAR wins ties)."""
    vt = pd.Timestamp(valid)
    o = obs[["station", "time", "type", var]].dropna(subset=[var])
    if valid_range is not None:
        o = o[(o[var] >= valid_range[0]) & (o[var] <= valid_range[1])]
    dt = (o["time"] - vt).abs()
    o = o.assign(_dt=dt, _sp=(o["type"] != "METAR").astype(int))
    o = o[o["_dt"] <= pd.Timedelta(minutes=MATCH_TOL_MIN)]
    o = o.sort_values(["station", "_dt", "_sp", "time"]).drop_duplicates("station")
    st = stations[["station", "lat", "lon", "elev_m"]]
    out = o.merge(st, on="station", how="inner")
    out = out[np.isfinite(out["lat"]) & np.isfinite(out["lon"])]
    return (
        out[["station", "lat", "lon", "elev_m", var]]
        .rename(columns={var: "value"})
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- OI core


def unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def gc_dist_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (haversine), broadcasting."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def obs_covariance(lat: np.ndarray, lon: np.ndarray, cfg: OIConfig) -> np.ndarray:
    """``B + R`` for the observations: ``sigma_b^2 exp(-r/L) + sigma_o^2 I`` (float32).

    Distances are computed in float64 row blocks (accurate for nearby stations)
    and stored in float32, so 5,000 stations need ~100 MB per matrix.
    """
    x = unit_vectors(lat, lon)
    n = len(x)
    a = np.empty((n, n), np.float32)
    for i in range(0, n, 512):
        m = x[i : i + 512] @ x.T
        np.clip(m, -1.0, 1.0, out=m)
        np.arccos(m, out=m)
        m *= -EARTH_R_KM / cfg.corr_length_km
        np.exp(m, out=m)
        m *= cfg.sigma_b**2
        a[i : i + 512] = m
    a[np.diag_indices_from(a)] += np.float32(cfg.sigma_o**2)
    return a


def robust_qc(d: np.ndarray, k: float) -> tuple[np.ndarray, float, float]:
    """Mask of innovations within ``k`` robust sd of the median; returns (ok, median, sd)."""
    med = float(np.median(d)) if len(d) else 0.0
    sd = float(1.4826 * np.median(np.abs(d - med))) if len(d) else 0.0
    sd = max(sd, 1e-6)
    return np.abs(d - med) <= k * sd, med, sd


def solve_oi(lat, lon, d, cfg: OIConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``alpha = (B + R)^-1 d`` and ``(B + R)^-1`` for the used observations.

    ``B + R`` is well conditioned (eigenvalues >= sigma_o^2), so float32 is ample.
    """
    ainv = np.linalg.inv(obs_covariance(lat, lon, cfg))
    return (ainv @ np.asarray(d, np.float32)).astype(np.float64), ainv


def withheld_residuals(
    alpha: np.ndarray, ainv: np.ndarray, lat, lon, radius_km: float = CLUSTER_KM
) -> np.ndarray:
    """obs - analysis at each station when it (and every station within ``radius_km``) is withheld.

    Block-inverse identity for OI/kriging: with ``A = B + R``, ``alpha = A^-1 d``,
    withholding the set C gives residuals ``(A^-1)_CC^-1 alpha_C``; no refit needed.
    """
    n = len(alpha)
    out = np.empty(n)
    x = unit_vectors(np.asarray(lat), np.asarray(lon))
    cos_r = np.cos(radius_km / EARTH_R_KM)
    for k in range(n):
        c = np.flatnonzero(x @ x[k] >= cos_r)
        if len(c) == 1:
            out[k] = alpha[k] / ainv[k, k]
        else:
            e = np.linalg.solve(ainv[np.ix_(c, c)], alpha[c])
            out[k] = e[np.searchsorted(c, k)]
    return out


def increment_grid(
    lats: np.ndarray, lons: np.ndarray, olat, olon, alpha, cfg: OIConfig
) -> np.ndarray:
    """``sum_k sigma_b^2 exp(-r_k/L) alpha_k`` on a global grid, within the cutoff of each obs."""
    nlat, nlon = len(lats), len(lons)
    d = float(lats[1] - lats[0])
    inc = np.zeros((nlat, nlon))
    cutoff = cfg.cutoff_km
    wj = int(np.ceil(cutoff / (np.pi * EARTH_R_KM / 180.0) / d)) + 1
    for la, lo, a in zip(np.asarray(olat), np.asarray(olon), np.asarray(alpha), strict=True):
        jc = int(round((la - lats[0]) / d))
        j0, j1 = max(jc - wj, 0), min(jc + wj + 1, nlat)
        maxabs = max(abs(lats[j0]), abs(lats[j1 - 1]))
        coslat = np.cos(np.radians(min(maxabs, 89.99)))
        wi = int(np.ceil(cutoff / (np.pi * EARTH_R_KM / 180.0 * coslat) / d)) + 1
        ic = int(round((lo + 180.0) / d))
        if 2 * wi + 1 >= nlon:
            ii = np.arange(nlon)
        else:
            ii = np.arange(ic - wi, ic + wi + 1) % nlon
        r = gc_dist_km(la, lo, lats[j0:j1, None], lons[None, ii])
        w = np.where(r <= cutoff, np.exp(-r / cfg.corr_length_km), 0.0)
        inc[j0:j1, ii] += (cfg.sigma_b**2 * a) * w
    return inc


def scores(err: np.ndarray) -> dict:
    err = err[np.isfinite(err)]
    if not len(err):
        return {"n": 0, "bias": None, "mae": None, "rmse": None}
    return {
        "n": int(len(err)),
        "bias": round(float(err.mean()), 3),
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err**2).mean())), 3),
    }


@dataclass
class VarResult:
    spec: VarSpec
    background: LatLon
    analysis: LatLon
    increment: LatLon
    stations: pd.DataFrame  # station, lat, lon, obs, bg, status, loo
    counts: dict
    validation: dict
    qc: dict


def analyse_var(
    spec: VarSpec,
    bg: LatLon,
    orog: LatLon | None,
    obs: pd.DataFrame,
    log: Callable[[str], None] = print,
) -> VarResult:
    """Run QC + OI + withheld-station validation for one variable."""
    cfg = spec.cfg
    lat = obs["lat"].to_numpy(float)
    lon = obs["lon"].to_numpy(float)
    val = obs["value"].to_numpy(float)
    bg_raw = at_points(bg, lat, lon)
    n = len(obs)
    status = np.zeros(n, dtype=np.int8)  # 0 used, 1 gross error, 2 orography, 3 outside grid
    dz = np.full(n, np.nan)
    if orog is not None:
        zm = at_points(orog, lat, lon)
        dz = obs["elev_m"].to_numpy(float) - zm
    adj = np.zeros(n)
    if spec.lapse:
        adj = np.where(np.isfinite(dz), LAPSE_K_PER_M * dz, 0.0)
    d = val + adj - bg_raw  # innovation at model orography height
    status[~np.isfinite(d)] = 3
    status[(status == 0) & np.isfinite(dz) & (np.abs(dz) > cfg.max_dz_m)] = 2
    cand = status == 0
    ok, med, sd = robust_qc(d[cand], cfg.qc_sd)
    idx = np.flatnonzero(cand)
    status[idx[~ok]] = 1
    used = status == 0
    lu, ou, du = lat[used], lon[used], d[used]
    t0 = time.monotonic()
    alpha, ainv = solve_oi(lu, ou, du, cfg)
    t_solve = time.monotonic() - t0
    resid = withheld_residuals(alpha, ainv, lu, ou)  # obs - analysis(withheld)
    del ainv
    t0 = time.monotonic()
    inc = increment_grid(bg.lats, bg.lons, lu, ou, alpha, cfg)
    t_grid = time.monotonic() - t0
    ana = bg.data + inc
    if spec.floor is not None:
        np.maximum(ana, spec.floor, out=ana)
        inc = ana - bg.data
    # in-sample analysis at stations, for the station dots: d - sigma_o^2 alpha
    fit = np.full(n, np.nan)
    fit[used] = du - cfg.sigma_o**2 * alpha
    loo = np.full(n, np.nan)
    loo[used] = resid
    validation = {
        "method": (
            f"each station withheld together with all stations within {CLUSTER_KM:g} km; "
            "analysis recomputed without them and compared with the report"
        ),
        "withheld_radius_km": CLUSTER_KM,
        "analysis": scores(resid),  # error = analysis - obs = -resid
        "background": scores(-du),  # bg (height-adjusted) - obs
        "model_raw": scores(bg_raw[used] - val[used]),  # no height adjustment
        "analysis_in_sample": scores(-fit[used]),
    }
    validation["analysis"]["bias"] = (
        None if validation["analysis"]["bias"] is None else -validation["analysis"]["bias"]
    )
    counts = {
        "candidates": int(n),
        "used": int(used.sum()),
        "rejected_gross": int((status == 1).sum()),
        "rejected_orography": int((status == 2).sum()),
        "outside_grid": int((status == 3).sum()),
    }
    log(
        f"  {spec.name}: {counts} solve {t_solve:.1f}s grid {t_grid:.1f}s; "
        f"withheld MAE {validation['analysis']['mae']} vs bg {validation['background']['mae']}"
        f" vs raw {validation['model_raw']['mae']}"
    )
    st = pd.DataFrame(
        {
            "station": obs["station"].to_numpy(),
            "lat": lat,
            "lon": lon,
            "obs": val,
            "bg": bg_raw,
            "adj": adj,
            "status": status,
            "loo": loo,
        }
    )
    return VarResult(
        spec,
        bg,
        LatLon(ana, bg.lat0, bg.d),
        LatLon(inc, bg.lat0, bg.d),
        st,
        counts,
        validation,
        {"median_innovation": round(med, 3), "robust_sd": round(sd, 3)},
    )


# --------------------------------------------------------------------------- rendering


def mercator_axes(width: int, lat_max: float = LAT_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre longitudes (width) and latitudes (height) of a Web-Mercator image.

    The image spans lon -180..180 and lat -lat_max..lat_max; rows are equally
    spaced in Mercator y, which is how Leaflet stretches an imageOverlay.
    """
    ymax = np.log(np.tan(np.pi / 4 + np.radians(lat_max) / 2))
    height = int(round(width * ymax / np.pi))
    lon = -180.0 + (np.arange(width) + 0.5) * 360.0 / width
    y = ymax - (np.arange(height) + 0.5) * (2 * ymax / height)
    lat = np.degrees(np.arctan(np.sinh(y)))
    return lon, lat


def resample(f: LatLon, plon: np.ndarray, plat: np.ndarray) -> np.ndarray:
    """Separable bilinear resampling of a global grid onto (plat x plon)."""
    fj = np.clip((plat - f.lat0) / f.d, 0, f.nlat - 1)
    j0 = np.minimum(np.floor(fj).astype(int), f.nlat - 2)
    wj = (fj - j0)[:, None]
    rows = (1 - wj) * f.data[j0] + wj * f.data[j0 + 1]
    fi = np.mod((plon + 180.0) / f.d, f.nlon)
    i0 = np.floor(fi).astype(int) % f.nlon
    i1 = (i0 + 1) % f.nlon
    wi = (fi - np.floor(fi))[None, :]
    return (1 - wi) * rows[:, i0] + wi * rows[:, i1]


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def ramp_rgb(x: np.ndarray, stops: list) -> np.ndarray:
    """Piecewise-linear colour ramp (as in the site's JS) -> (..., 3) uint8."""
    xs = np.array([s[0] for s in stops], float)
    rgb = np.array([_hex(s[1]) for s in stops], float)
    return np.stack([np.rint(np.interp(x, xs, rgb[:, k])) for k in range(3)], -1).astype(np.uint8)


@dataclass(frozen=True)
class Palette:
    """Values quantised to ``lo + k*step`` (k = 0..n-1), drawn as PNG palette index k+1.

    Index 0 is fully transparent (missing / masked). With 0.5 degC or 0.5 kt
    steps the colour difference between neighbouring levels is invisible, and a
    palette PNG is 4x smaller than RGBA before compression.
    """

    lo: float
    step: float
    n: int  # <= 255
    stops: tuple
    scale: float = 1.0  # colour = ramp(value / scale): increments use the [-1, 1] DIV ramp
    fade_below: float = 0.0  # |value| < fade_below transparent, fading in up to 2x that

    def rgba(self) -> np.ndarray:
        vals = self.lo + self.step * np.arange(self.n)
        pal = np.zeros((self.n + 1, 4), np.uint8)
        pal[1:, :3] = ramp_rgb(vals / self.scale, list(self.stops))
        if self.fade_below > 0:
            f = np.clip((np.abs(vals) - self.fade_below) / self.fade_below + 0.25, 0, 1)
            pal[1:, 3] = np.where(np.abs(vals) < self.fade_below - 1e-9, 0, np.rint(255 * f))
        else:
            pal[1:, 3] = 255
        return pal

    def index(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        k = np.rint((np.nan_to_num(values, nan=self.lo) - self.lo) / self.step)
        idx = (np.clip(k, 0, self.n - 1) + 1).astype(np.uint8)
        bad = ~np.isfinite(values)
        if mask is not None:
            bad |= ~mask
        idx[bad] = 0
        return idx


def value_palette(spec: VarSpec, step: float = 0.5) -> Palette:
    lo, hi = spec.stops[0][0], spec.stops[-1][0]
    return Palette(lo, step, int(round((hi - lo) / step)) + 1, tuple(map(tuple, spec.stops)))


def increment_palette(spec: VarSpec) -> Palette:
    step = spec.inc_range / 50.0
    return Palette(
        -spec.inc_range,
        step,
        101,
        tuple(map(tuple, DIV_STOPS)),
        spec.inc_range,
        fade_below=spec.inc_threshold,
    )


def png_indexed(idx: np.ndarray, pal: np.ndarray, level: int = 6) -> bytes:
    """Encode an (h, w) uint8 index image + (n, 4) RGBA palette as an 8-bit palette PNG.

    Pure stdlib (zlib + struct). Every row after the first uses the 'Up' filter,
    which turns smooth fields into long runs of zeros.
    """
    h, w = idx.shape
    raw = np.empty((h, w + 1), np.uint8)
    raw[:, 0] = 2
    raw[0, 0] = 0
    raw[0, 1:] = idx[0]
    np.subtract(idx[1:], idx[:-1], out=raw[1:, 1:])  # uint8 wraps mod 256 as PNG expects

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    pal = np.asarray(pal, np.uint8)
    alpha = pal[:, 3]
    last_t = int(np.flatnonzero(alpha < 255).max()) + 1 if (alpha < 255).any() else 0
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
    out += chunk(b"PLTE", pal[:, :3].tobytes())
    if last_t:
        out += chunk(b"tRNS", alpha[:last_t].tobytes())
    out += chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
    return out + chunk(b"IEND", b"")


def render_images(res: VarResult, out: Path, width: int) -> dict[str, dict]:
    plon, plat = mercator_axes(width)
    spec = res.spec
    vp, ip = value_palette(spec), increment_palette(spec)
    info = {}
    for kind, field in (
        ("analysis", res.analysis),
        ("background", res.background),
        ("increment", res.increment),
    ):
        v = resample(field, plon, plat)
        if kind == "increment":
            data = png_indexed(ip.index(v), ip.rgba())
        else:
            data = png_indexed(vp.index(v), vp.rgba())
        del v
        name = f"{spec.name}_{kind}.png"
        (out / name).write_bytes(data)
        info[kind] = {"file": name, "bytes": len(data)}
    return info


# --------------------------------------------------------------------------- JSON


def coarse(f: LatLon, step: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = np.arange(-LAT_MAX, LAT_MAX + 1e-9, step)
    lons = np.arange(-180.0, 180.0 - 1e-9, step)
    return lats, lons, resample(f, lons, lats)


def _ints(a: np.ndarray, scale: float = 10.0) -> list:
    return [int(x) for x in np.rint(a.ravel() * scale)]


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def var_doc(res: VarResult, images: dict, grid_step: float) -> dict:
    spec = res.spec
    lats, lons, ana = coarse(res.analysis, grid_step)
    _, _, inc = coarse(res.increment, grid_step)
    st = res.stations
    stations = {
        "id": st["station"].tolist(),
        "lat": [_r(x, 3) for x in st["lat"]],
        "lon": [_r(x, 3) for x in st["lon"]],
        "obs": [_r(x) for x in st["obs"]],
        "bg": [_r(x) for x in st["bg"]],
        "adj": [_r(x, 2) for x in st["adj"]],
        "status": [int(x) for x in st["status"]],
        "loo": [_r(x, 2) for x in st["loo"]],
    }
    return {
        "label": spec.label,
        "unit": spec.unit,
        "images": images,
        "params": {
            **asdict(spec.cfg),
            "cutoff_km": spec.cfg.cutoff_km,
            "error_variance_ratio": round(spec.cfg.sigma_o**2 / spec.cfg.sigma_b**2, 3),
            "lapse_rate_k_per_km": LAPSE_K_PER_M * 1000 if spec.lapse else None,
        },
        "counts": res.counts,
        "qc": res.qc,
        "validation": res.validation,
        "scale": {
            "value": spec.stops,
            "increment": {
                "range": spec.inc_range,
                "threshold": spec.inc_threshold,
                "stops": DIV_STOPS,
            },
        },
        "grid": {
            "lat0": float(lats[0]),
            "lon0": float(lons[0]),
            "step": grid_step,
            "nlat": len(lats),
            "nlon": len(lons),
            "scale": 10,
            "order": "row-major from (lat0, lon0), latitude ascending",
            "analysis": _ints(ana),
            "increment": _ints(inc),
        },
        "stations": stations,
        "station_status": {
            "0": "used",
            "1": "rejected: gross error",
            "2": "rejected: elevation differs from model orography",
            "3": "outside model grid",
        },
    }


def _peak_rss_mb() -> int:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r / 2**20 if sys.platform == "darwin" else r / 1024)
    except Exception:  # pragma: no cover - non-Unix
        return -1


# --------------------------------------------------------------------------- driver


def run(
    data_dir: Path,
    out: Path,
    *,
    valid: datetime | None = None,
    width: int = 2048,
    grid_step: float = 1.0,
    variables: Sequence[str] = ("t", "wspd"),
    background: Background | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    t_start = time.monotonic()
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stations = pd.read_parquet(data_dir / "stations.parquet")
    valid = valid or pick_valid_time(data_dir)
    valid = valid.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log(f"analysis valid {valid:%Y-%m-%d %H:%M}Z")
    t0 = time.monotonic()
    bg = background or fetch_background(valid, log=log)
    t_fetch = time.monotonic() - t0
    obs = load_obs(data_dir, valid)
    doc = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bounds": [[-LAT_MAX, -180.0], [LAT_MAX, 180.0]],
        "projection": "EPSG:3857 (Web Mercator), rows equally spaced in Mercator y",
        "image_width": width,
        "background": {
            "model": "ecmwf",
            "label": "ECMWF IFS open data 0.25°",
            "run": bg.run.strftime("%Y%m%d%H"),
            "stream": bg.stream,
            "steps": list(bg.steps),
            "step_weights": [round(w, 4) for w in bg.weights],
            "lead_h": round((valid - bg.run).total_seconds() / 3600.0, 2),
            "downloaded_mb": round(bg.bytes / 1e6, 2),
        },
        "obs_window_min": MATCH_TOL_MIN,
        "variables": {},
    }
    for name in variables:
        spec = VARS[name]
        sel = select_obs(obs, stations, valid, name, spec.valid_range)
        if len(sel) < 10:
            log(f"  {name}: only {len(sel)} observations, skipped")
            continue
        res = analyse_var(spec, bg.fields[name], bg.fields.get("orog"), sel, log=log)
        images = render_images(res, out, width)
        doc["variables"][name] = var_doc(res, images, grid_step)
        del res
    doc["runtime"] = {
        "seconds": round(time.monotonic() - t_start, 1),
        "fetch_seconds": round(t_fetch, 1),
        "peak_rss_mb": _peak_rss_mb(),
    }
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    (out / "analysis.json").write_text(text)
    log(
        f"wrote {out}/analysis.json ({len(text) / 1e3:.0f} kB) in {doc['runtime']['seconds']} s, "
        f"peak RSS {doc['runtime']['peak_rss_mb']} MB"
    )
    return doc


def _parse_time(s: str) -> datetime:
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").to_pydatetime()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m stationtruth.analysis",
        description="Hourly 2 m temperature / 10 m wind analysis: ECMWF background + METARs (OI).",
    )
    p.add_argument("--data-dir", default="data", help="directory with stations/obs parquet")
    p.add_argument("--out", default="site/data/analysis", help="output directory")
    p.add_argument("--time", help="analysis hour (UTC, ISO); default: latest observed hour")
    p.add_argument("--width", type=int, default=2048, help="image width in pixels")
    p.add_argument("--vars", default="t,wspd", help="comma-separated: t,wspd")
    args = p.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        run(
            Path(args.data_dir),
            Path(args.out),
            valid=_parse_time(args.time) if args.time else None,
            width=args.width,
            variables=[v for v in args.vars.split(",") if v],
            log=log,
        )
    except Exception as exc:
        log(f"analysis failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
