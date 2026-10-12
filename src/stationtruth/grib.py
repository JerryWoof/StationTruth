"""Fetch selected GRIB2 messages with HTTP range requests and interpolate to stations.

Only the messages we need are downloaded: each source publishes an index
(byte offset/length per message), adjacent ranges are merged, and a modest
thread pool downloads steps in parallel. Decoding (eccodes) and the bilinear
interpolation run in the calling thread, one step at a time, so memory stays
at a few global fields.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import http
from .models import IndexEntry, ModelSource

KT_PER_MPS = 1.943844
K0 = 273.15

FCST_COLUMNS = [
    "station",
    "run",
    "valid",
    "lead_h",
    "t",
    "td",
    "wspd",
    "wdir",
    "gust",
    "mslp",
    "tcc",
    "vis_m",
]
FCST_FLOATS = ["t", "td", "wspd", "wdir", "gust", "mslp", "tcc", "vis_m"]


# --------------------------------------------------------------------------- byte ranges


@dataclass
class ByteRange:
    start: int
    end: int | None  # inclusive; None = to end of file
    members: list[tuple[str, int, int | None]]  # (var, offset within chunk, length)


def merge_ranges(selected: dict[str, IndexEntry], max_gap: int = 0) -> list[ByteRange]:
    """Merge messages whose byte ranges touch (or are within ``max_gap`` bytes)."""
    items = sorted(selected.items(), key=lambda kv: kv[1].offset)
    out: list[ByteRange] = []
    for var, e in items:
        end = None if e.length is None else e.offset + e.length - 1
        cur = out[-1] if out else None
        if cur is not None and cur.end is not None and e.offset <= cur.end + 1 + max_gap:
            cur.members.append((var, e.offset - cur.start, e.length))
            cur.end = None if end is None else max(cur.end, end)
        else:
            out.append(ByteRange(e.offset, end, [(var, 0, e.length)]))
    return out


def download_messages(
    url: str,
    selected: dict[str, IndexEntry],
    *,
    max_gap: int = 0,
    getter: Callable[[str, int, int | None], bytes] | None = None,
) -> tuple[dict[str, bytes], int, int]:
    """Download the selected messages; return ``({var: bytes}, n_requests, n_bytes)``."""
    getter = getter or (lambda u, a, b: http.get_range(u, a, b, timeout=120))
    out: dict[str, bytes] = {}
    n_bytes = 0
    ranges = merge_ranges(selected, max_gap=max_gap)
    for rng in ranges:
        blob = getter(url, rng.start, rng.end)
        n_bytes += len(blob)
        for var, off, length in rng.members:
            out[var] = blob[off:] if length is None else blob[off : off + length]
    return out, len(ranges), n_bytes


# --------------------------------------------------------------------------- decoding


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid as described by GRIB keys (first point, increments, scanning)."""

    ni: int
    nj: int
    lat0: float  # latitude of first grid point
    lon0: float  # longitude of first grid point
    di: float  # positive increment along i (degrees)
    dj: float  # positive increment along j (degrees)
    i_negative: bool  # iScansNegatively: longitude decreases along i
    j_positive: bool  # jScansPositively: latitude increases along j

    @property
    def is_global_lon(self) -> bool:
        return abs(self.ni * self.di - 360.0) < self.di * 0.5


def decode_message(msg: bytes) -> tuple[np.ndarray, Grid, dict]:
    """Decode one GRIB message into a ``(nj, ni)`` float64 array (NaN for missing)."""
    import eccodes

    h = eccodes.codes_new_from_message(msg)
    try:
        grid_type = eccodes.codes_get(h, "gridType")
        if grid_type != "regular_ll":
            raise ValueError(f"unsupported grid type {grid_type}")
        ni = eccodes.codes_get(h, "Ni")
        nj = eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        lat1 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
        lon1 = eccodes.codes_get(h, "longitudeOfLastGridPointInDegrees")
        i_neg = bool(eccodes.codes_get(h, "iScansNegatively"))
        j_pos = bool(eccodes.codes_get(h, "jScansPositively"))
        j_consec = bool(eccodes.codes_get(h, "jPointsAreConsecutive"))
        # Derive increments from the corner points when possible (more precise
        # than the millidegree-rounded increment keys); fall back to the keys.
        if ni > 1:
            span = (lon0 - lon1) if i_neg else (lon1 - lon0)
            span %= 360.0
            di = (
                span / (ni - 1)
                if span > 0
                else eccodes.codes_get(h, "iDirectionIncrementInDegrees")
            )
        else:
            di = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        dj = abs(lat1 - lat0) / (nj - 1) if nj > 1 else 1.0
        values = eccodes.codes_get_values(h).astype("float64")
        if eccodes.codes_get(h, "bitmapPresent"):
            missing = eccodes.codes_get(h, "missingValue")
            values[values == missing] = np.nan
        meta = {
            "shortName": eccodes.codes_get(h, "shortName"),
            "units": eccodes.codes_get(h, "units"),
        }
    finally:
        eccodes.codes_release(h)
    if j_consec:
        field = values.reshape(ni, nj).T
    else:
        field = values.reshape(nj, ni)
    grid = Grid(ni, nj, float(lat0), float(lon0), float(di), float(dj), i_neg, j_pos)
    return field, grid, meta


# --------------------------------------------------------------------------- interpolation


@dataclass
class Weights:
    i0: np.ndarray
    i1: np.ndarray
    j0: np.ndarray
    j1: np.ndarray
    wi: np.ndarray  # weight of i1
    wj: np.ndarray  # weight of j1
    valid: np.ndarray  # station inside grid

    def apply(self, field: np.ndarray) -> np.ndarray:
        f00 = field[self.j0, self.i0]
        f01 = field[self.j0, self.i1]
        f10 = field[self.j1, self.i0]
        f11 = field[self.j1, self.i1]
        wi, wj = self.wi, self.wj
        out = (1 - wj) * ((1 - wi) * f00 + wi * f01) + wj * ((1 - wi) * f10 + wi * f11)
        out[~self.valid] = np.nan
        return out


def bilinear_weights(grid: Grid, lat: np.ndarray, lon: np.ndarray) -> Weights:
    """Vectorised bilinear weights for points on a regular lat/lon grid.

    Longitudes are wrapped, so a 0..360 grid (GFS) and a -180..180 grid
    (ECMWF) both work with station longitudes in [-180, 180).
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    # fractional i index
    dlon = (lon - grid.lon0) if not grid.i_negative else (grid.lon0 - lon)
    fi = np.mod(dlon, 360.0) / grid.di
    # fractional j index
    dlat = (lat - grid.lat0) if grid.j_positive else (grid.lat0 - lat)
    fj = dlat / grid.dj

    valid = np.isfinite(fi) & np.isfinite(fj) & (fj >= -1e-9) & (fj <= grid.nj - 1 + 1e-9)
    fj = np.clip(fj, 0, grid.nj - 1)
    j0 = np.floor(fj).astype(np.int64)
    j0 = np.clip(j0, 0, max(grid.nj - 2, 0))
    j1 = np.minimum(j0 + 1, grid.nj - 1)
    wj = fj - j0

    if grid.is_global_lon:
        fi = np.mod(fi, grid.ni)
        i0 = np.floor(fi).astype(np.int64) % grid.ni
        i1 = (i0 + 1) % grid.ni
        wi = fi - np.floor(fi)
    else:
        valid &= fi <= grid.ni - 1 + 1e-9
        fi = np.clip(fi, 0, grid.ni - 1)
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, max(grid.ni - 2, 0))
        i1 = np.minimum(i0 + 1, grid.ni - 1)
        wi = fi - i0
    return Weights(i0, i1, j0, j1, wi, wj, valid)


# --------------------------------------------------------------------------- unit conversion


def wind_dir(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (degrees the wind blows FROM) from u/v components."""
    d = np.degrees(np.arctan2(-u, -v)) % 360.0
    return np.where((u == 0) & (v == 0), np.nan, d)


def to_station_values(raw: dict[str, np.ndarray], step: int, n: int) -> dict[str, np.ndarray]:
    """Convert interpolated internal fields to contract columns."""
    nan = np.full(n, np.nan)

    def g(k):
        return raw.get(k, nan)

    u, v = g("u"), g("v")
    tcc = g("tcc")
    if np.isfinite(tcc).any() and np.nanmax(tcc) <= 1.0001:
        tcc = tcc * 100.0
    return {
        "t": g("t") - K0,
        "td": g("td") - K0,
        "wspd": np.hypot(u, v) * KT_PER_MPS,
        "wdir": wind_dir(u, v),
        "gust": g("gust") * KT_PER_MPS,
        "mslp": g("mslp") / 100.0,
        "tcc": np.clip(tcc, 0.0, 100.0),
        "vis_m": g("vis"),
    }


# --------------------------------------------------------------------------- runs


def run_available(source: ModelSource, run: datetime) -> bool:
    """A run counts as available once the index of its last step exists."""
    try:
        return http.exists(source.index_url(run, source.steps[-1]), retries=2, timeout=30)
    except Exception:
        return False


def candidate_runs(source: ModelSource, now: datetime, lookback_h: int = 48) -> list[datetime]:
    """Run times (newest first) within ``lookback_h`` hours of ``now``."""
    now = now.astimezone(timezone.utc)
    start = now - timedelta(hours=lookback_h)
    t = now.replace(minute=0, second=0, microsecond=0)
    out = []
    while t >= start:
        if t.hour in source.run_hours:
            out.append(t)
        t -= timedelta(hours=1)
    return out


@dataclass
class StepResult:
    step: int
    fields: dict[str, bytes]
    requests: int
    nbytes: int
    error: str | None = None


def _fetch_step(source: ModelSource, run: datetime, step: int, max_gap: int) -> StepResult:
    try:
        index_text = http.get(source.index_url(run, step), timeout=60).decode("utf-8", "replace")
        entries = source.parse_index(index_text)
        selected = source.select(entries, step)
        if not selected:
            return StepResult(step, {}, 1, len(index_text), "no matching messages")
        fields, nreq, nbytes = download_messages(
            source.grib_url(run, step), selected, max_gap=max_gap
        )
        return StepResult(step, fields, nreq + 1, nbytes + len(index_text))
    except Exception as exc:
        return StepResult(step, {}, 0, 0, f"{type(exc).__name__}: {exc}")


def fetch_run(
    source: ModelSource,
    run: datetime,
    stations: pd.DataFrame,
    *,
    workers: int = 8,
    max_gap: int = 64 * 1024,
    log: Callable[[str], None] | None = print,
    fetch_step: Callable[[ModelSource, datetime, int, int], StepResult] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one run and interpolate every step to the stations.

    Returns the contract forecast frame and stats (bytes, requests, seconds,
    failed steps).
    """
    fetch_step = fetch_step or _fetch_step
    run = run.astimezone(timezone.utc)
    lat = stations["lat"].to_numpy("float64")
    lon = stations["lon"].to_numpy("float64")
    ids = stations["station"].astype(str).to_numpy()
    n = len(ids)
    weights_cache: dict[Grid, Weights] = {}
    frames = []
    stats = {"bytes": 0, "requests": 0, "failed_steps": [], "seconds": 0.0}
    t0 = time.monotonic()

    def handle(res: StepResult) -> None:
        stats["bytes"] += res.nbytes
        stats["requests"] += res.requests
        if res.error:
            stats["failed_steps"].append(res.step)
            if log:
                log(f"  {source.name} {run:%Y%m%d%H} +{res.step}h failed: {res.error}")
            return
        raw: dict[str, np.ndarray] = {}
        for var, msg in res.fields.items():
            try:
                field, grid, _ = decode_message(msg)
            except Exception as exc:
                if log:
                    log(f"  {source.name} +{res.step}h {var}: decode failed: {exc}")
                continue
            w = weights_cache.get(grid)
            if w is None:
                w = weights_cache[grid] = bilinear_weights(grid, lat, lon)
            raw[var] = w.apply(field)
            del field
        vals = to_station_values(raw, res.step, n)
        df = pd.DataFrame({"station": ids, **vals})
        df["lead_h"] = res.step
        frames.append(df)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch_step, source, run, s, max_gap) for s in source.steps]
        for k in range(len(futures)):  # in step order; downloads continue meanwhile
            res = futures[k].result()
            futures[k] = None  # drop the raw bytes as soon as they are decoded
            handle(res)
            del res

    stats["seconds"] = round(time.monotonic() - t0, 1)
    if not frames:
        return empty_fcst(), stats
    out = pd.concat(frames, ignore_index=True)
    out["run"] = pd.Timestamp(run)
    return to_fcst_frame(out), stats


def empty_fcst() -> pd.DataFrame:
    return to_fcst_frame(pd.DataFrame(columns=FCST_COLUMNS))


def to_fcst_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=FCST_COLUMNS).copy()
    df["station"] = df["station"].astype(str)
    df["run"] = pd.to_datetime(df["run"], utc=True).astype("datetime64[ns, UTC]")
    df["lead_h"] = pd.to_numeric(df["lead_h"]).astype("int16")
    df["valid"] = df["run"] + pd.to_timedelta(df["lead_h"].astype("int64"), unit="h")
    for col in FCST_FLOATS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df.sort_values(["station", "lead_h"]).reset_index(drop=True)


def write_fcst(df: pd.DataFrame, path) -> None:
    df = to_fcst_frame(df)
    # Round to what the data supports: smaller files, identical verification.
    for col, nd in (
        ("t", 2),
        ("td", 2),
        ("wspd", 2),
        ("wdir", 1),
        ("gust", 2),
        ("mslp", 2),
        ("tcc", 1),
        ("vis_m", 0),
    ):
        df[col] = df[col].round(nd).astype("float32")
    df.to_parquet(path, index=False, compression="zstd")
