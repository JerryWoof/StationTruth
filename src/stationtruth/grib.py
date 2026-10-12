"""Fetch selected GRIB2 messages with HTTP range requests and interpolate to stations.

Only the messages we need are downloaded: each source publishes an index
(byte offset/length per message), adjacent ranges are merged, and a modest
thread pool downloads steps in parallel. Decoding (eccodes) and the
interpolation run in the calling thread, one step at a time, so memory stays
at a few global fields.

Regular lat/lon grids (ECMWF, GFS, AIGFS) are interpolated bilinearly. DWD ICON
comes per parameter and step as small bz2 files on its native unstructured
(icosahedral) grid; there each station gets the inverse-distance-weighted mean
of its 3 nearest cell centres (:func:`unstructured_weights`, cached per grid).
"""

from __future__ import annotations

import bz2
import hashlib
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


@dataclass(frozen=True)
class UGrid:
    """Unstructured grid (ICON): identified by its GRIB ``uuidOfHGrid``."""

    uuid: str
    n: int


def decode_message(msg: bytes) -> tuple[np.ndarray, Grid | UGrid, dict]:
    """Decode one GRIB message (NaN for missing values).

    Regular lat/lon grids give a ``(nj, ni)`` float64 array and a :class:`Grid`;
    unstructured grids (ICON) a 1-D array over the cells and a :class:`UGrid`.
    """
    import eccodes

    h = eccodes.codes_new_from_message(msg)
    try:
        grid_type = eccodes.codes_get(h, "gridType")
        if grid_type == "unstructured_grid":
            values = eccodes.codes_get_values(h).astype("float64")
            if eccodes.codes_get(h, "bitmapPresent"):
                values[values == eccodes.codes_get(h, "missingValue")] = np.nan
            meta = {
                "shortName": eccodes.codes_get(h, "shortName"),
                "units": eccodes.codes_get(h, "units"),
            }
            return values, UGrid(str(eccodes.codes_get(h, "uuidOfHGrid")), len(values)), meta
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


# --------------------------------------------------------------------------- unstructured grids

#: number of nearest cell centres averaged for a station on an unstructured grid
UGRID_K = 3
EARTH_KM = 6371.0


@dataclass
class PointWeights:
    """Weighted mean of ``k`` grid points per station (unstructured grids)."""

    idx: np.ndarray  # (n, k) int64 point indices
    w: np.ndarray  # (n, k) weights, rows sum to 1 (0 for padding)
    valid: np.ndarray  # (n,) bool

    def apply(self, field: np.ndarray) -> np.ndarray:
        v = field[self.idx]
        w = np.where(np.isfinite(v), self.w, 0.0)
        tot = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (w * np.nan_to_num(v)).sum(axis=1) / tot
        out[(tot <= 0) | ~self.valid] = np.nan
        return out


def _xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la = np.deg2rad(np.asarray(lat, dtype="float64"))
    lo = np.deg2rad(np.asarray(lon, dtype="float64"))
    c = np.cos(la)
    return np.stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)], axis=1)


def unstructured_weights(
    clat: np.ndarray,
    clon: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    k: int = UGRID_K,
    cell: float = 0.01,
) -> PointWeights:
    """Inverse-distance weights of the ``k`` nearest grid points to each station.

    Grid points (``clat``/``clon``, degrees) are bucketed in 3-D unit-vector
    space with bucket size ``cell`` (0.01 ~ 64 km); each station looks at its
    own and the 26 neighbouring buckets, so every point within ``cell`` x 6371
    km is a candidate (ICON's ~13 km cells give dozens). A station on a grid
    point takes that point's value. No scipy needed.
    """
    P = _xyz(clat, clon)
    Q = _xyz(lat, lon)
    nq = len(Q)
    nb = int(np.ceil(2.0 / cell)) + 3

    def key(ijk: np.ndarray) -> np.ndarray:
        return ((ijk[:, 0] + 1) * nb + ijk[:, 1] + 1) * nb + ijk[:, 2] + 1

    pk = key(np.floor((P + 1.0) / cell).astype(np.int64))
    order = np.argsort(pk, kind="stable")
    pks = pk[order]
    del pk
    ok_q = np.isfinite(Q).all(axis=1)
    qi = np.floor((np.where(ok_q[:, None], Q, 0.0) + 1.0) / cell).astype(np.int64)
    reps, pts = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kk = key(qi + np.array([dx, dy, dz]))
                lo = np.searchsorted(pks, kk, side="left")
                hi = np.searchsorted(pks, kk, side="right")
                lens = hi - lo
                tot = int(lens.sum())
                if tot == 0:
                    continue
                rep = np.repeat(np.arange(nq), lens)
                within = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens)
                reps.append(rep)
                pts.append(order[np.repeat(lo, lens) + within])
    idx = np.zeros((nq, k), dtype=np.int64)
    w = np.zeros((nq, k), dtype=np.float64)
    valid = np.zeros(nq, dtype=bool)
    if reps:
        rep = np.concatenate(reps)
        pt = np.concatenate(pts)
        d2 = ((P[pt] - Q[rep]) ** 2).sum(axis=1)  # squared chord length on the unit sphere
        srt = np.lexsort((d2, rep))
        rep, pt, d2 = rep[srt], pt[srt], d2[srt]
        start = np.searchsorted(rep, np.arange(nq), side="left")
        end = np.searchsorted(rep, np.arange(nq), side="right")
        for j in range(k):
            has = (start + j < end) & ok_q
            pos = np.minimum(start + j, len(rep) - 1)
            idx[has, j] = pt[pos[has]]
            w[has, j] = 1.0 / np.maximum(d2[pos[has]], 1e-18)
        valid = (end > start) & ok_q
        # a station (numerically) on a grid point takes that value
        exact = valid & (d2[np.minimum(start, len(rep) - 1)] < 1e-14)
        w[exact] = 0.0
        w[exact, 0] = 1.0
        s = w.sum(axis=1)
        w[valid] /= s[valid, None]
    return PointWeights(idx, w, valid)


def _cache_dir() -> Path:
    return Path(os.environ.get("STATIONTRUTH_CACHE") or Path.home() / ".cache" / "stationtruth")


def _stations_key(lat: np.ndarray, lon: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(lat, dtype="float64").tobytes())
    h.update(np.ascontiguousarray(lon, dtype="float64").tobytes())
    return h.hexdigest()[:16]


def fetch_ugrid_coords(source: ModelSource, run: datetime) -> tuple[np.ndarray, np.ndarray, str]:
    """(clat, clon, uuid) of an unstructured-grid source (ICON) for ``run``."""
    out = []
    uuid = ""
    for which in ("CLAT", "CLON"):
        blob = http.get(source.coord_url(run, which), timeout=120)  # type: ignore[attr-defined]
        values, grid, _ = decode_message(bz2.decompress(blob))
        uuid = getattr(grid, "uuid", "")
        out.append(values)
    return out[0], out[1], uuid


def ugrid_weights(
    source: ModelSource,
    run: datetime,
    grid: UGrid,
    lat: np.ndarray,
    lon: np.ndarray,
    coords: Callable[[ModelSource, datetime], tuple[np.ndarray, np.ndarray, str]] | None = None,
    cache_dir: Path | None = None,
) -> PointWeights:
    """Station weights for an unstructured grid, cached on disk per grid and station list."""
    cache_dir = _cache_dir() if cache_dir is None else cache_dir
    path = cache_dir / f"ugrid-{source.name}-{grid.uuid}-{grid.n}-{_stations_key(lat, lon)}.npz"
    if path.exists():
        try:
            z = np.load(path)
            return PointWeights(z["idx"], z["w"], z["valid"])
        except Exception:
            pass
    clat, clon, uuid = (coords or fetch_ugrid_coords)(source, run)
    if len(clat) != grid.n or (uuid and uuid != grid.uuid):
        raise ValueError(f"{source.name}: coordinate grid {uuid}/{len(clat)} != field grid")
    pw = unstructured_weights(clat, clon, lat, lon)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, idx=pw.idx, w=pw.w, valid=pw.valid)
        tmp.replace(path)
    except OSError:
        pass
    return pw


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
    """A run counts as available once the index (or file) of its last step exists."""
    try:
        return http.exists(source.availability_url(run), retries=2, timeout=30)
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


def _fetch_files(source: ModelSource, run: datetime, step: int) -> StepResult:
    """``layout == "files"`` (ICON): one bz2-compressed GRIB file per parameter."""
    fields: dict[str, bytes] = {}
    nreq = nbytes = 0
    for var, param in source.files(step):  # type: ignore[attr-defined]
        blob = http.get(source.file_url(run, step, param), timeout=120)  # type: ignore[attr-defined]
        nreq += 1
        nbytes += len(blob)
        fields[var] = bz2.decompress(blob)
    return StepResult(step, fields, nreq, nbytes)


def _fetch_step(source: ModelSource, run: datetime, step: int, max_gap: int) -> StepResult:
    try:
        if source.layout == "files":
            return _fetch_files(source, run, step)
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
    ugrid_coords: Callable[[ModelSource, datetime], tuple[np.ndarray, np.ndarray, str]]
    | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one run and interpolate every step to the stations.

    Returns the contract forecast frame and stats (bytes, requests, seconds,
    failed steps). ``ugrid_coords`` overrides how unstructured-grid coordinates
    are fetched (tests).
    """
    fetch_step = fetch_step or _fetch_step
    run = run.astimezone(timezone.utc)
    lat = stations["lat"].to_numpy("float64")
    lon = stations["lon"].to_numpy("float64")
    ids = stations["station"].astype(str).to_numpy()
    n = len(ids)
    weights_cache: dict[Grid | UGrid, Weights | PointWeights] = {}
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
                try:
                    if isinstance(grid, UGrid):
                        w = ugrid_weights(source, run, grid, lat, lon, coords=ugrid_coords)
                    else:
                        w = bilinear_weights(grid, lat, lon)
                except Exception as exc:
                    if log:
                        log(f"  {source.name} +{res.step}h {var}: no station weights: {exc}")
                    stats["failed_steps"].append(res.step)
                    return
                weights_cache[grid] = w
            raw[var] = w.apply(field)
            del field
        vals = to_station_values(raw, res.step, n)
        df = pd.DataFrame({"station": ids, **vals})
        df["lead_h"] = res.step
        frames.append(df)

    steps = list(source.steps)
    workers = max(1, workers)
    ahead = 2 * workers  # bound the downloaded-but-not-decoded steps held in memory
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[int, object] = {}
        nxt = 0
        for k in range(len(steps)):  # in step order; downloads continue meanwhile
            while nxt < len(steps) and nxt < k + ahead:
                futures[nxt] = pool.submit(fetch_step, source, run, steps[nxt], max_gap)
                nxt += 1
            res = futures.pop(k).result()  # drop the raw bytes once decoded
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
