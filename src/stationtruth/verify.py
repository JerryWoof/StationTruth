"""Load observations and forecasts, match forecast/observation pairs, compute scores.

Rules (docs/DATA.md, "Verification rules"):

* each forecast row is matched with the observation at the same station closest
  to ``valid`` within +-30 minutes; routine METARs win ties over SPECIs (if two
  reports of the same type are equally close, the earlier one is used);
* error = forecast - observed, for ``t``, ``td``, ``wspd``, ``gust`` (only when
  both report a gust), ``mslp`` against observed ``slp`` only, and ``wdir``
  as the circular difference wrapped to [-180, 180) when observed wspd >= 5 kt;
* scores: ``n``, ``bias`` (mean error), ``mae``, ``rmse``.

Internally everything is plain numpy: times are integer minutes since the Unix
epoch (UTC), stations are int32 ids (``sid``) into ``Dataset.station_ids``. A
forecast *table* is a ``dict[str, np.ndarray]`` with columns ``sid``, ``run``,
``lead``, ``valid``, the forecast variables and ``oi`` (index of the matched
observation in ``Dataset.obs``, -1 if none).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import registry
from .regions import REGIONS, region_index

log = logging.getLogger(__name__)

#: raw models, then their bias-corrected versions (``<id>_bc``), then ``blend``
#: (see :mod:`stationtruth.registry`)
RAW_MODELS: tuple[str, ...] = registry.RAW_MODELS
MODELS: tuple[str, ...] = registry.MODELS
FC_VARS: tuple[str, ...] = ("t", "td", "wspd", "gust", "wdir", "mslp")
VERIFY_VARS: tuple[str, ...] = ("t", "td", "wspd", "gust", "mslp", "wdir")
OBS_VARS: tuple[str, ...] = ("t", "td", "wdir", "wspd", "gust", "slp")
#: forecast variable -> observed column it is verified against
OBS_COL: dict[str, str] = {
    "t": "t",
    "td": "td",
    "wspd": "wspd",
    "gust": "gust",
    "mslp": "slp",
    "wdir": "wdir",
}
MATCH_TOL_MIN = 30
WDIR_MIN_WSPD = 5.0
MAX_LEAD = 240
WINDOWS: dict[str, int] = {"7d": 7, "30d": 30}
CHUNK = 2_000_000  # rows per chunk for memory-hungry vectorised steps

#: plausible ranges; values outside are treated as missing (decode/GRIB errors)
VALID_RANGE: dict[str, tuple[float, float]] = {
    "t": (-90.0, 65.0),
    "td": (-100.0, 40.0),
    "wspd": (0.0, 250.0),
    "gust": (0.0, 300.0),
    "wdir": (0.0, 360.0),
    "mslp": (850.0, 1090.0),
    "slp": (850.0, 1090.0),
}

_FCST_RE = re.compile(r"^fcst-([a-z0-9_]+)-(\d{10})\.parquet$")
_OBS_RE = re.compile(r"^obs-(\d{4}-\d{2}-\d{2})\.parquet$")

Table = dict[str, np.ndarray]


# --------------------------------------------------------------------- time


def to_minutes(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() // 60)


def from_minutes(m: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=int(m))


def iso(m: int | None) -> str | None:
    if m is None:
        return None
    return from_minutes(m).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(m: int) -> str:
    return from_minutes(m).strftime("%Y%m%d%H")


def iso_array(minutes: np.ndarray) -> list[str]:
    """Vectorised ISO formatting for arrays of epoch minutes."""
    if len(minutes) == 0:
        return []
    d = np.asarray(minutes, dtype="int64").astype("datetime64[m]")
    return [s + ":00Z" for s in np.datetime_as_string(d, unit="m").tolist()]


NAT = np.iinfo("int64").min


def _arrow_minutes(col: pa.ChunkedArray) -> np.ndarray:
    """Arrow column of timestamps (or strings) -> int64 epoch minutes, NaT -> INT64 min."""
    if pa.types.is_timestamp(col.type):
        arr = pc.cast(col, pa.timestamp("s", tz=col.type.tz), safe=False)
        sec = pc.fill_null(pc.cast(arr, pa.int64()), NAT).to_numpy()
        sec = np.asarray(sec, dtype=np.int64)
        return np.where(sec == NAT, sec, sec // 60)
    s = pd.to_datetime(col.to_pandas(), utc=True, errors="coerce")
    v = s.values.astype("datetime64[m]").astype("int64")
    return np.where(pd.isna(s).to_numpy(), NAT, v)


# ----------------------------------------------------------------- stations


class StationVocab:
    """Maps ICAO ids to dense int32 station ids, growing as new ids appear."""

    def __init__(self, ids: list[str] | None = None):
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        for s in ids or []:
            self.add(s)

    def add(self, s: str) -> int:
        i = self.index.get(s)
        if i is None:
            i = len(self.ids)
            self.ids.append(s)
            self.index[s] = i
        return i

    def encode(self, col: pa.ChunkedArray) -> np.ndarray:
        """Arrow string column -> int32 ids (-1 for null/empty)."""
        col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        if not pa.types.is_dictionary(col.type):
            col = pc.dictionary_encode(col)
        codes = np.asarray(pc.fill_null(col.indices, -1).to_numpy(zero_copy_only=False))
        dictionary = col.dictionary.to_pylist()
        lut = np.empty(len(dictionary) + 1, dtype=np.int32)
        lut[-1] = -1  # code -1 indexes the last slot
        for j, s in enumerate(dictionary):
            s = str(s).strip().upper() if s is not None else ""
            lut[j] = self.add(s) if s else -1
        return lut[codes.astype(np.int64)]


@dataclass
class Dataset:
    now: int  # epoch minutes
    station_ids: list[str]
    stations: pd.DataFrame  # one row per sid: station, name, country, lat, lon, elev_m
    region: np.ndarray  # int8 per sid, -1 unknown
    obs: Table  # sorted by (sid, time)
    fc: dict[str, Table] = field(default_factory=dict)  # model -> forecast table
    obs_count_total: int = 0
    first_obs: int | None = None
    runs: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n_sid(self) -> int:
        return len(self.station_ids)


def _read(path: Path, wanted: tuple[str, ...]) -> pa.Table | None:
    try:
        schema = pq.read_schema(path)
        cols = [c for c in wanted if c in schema.names]
        return pq.read_table(path, columns=cols)
    except Exception as exc:  # corrupt/partial file: skip, don't kill the build
        log.warning("cannot read %s: %s", path, exc)
        return None


def _float_col(tbl: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in tbl.column_names:
        return np.full(n, np.nan, dtype=np.float32)
    col = tbl.column(name)
    try:
        col = pc.cast(col, pa.float32())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return pd.to_numeric(col.to_pandas(), errors="coerce").to_numpy(np.float32)
    return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32)


def _qc(name: str, v: np.ndarray) -> np.ndarray:
    lo, hi = VALID_RANGE.get(name, (-np.inf, np.inf))
    if name == "wdir":
        v = np.where(v == 360.0, np.float32(0.0), v)
    bad = (v < lo) | (v > hi)
    if bad.any():
        v = v.copy()
        v[bad] = np.nan
    return v


def list_files(data_dir: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, int, Path]]]:
    """(obs files as (date, path), fcst files as (model, run_minutes, path)), sorted."""
    obs, fcst = [], []
    if not data_dir.is_dir():
        return obs, fcst
    for p in sorted(data_dir.iterdir()):
        m = _OBS_RE.match(p.name)
        if m:
            obs.append((m.group(1), p))
            continue
        m = _FCST_RE.match(p.name)
        if m:
            try:
                run = datetime.strptime(m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            fcst.append((m.group(1), to_minutes(run), p))
    return obs, fcst


def load_stations(data_dir: Path, vocab: StationVocab) -> pd.DataFrame:
    path = data_dir / "stations.parquet"
    cols = ("station", "name", "country", "lat", "lon", "elev_m")
    if path.exists():
        tbl = _read(path, cols)
    else:
        tbl = None
    if tbl is None or "station" not in tbl.column_names or tbl.num_rows == 0:
        return pd.DataFrame(columns=list(cols))
    df = tbl.to_pandas()
    for c in cols:
        if c not in df:
            df[c] = np.nan if c in ("lat", "lon", "elev_m") else ""
    df["station"] = df["station"].astype(str).str.strip().str.upper()
    df = df[df["station"] != ""].drop_duplicates("station", keep="last")
    for s in df["station"]:
        vocab.add(s)
    return df[list(cols)]


def load(data_dir: Path, now: datetime, horizon_days: int = 48) -> Dataset:
    """Read stations, obs and forecast files relevant for ``now``.

    Only data in ``[now - horizon_days, now]`` is loaded (30-day scores + 14-day
    correction window + 72 h leads fit in 48 days). Data after ``now`` is
    ignored, so rebuilding for a past ``now`` is free of look-ahead.
    """
    data_dir = Path(data_dir)
    now_m = to_minutes(now)
    start_m = now_m - horizon_days * 1440
    vocab = StationVocab()
    st = load_stations(data_dir, vocab)
    obs_files, fcst_files = list_files(data_dir)

    # ---- observations
    obs_count_total = 0
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("sid", "time", "metar", *OBS_VARS)}
    for day, path in obs_files:
        try:
            obs_count_total += pq.read_metadata(path).num_rows
        except Exception:
            pass
        d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if to_minutes(d) > now_m or to_minutes(d) + 1440 < start_m:
            continue
        tbl = _read(path, ("station", "time", "type", *OBS_VARS))
        if tbl is None or tbl.num_rows == 0 or not {"station", "time"} <= set(tbl.column_names):
            continue
        n = tbl.num_rows
        sid = vocab.encode(tbl.column("station"))
        tm = _arrow_minutes(tbl.column("time"))
        keep = (sid >= 0) & (tm != NAT) & (tm <= now_m) & (tm >= start_m)
        if "type" in tbl.column_names:
            ty = pc.cast(tbl.column("type"), pa.string())
            metar = np.asarray(
                pc.fill_null(pc.not_equal(pc.utf8_upper(ty), "SPECI"), True).to_numpy(
                    zero_copy_only=False
                ),
                dtype=bool,
            )
        else:
            metar = np.ones(n, dtype=bool)
        parts["sid"].append(sid[keep])
        parts["time"].append(tm[keep])
        parts["metar"].append(metar[keep])
        for v in OBS_VARS:
            parts[v].append(_qc(v, _float_col(tbl, v, n))[keep])
    obs = _finish_obs(parts)

    # ---- forecasts: preallocate per model from parquet metadata and fill in place
    # (appending per-file arrays and concatenating doubles peak memory)
    todo: dict[str, list[tuple[int, Path, int]]] = {}
    for model, run_m, path in fcst_files:
        if model not in RAW_MODELS or run_m > now_m or run_m < start_m:
            continue
        try:
            nrows = pq.read_metadata(path).num_rows
        except Exception as exc:
            log.warning("cannot read %s: %s", path, exc)
            continue
        todo.setdefault(model, []).append((run_m, path, nrows))
    # Stations that never report (a third of the forecast list) cannot be verified,
    # corrected or shown: drop their forecast rows, unless there are no obs at all.
    has_obs = None
    if len(obs["sid"]):
        has_obs = np.zeros(len(vocab.ids), dtype=bool)
        has_obs[obs["sid"]] = True
    fc: dict[str, Table] = {}
    runs: dict[str, list[int]] = {}
    dtypes = {"sid": np.int32, "run": np.int32, "lead": np.int16, "valid": np.int32}
    for model, files in todo.items():
        cap = sum(f[2] for f in files)
        T = {k: np.empty(cap, dtype=dt) for k, dt in dtypes.items()}
        for v in FC_VARS:
            T[v] = np.empty(cap, dtype=np.float32)
        off = 0
        for run_m, path, _nrows in files:
            got = _read_fcst(path, run_m, vocab, has_obs)
            if got is None:
                continue
            k = min(len(got["sid"]), cap - off)
            for c, arr in got.items():
                T[c][off : off + k] = arr[:k]
            T["run"][off : off + k] = run_m
            off += k
            runs.setdefault(model, []).append(run_m)
        T["valid"][:off] = T["run"][:off] + T["lead"][:off].astype(np.int32) * 60
        fc[model] = {c: arr[:off] for c, arr in T.items()}

    # ---- station table aligned with sids (stations seen only in data get blanks)
    ids = vocab.ids
    meta = pd.DataFrame({"station": ids})
    meta = meta.merge(st, on="station", how="left") if len(st) else meta
    for c, dflt in (("name", ""), ("country", ""), ("lat", np.nan), ("lon", np.nan)):
        if c not in meta:
            meta[c] = dflt
    if "elev_m" not in meta:
        meta["elev_m"] = np.nan
    meta["name"] = meta["name"].fillna("").astype(str)
    meta["country"] = meta["country"].fillna("").astype(str)
    for c in ("lat", "lon", "elev_m"):
        meta[c] = pd.to_numeric(meta[c], errors="coerce").astype("float64")
    region = region_index(meta["country"].tolist(), meta["lat"].to_numpy(), meta["lon"].to_numpy())

    ds = Dataset(
        now=now_m,
        station_ids=ids,
        stations=meta,
        region=region,
        obs=obs,
        fc=fc,
        obs_count_total=obs_count_total,
        first_obs=_first_obs(obs_files, now_m),
        runs={m: sorted(r) for m, r in runs.items()},
    )
    return ds


def _read_fcst(
    path: Path, run_m: int, vocab: StationVocab, has_obs: np.ndarray | None = None
) -> Table | None:
    """One forecast file -> sid, lead and FC_VARS arrays (one row per station/lead)."""
    tbl = _read(path, ("station", "valid", "lead_h", *FC_VARS))
    if tbl is None or tbl.num_rows == 0 or "station" not in tbl.column_names:
        return None
    n = tbl.num_rows
    sid = vocab.encode(tbl.column("station"))
    if "lead_h" in tbl.column_names:
        lead = _float_col(tbl, "lead_h", n)
    elif "valid" in tbl.column_names:
        vm = _arrow_minutes(tbl.column("valid"))
        lead = np.where(vm == NAT, np.nan, (vm - run_m) / 60.0).astype(np.float32)
    else:
        return None
    keep = (sid >= 0) & np.isfinite(lead) & (lead >= 0) & (lead <= MAX_LEAD)
    if has_obs is not None:
        known = sid < len(has_obs)
        keep[known] &= has_obs[sid[known]]
        keep[~known] = False
    lead_i = np.where(keep, lead, 0).astype(np.int16)
    key = sid.astype(np.int64) * 1024 + lead_i
    key[~keep] = -1
    _, first = np.unique(key, return_index=True)  # one row per (station, lead)
    sel = first[key[first] >= 0]
    out: Table = {"sid": sid[sel], "lead": lead_i[sel]}
    for v in FC_VARS:
        out[v] = _qc(v, _float_col(tbl, v, n))[sel]
    return out


def _first_obs(obs_files: list[tuple[str, Path]], now_m: int) -> int | None:
    """Earliest observation time in the whole archive (not just the loaded window)."""
    for _, path in obs_files:
        tbl = _read(path, ("time",))
        if tbl is None or "time" not in tbl.column_names or tbl.num_rows == 0:
            continue
        tm = _arrow_minutes(tbl.column("time"))
        tm = tm[(tm != NAT) & (tm <= now_m)]
        if len(tm):
            return int(tm.min())
    return None


def _finish_obs(parts: dict[str, list[np.ndarray]]) -> Table:
    if not parts["sid"]:
        out: Table = {
            "sid": np.zeros(0, np.int32),
            "time": np.zeros(0, np.int64),
            "metar": np.zeros(0, bool),
        }
        for v in OBS_VARS:
            out[v] = np.zeros(0, np.float32)
        out["key"] = np.zeros(0, np.int64)
        return out
    o = {k: np.concatenate(v) for k, v in parts.items()}
    key = (o["sid"].astype(np.int64) << 32) | o["time"].astype(np.int64)
    # sort by key, METAR first among duplicates, then keep the first of each key
    order = np.lexsort((~o["metar"], key))
    key = key[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    order = order[first]
    out = {k: v[order] for k, v in o.items()}
    out["time"] = out["time"].astype(np.int64)
    out["key"] = key[first]
    return out


# ----------------------------------------------------------------- matching


def match_nearest(
    okey: np.ndarray, ometar: np.ndarray, qkey: np.ndarray, tol: int = MATCH_TOL_MIN
) -> np.ndarray:
    """Index of the nearest observation for each query key, -1 if none within ``tol``.

    ``okey`` (sorted) and ``qkey`` are ``sid << 32 | minutes``; the station part
    keeps matches within one station. On equal distance a METAR beats a SPECI;
    otherwise the earlier report wins.
    """
    n = len(okey)
    out = np.full(len(qkey), -1, dtype=np.int64)
    if n == 0 or len(qkey) == 0:
        return out
    r = np.searchsorted(okey, qkey, side="left")  # okey[r] >= q
    l_ = r - 1  # okey[l] < q
    big = np.int64(1) << 40
    rc = np.minimum(r, n - 1)
    lc = np.maximum(l_, 0)
    dr = np.where(r < n, okey[rc] - qkey, big)
    dl = np.where(l_ >= 0, qkey - okey[lc], big)
    take_r = (dr < dl) | ((dr == dl) & ometar[rc] & ~ometar[lc])
    idx = np.where(take_r, rc, lc)
    dist = np.where(take_r, dr, dl)
    ok = dist <= tol
    out[ok] = idx[ok]
    return out


def match_table(T: Table, obs: Table) -> None:
    """Attach ``oi`` (matched obs index or -1) to a forecast table, in place."""
    n = len(T["sid"])
    oi = np.empty(n, dtype=np.int32)
    for a in range(0, n, CHUNK):  # chunked: bounded temporaries
        b = min(a + CHUNK, n)
        q = (T["sid"][a:b].astype(np.int64) << 32) | T["valid"][a:b].astype(np.int64)
        oi[a:b] = match_nearest(obs["key"], obs["metar"], q)
    T["oi"] = oi


def wrap_dir(d: np.ndarray) -> np.ndarray:
    """Angle difference wrapped to [-180, 180)."""
    return ((d + 180.0) % 360.0) - 180.0


def errors_at(T: Table, obs: Table, var: str, rows: np.ndarray) -> np.ndarray:
    """float32 errors fcst - obs for ``rows`` of T (all matched); NaN if not verifiable."""
    o_idx = T["oi"][rows]
    f = T[var][rows]
    o = obs[OBS_COL[var]][o_idx]
    if var == "wdir":
        e = wrap_dir(f - o)
        e[~(obs["wspd"][o_idx] >= WDIR_MIN_WSPD)] = np.nan
    else:
        e = f - o
    return e.astype(np.float32, copy=False)


def pair_errors(T: Table, obs: Table, var: str) -> tuple[np.ndarray, np.ndarray]:
    """(row indices into T, float32 errors fcst - obs) for verifiable pairs of ``var``."""
    sel = np.flatnonzero(T["oi"] >= 0)
    e = errors_at(T, obs, var, sel)
    ok = np.isfinite(e)
    return sel[ok], e[ok]


# ------------------------------------------------------------------- scores


@dataclass
class Sums:
    """Accumulated n, sum(e), sum|e|, sum(e^2) over some index space."""

    n: np.ndarray
    s: np.ndarray
    sa: np.ndarray
    ss: np.ndarray

    @classmethod
    def of(cls, idx: np.ndarray, e: np.ndarray, size: int) -> Sums:
        e64 = e.astype(np.float64)
        return cls(
            np.bincount(idx, minlength=size)[:size].astype(np.float64),
            np.bincount(idx, weights=e64, minlength=size)[:size],
            np.bincount(idx, weights=np.abs(e64), minlength=size)[:size],
            np.bincount(idx, weights=e64 * e64, minlength=size)[:size],
        )


def stat(n, s, sa, ss, nd: int = 2) -> dict:
    """Score dict; scores.json keeps 2 decimals so close models stay distinguishable."""
    n = int(n)
    if n == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None}
    return {
        "n": n,
        "bias": round(float(s) / n, nd),
        "mae": round(float(sa) / n, nd),
        "rmse": round(float(np.sqrt(max(ss, 0.0) / n)), nd),
    }


@dataclass
class ScoreResult:
    scores: dict  # scores.json content
    station: dict  # window -> model -> var -> Sums over sid
    pairs: dict  # model -> number of matched forecast rows (any variable)


class Scorer:
    """Accumulates scores model by model (so big tables can be freed in between).

    Scores by window x (global | region) x model x var x lead, plus per-station
    sums. Each pair goes to the narrowest window containing it (by valid time);
    wider windows are cumulative sums over those slots.
    """

    def __init__(self, obs: Table, region: np.ndarray, now: int):
        self.obs, self.now = obs, now
        self.n_sid = len(region)
        self.nreg = len(REGIONS) + 1  # slot 0 = unknown region
        self.nl = MAX_LEAD + 1
        self.wins = sorted(WINDOWS.items(), key=lambda kv: kv[1])
        self.onehot = np.zeros((self.nreg, self.n_sid))
        self.onehot[region.astype(np.int64) + 1, np.arange(self.n_sid)] = 1.0
        self.scores: dict = {"windows": {}}
        self.station: dict = {}
        self.pairs: dict[str, int] = {}
        for wname, days in WINDOWS.items():
            self.scores["windows"][wname] = {
                "from": iso(now - days * 1440),
                "to": iso(now),
                "global": {m: {} for m in MODELS},
                "regions": {r: {m: {} for m in MODELS} for r in REGIONS},
            }
            self.station[wname] = {m: {} for m in MODELS}

    def add(self, model: str, T: Table | None) -> None:
        nw, n_sid, nl, now = len(self.wins), self.n_sid, self.nl, self.now
        size = nw * n_sid * nl
        if T is not None:
            self.pairs[model] = int((T["oi"] >= 0).sum())
        for var in VERIFY_VARS:
            if T is not None:
                rows, e = pair_errors(T, self.obs, var)
                age = now - T["valid"][rows].astype(np.int64)
                slot = np.full(len(rows), nw, dtype=np.int64)
                for k in range(nw - 1, -1, -1):
                    slot[age < self.wins[k][1] * 1440] = k
                keep = slot < nw
                rows = rows[keep]
                cell = (slot[keep] * n_sid + T["sid"][rows]) * nl + T["lead"][rows]
                S = Sums.of(cell, e[keep], size)
                del rows, e, age, slot, keep, cell
            else:
                S = Sums.of(np.zeros(0, np.int64), np.zeros(0), size)
            arrs = [np.cumsum(a.reshape(nw, n_sid, nl), axis=0) for a in (S.n, S.s, S.sa, S.ss)]
            for k, (wname, _) in enumerate(self.wins):
                win = self.scores["windows"][wname]
                M = [a[k] for a in arrs]  # (n_sid, nl)
                self.station[wname][model][var] = Sums(*(m.sum(axis=1) for m in M))
                grid = [self.onehot @ m for m in M]  # (nreg, nl)
                win["global"][model][var] = _var_block([g.sum(axis=0) for g in grid])
                for ri, rname in enumerate(REGIONS):
                    win["regions"][rname][model][var] = _var_block([g[ri + 1] for g in grid])

    def result(self) -> ScoreResult:
        for m in MODELS:  # models never added still get the full (empty) shape
            if not self.station[self.wins[0][0]][m]:
                self.add(m, None)
        return ScoreResult(self.scores, self.station, self.pairs)


def compute_scores(
    tables: dict[str, Table], obs: Table, region: np.ndarray, now: int
) -> ScoreResult:
    """Scores for all models in ``tables`` (see Scorer)."""
    sc = Scorer(obs, region, now)
    for m in MODELS:
        if m in tables:
            sc.add(m, tables[m])
    return sc.result()


def _var_block(arrs: list[np.ndarray]) -> dict:
    n, s, sa, ss = arrs
    by_lead = [
        {"lead_h": int(lead), **stat(n[lead], s[lead], sa[lead], ss[lead])}
        for lead in np.flatnonzero(n)
    ]
    return {"by_lead": by_lead, "all": stat(n.sum(), s.sum(), sa.sum(), ss.sum())}
