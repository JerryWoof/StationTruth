"""Bias correction (docs/DATA.md, "Bias correction").

The correction for run R of a raw model is learned only from verified pairs
that were *available at R*:

* pair valid time ``<= R`` (the observation existed),
* pair run time ``< R`` (strictly earlier run; excludes R's own analysis step),
* pair valid time ``> R - 14 days``.

It is a three-level shrinkage ("pseudo-count") estimate of the mean error
(forecast - observed). With ``S``, ``n`` the error sum and pair count of a group:

* region prior  ``b_r = S_r / (n_r + K_REGION)``, group = region x lead bucket x
  valid-hour bucket (all stations of the region; one global region if no
  region map is given),
* station       ``b_s = (S_s + K_STATION * b_r) / (n_s + K_STATION)``, group = the
  station over all lead and hour buckets,
* cell          ``b_c = (S_c + K_CELL * b_s) / (n_c + K_CELL)``, group = station x
  lead bucket (0-24, 27-48, 51-72 h) x valid-hour bucket (00-05, ... UTC).

Each level is the mean of its own pairs pulled toward the broader level as if
K extra pairs with the broader mean had been seen, so a handful of pairs moves
the correction only a little and a well-sampled cell gets its own mean.
``corrected = raw - b_c``; wspd/gust are clipped at 0 and the corrected td is
capped at the corrected t. Without any pair (all levels empty) the raw value is
kept. Wind direction is not corrected. The constants were chosen by
``scripts/evaluate_correction.py`` (out-of-sample RMSE on real data).

Vectorised: per grouping level the matched pairs are sorted by
``group << 33 | (2 * valid + is_lead0)``; the extra half-step for lead-0 pairs
(whose run equals their valid time) makes them usable only for later runs.
Window sums are differences of a cumulative sum at two ``searchsorted``
positions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import registry
from .verify import CHUNK, FC_VARS, Table, errors_at

WINDOW_DAYS = 14
#: pseudo-counts: how many pairs of the broader level the narrower level's mean
#: is blended with (see module docstring)
K_CELL = 10
K_STATION = 5
K_REGION = 20
#: a correction exists as soon as one pair is available at any level (meta.json)
MIN_PAIRS = 1
CORRECTED_VARS: tuple[str, ...] = ("t", "td", "wspd", "gust", "mslp")
NONNEG: frozenset[str] = frozenset({"wspd", "gust"})
N_HOUR_BUCKETS = 4
N_LEAD_BUCKETS = 3


def lead_bucket(lead: np.ndarray) -> np.ndarray:
    lead = lead.astype(np.int64)
    return np.where(lead <= 24, 0, np.where(lead <= 48, 1, 2))


def hour_bucket(valid_min: np.ndarray) -> np.ndarray:
    return (valid_min.astype(np.int64) // 60 % 24) // 6


def bucket_group(T: Table) -> np.ndarray:
    """Dense group id: sid x lead bucket x valid-hour bucket."""
    return (
        T["sid"].astype(np.int64) * (N_LEAD_BUCKETS * N_HOUR_BUCKETS)
        + lead_bucket(T["lead"]) * N_HOUR_BUCKETS
        + hour_bucket(T["valid"])
    )


def region_group(T: Table, region: np.ndarray | None) -> np.ndarray:
    """Dense group id: region (slot 0 = unknown/global) x lead bucket x valid-hour bucket."""
    if region is None or len(region) == 0:
        reg = np.zeros(len(T["sid"]), np.int64)
    else:
        sid = T["sid"].astype(np.int64)
        reg = np.zeros(len(sid), np.int64)
        known = sid < len(region)
        reg[known] = region[sid[known]].astype(np.int64) + 1
    return (
        reg * (N_LEAD_BUCKETS * N_HOUR_BUCKETS)
        + lead_bucket(T["lead"]) * N_HOUR_BUCKETS
        + hour_bucket(T["valid"])
    )


#: errors are summed as integers of 1e-4 units, so window sums are exact and do not
#: depend on unrelated (e.g. later) pairs that precede them in the cumulative sum
ERR_SCALE = 10_000


def _windows(
    T: Table, group: np.ndarray, matched: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(order of ``matched`` by group/availability, hi, lo) for the query ``rows``.

    For query row r the pairs of its group available at its run R are
    ``matched[order[lo[r]:hi[r]]]``.
    """
    # 2*valid (+1 for lead 0): a lead-0 pair (run == valid) sorts after the other pairs
    # valid at the same minute, so "<= 2R" keeps pairs with valid <= R and run < R.
    keys = (group[matched] << 33) | (
        2 * T["valid"][matched].astype(np.int64) + (T["lead"][matched] == 0)
    )
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    n_q = len(rows)
    hi = np.empty(n_q, dtype=np.int32)  # positions < 2**31 (pairs per model)
    lo = np.empty(n_q, dtype=np.int32)
    for a in range(0, n_q, CHUNK):
        r = rows[a : a + CHUNK]
        base = group[r] << 33
        run2 = 2 * T["run"][r].astype(np.int64)
        hi[a : a + CHUNK] = np.searchsorted(keys, base | run2, side="right")
        # lower edge: valid > R - 14 d, i.e. avail > 2 (R - 14 d) + 1
        lo[a : a + CHUNK] = np.searchsorted(
            keys, base | (run2 - 2 * WINDOW_DAYS * 1440 + 1), side="right"
        )
    return order, hi, lo


def _window_sums(
    e: np.ndarray, order: np.ndarray, hi: np.ndarray, lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(error sum, pair count) over each query's window; ``e`` aligned with matched."""
    e = e[order]
    fin = np.isfinite(e)
    cs = np.zeros(len(e) + 1, dtype=np.int64)
    np.cumsum(np.rint(np.where(fin, e, 0).astype(np.float64) * ERR_SCALE), out=cs[1:])
    cn = np.zeros(len(e) + 1, dtype=np.int32)
    np.cumsum(fin, out=cn[1:])
    del e, fin
    S = np.empty(len(hi), dtype=np.float64)
    n = np.empty(len(hi), dtype=np.int32)
    for a in range(0, len(hi), CHUNK):
        h, lw = hi[a : a + CHUNK], lo[a : a + CHUNK]
        S[a : a + CHUNK] = (cs[h] - cs[lw]) / float(ERR_SCALE)
        n[a : a + CHUNK] = cn[h] - cn[lw]
    return S, n


def shrunk_mean(S: np.ndarray, n: np.ndarray, k: float, prior: np.ndarray | float = 0.0):
    """Pseudo-count shrinkage: ``(S + k * prior) / (n + k)``."""
    return (S + k * prior) / (n + k)


def corrections(
    T: Table,
    obs: Table,
    variables: tuple[str, ...] = CORRECTED_VARS,
    rows: np.ndarray | None = None,
    region: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """var -> (error to subtract, cell pair count) for ``rows`` of T (default: all).

    ``region`` maps sid -> region index (-1 unknown) for the region prior; None
    pools all stations into one global prior. The correction is NaN where no
    pair is available at any level.
    """
    if rows is None:
        rows = np.arange(len(T["sid"]))
    matched = np.flatnonzero(T["oi"] >= 0)
    levels = []
    for group in (bucket_group(T), T["sid"].astype(np.int64), region_group(T, region)):
        levels.append(_windows(T, group, matched, rows))
        del group
    out = {}
    for var in variables:
        e = errors_at(T, obs, var, matched)
        (S_c, n_c), (S_s, n_s), (S_r, n_r) = (_window_sums(e, *lev) for lev in levels)
        del e
        b = shrunk_mean(S_r, n_r, K_REGION)
        b = shrunk_mean(S_s, n_s, K_STATION, b)
        b = shrunk_mean(S_c, n_c, K_CELL, b)
        b[(n_s == 0) & (n_r == 0)] = np.nan  # nothing known at all (cell is in station)
        out[var] = (b.astype(np.float32), n_c)
    return out


def correction(
    T: Table, obs: Table, var: str, region: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(error to subtract, cell pair count) for every row of T, one variable."""
    return corrections(T, obs, (var,), region=region)[var]


def corrected_table(
    T: Table, obs: Table, rows: np.ndarray | None = None, region: np.ndarray | None = None
) -> tuple[Table, dict[str, float]]:
    """Bias-corrected copy of raw table T, restricted to ``rows`` if given.

    Columns that are not corrected (keys, wdir, oi) are shared with T when
    ``rows`` is None, else they are (new) subsets.
    """
    corr = corrections(T, obs, CORRECTED_VARS, rows, region)
    out = {k: (v if rows is None else v[rows]) for k, v in T.items() if k not in corr}
    applied: dict[str, float] = {}
    for var in CORRECTED_VARS:
        c, _ = corr.pop(var)
        has = np.isfinite(c)
        raw = T[var] if rows is None else T[var][rows]
        v = raw - np.where(has, c, np.float32(0))
        if var in NONNEG:
            v = np.maximum(v, np.float32(0))  # NaN stays NaN
        out[var] = v.astype(np.float32)
        finite = np.isfinite(raw)
        applied[var] = float(has[finite].mean()) if finite.any() else 0.0
    if "t" in out and "td" in out:
        # dew point cannot exceed temperature; NaN t leaves td unchanged
        out["td"] = np.fmin(out["td"], out["t"])
    return out, applied


BLEND_MAX_AGE_H = 12
BLEND_MAX_LEAD_H = 72


def _rebase(T: Table, blend_runs: np.ndarray) -> Table:
    """Re-key a corrected table onto blend runs.

    For each blend run R the component is the latest run of T with
    ``R - 12 h <= run <= R``; its rows with ``R <= valid <= run + 72 h`` are
    returned with ``run = R`` and ``lead = valid - R``. One source run can feed
    several blend runs (e.g. ECMWF 00Z for blend 00Z and 06Z).
    """
    order = np.argsort(T["run"], kind="stable")
    run_sorted = T["run"][order]
    runs = np.unique(run_sorted)
    src_i = np.searchsorted(runs, blend_runs, side="right") - 1
    parts: list[np.ndarray] = []
    new_run: list[np.ndarray] = []
    for R, i in zip(blend_runs.tolist(), src_i.tolist(), strict=True):
        if i < 0 or R - int(runs[i]) > BLEND_MAX_AGE_H * 60:
            continue
        src = int(runs[i])
        a = np.searchsorted(run_sorted, src, side="left")
        b = np.searchsorted(run_sorted, src, side="right")
        rows = order[a:b]
        v = T["valid"][rows]
        rows = rows[(v >= R) & (T["lead"][rows].astype(np.int32) <= BLEND_MAX_LEAD_H)]
        parts.append(rows)
        new_run.append(np.full(len(rows), R, dtype=np.int32))
    rows = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    out = {k: v[rows] for k, v in T.items()}
    out["run"] = np.concatenate(new_run) if new_run else np.zeros(0, np.int32)
    out["lead"] = ((out["valid"].astype(np.int64) - out["run"]) // 60).astype(np.int16)
    return out


def blend_table(*tables: Table | None) -> Table | None:
    """Real-time blend of N corrected tables (``ecmwf_bc``, ``gfs_bc``, ``aifs_bc``, ...).

    Blend runs are the union of all models' run times. For blend run R each
    component comes from that model's latest run <= R (at most 12 h older, valid
    time still within its 72 h), matched on (station, valid); ``lead = valid - R``.
    Values are the mean of the components available for that row and variable
    (circular mean for wdir; a model without a variable, e.g. AIFS gust, simply
    does not contribute), falling back to the single available one. Uses no run
    later than R, and each component's correction is itself look-ahead free, so
    the blend is too.
    """
    tabs = [t for t in tables if t is not None and len(t["sid"])]
    if not tabs:
        return None
    blend_runs = np.unique(np.concatenate([t["run"] for t in tabs]))
    tabs = [r for r in (_rebase(t, blend_runs) for t in tabs) if len(r["sid"])]
    return _join_mean(tabs) if tabs else None


def _join_mean(tabs: list[Table]) -> Table:
    """Row-wise mean of any number of tables on (sid, run, lead), NaN-aware.

    The tables are stacked and sorted by key; each group's sums and finite counts
    come from ``np.add.reduceat``. Output rows are sorted by (sid, run, lead).
    """
    if len(tabs) == 1:
        return dict(tabs[0])

    def key(t: Table) -> np.ndarray:
        return (
            (t["sid"].astype(np.int64) << 40)
            | (t["run"].astype(np.int64) << 8)
            | t["lead"].astype(np.int64)
        )

    k = np.concatenate([key(t) for t in tabs])
    order = np.argsort(k, kind="stable")
    k = k[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    del k

    def stacked(c: str) -> np.ndarray:
        return np.concatenate([t[c] for t in tabs])[order]

    out: Table = {}
    for c in ("sid", "run", "lead", "valid"):
        out[c] = stacked(c)[starts]
    # every component matched (sid, valid) against the same observations: any
    # matched index is the match (-1 = none)
    out["oi"] = np.maximum.reduceat(stacked("oi"), starts)
    for var in FC_VARS:
        x = stacked(var)
        fin = np.isfinite(x)
        n = np.add.reduceat(fin.astype(np.int32), starts)
        if var == "wdir":
            r = np.deg2rad(np.where(fin, x, 0).astype(np.float64))
            s = np.add.reduceat(np.where(fin, np.sin(r), 0.0), starts)
            c = np.add.reduceat(np.where(fin, np.cos(r), 0.0), starts)
            m = np.rad2deg(np.arctan2(s, c)) % 360.0
        else:
            m = np.add.reduceat(np.where(fin, x, 0).astype(np.float64), starts)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = m / n
        m[n == 0] = np.nan
        out[var] = m.astype(np.float32)
        del x, fin
    return out


def build_models(
    raw: dict[str, Table],
    obs: Table,
    keep_from: int | None = None,
    on_table: Callable[[str, Table], None] | None = None,
    trim_from: int | None = None,
    region: np.ndarray | None = None,
) -> tuple[dict[str, Table], dict]:
    """Raw tables -> all models (raw, ``<id>_bc``, blend). Raw tables must have ``oi``.

    Corrections are learned from all raw pairs; ``region`` (sid -> region index,
    e.g. ``Dataset.region``) enables the regional prior, else it is global.
    Corrected tables (and the blend) are produced for rows with
    ``valid > keep_from`` (all rows if None).

    Memory-saving hooks for big archives: ``on_table(model, table)`` is called
    once per model with its complete table (e.g. to accumulate scores); after
    that call, returned tables are trimmed to ``valid > trim_from``. ``raw`` is
    emptied as it is processed so the full history can be freed.
    """

    def subset(T: Table, rows: np.ndarray | None) -> Table:
        return T if rows is None else {k: v[rows] for k, v in T.items()}

    def rows_after(T: Table, t: int | None) -> np.ndarray | None:
        return None if t is None else np.flatnonzero(T["valid"] > t)

    def done(m: str, T: Table) -> Table:
        if on_table is not None:
            on_table(m, T)
        return subset(T, rows_after(T, trim_from))

    tables: dict[str, Table] = {}
    info: dict = {}
    for m in list(raw):
        T = raw.pop(m)
        if m in registry.RAW_MODELS:
            bc = registry.get(m).corrected
            tables[bc], info[bc] = corrected_table(T, obs, rows_after(T, keep_from), region)
        tables[m] = done(m, T)
        del T
    bl = blend_table(*(tables.get(m) for m in registry.CORRECTED_MODELS))
    for m in registry.CORRECTED_MODELS:
        if m in tables:
            tables[m] = done(m, tables[m])
    if bl is not None:
        tables[registry.BLEND] = done(registry.BLEND, bl)
        del bl
    return {m: tables[m] for m in registry.MODELS if m in tables}, info
