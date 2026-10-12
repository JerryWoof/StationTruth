"""Bias correction (docs/DATA.md, "Bias correction").

For each raw model, station, variable, lead bucket (0-24, 27-48, 51-72 h) and
valid-hour bucket (00-05, 06-11, 12-17, 18-23 UTC) the correction for run R is
the mean error of verified pairs that were *available at R*:

* pair valid time ``<= R`` (the observation existed), and
* pair run time ``< R`` (strictly earlier run; excludes R's own analysis step),
* pair valid time ``> R - 14 days``.

At least 8 such pairs are required, otherwise the raw forecast is kept.
``corrected = raw - mean_error``; wspd/gust are clipped at 0. Wind direction
is not corrected. ``blend`` is the mean of ``ecmwf_bc`` and ``gfs_bc`` for the
same station/run/lead (circular mean for wdir), falling back to whichever
exists.

Vectorised: per (model, variable) the pairs are sorted by
``group << 33 | (2 * valid + is_lead0)``; the extra half-step for lead-0 pairs
(whose run equals their valid time) makes them usable only for later runs.
Window sums are differences of a cumulative sum at two
``searchsorted`` positions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .verify import CHUNK, FC_VARS, Table, errors_at

WINDOW_DAYS = 14
MIN_PAIRS = 8
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


#: errors are summed as integers of 1e-4 units, so window sums are exact and do not
#: depend on unrelated (e.g. later) pairs that precede them in the cumulative sum
ERR_SCALE = 10_000


def corrections(
    T: Table,
    obs: Table,
    variables: tuple[str, ...] = CORRECTED_VARS,
    rows: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """var -> (mean error to subtract, pair count) for ``rows`` of T (default: all).

    The mean is NaN where fewer than MIN_PAIRS pairs are available.
    """
    group = bucket_group(T)
    if rows is None:
        rows = np.arange(len(T["sid"]))
    matched = np.flatnonzero(T["oi"] >= 0)
    # 2*valid (+1 for lead 0): a lead-0 pair (run == valid) sorts after the other pairs
    # valid at the same minute, so "<= 2R" keeps pairs with valid <= R and run < R.
    keys = (group[matched] << 33) | (
        2 * T["valid"][matched].astype(np.int64) + (T["lead"][matched] == 0)
    )
    order = np.argsort(keys)
    keys = keys[order]
    matched = matched[order]
    del order
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
    del keys, group
    out = {}
    for var in variables:
        e = errors_at(T, obs, var, matched)
        fin = np.isfinite(e)
        cs = np.zeros(len(e) + 1, dtype=np.int64)
        np.cumsum(np.rint(np.where(fin, e, 0).astype(np.float64) * ERR_SCALE), out=cs[1:])
        cn = np.zeros(len(e) + 1, dtype=np.int32)
        np.cumsum(fin, out=cn[1:])
        del e, fin
        mean = np.empty(n_q, dtype=np.float32)
        count = np.empty(n_q, dtype=np.int32)
        for a in range(0, n_q, CHUNK):
            h, lw = hi[a : a + CHUNK], lo[a : a + CHUNK]
            n = cn[h] - cn[lw]
            with np.errstate(invalid="ignore", divide="ignore"):
                m = (cs[h] - cs[lw]) / (n * float(ERR_SCALE))
            m[n < MIN_PAIRS] = np.nan
            mean[a : a + CHUNK] = m
            count[a : a + CHUNK] = n
        out[var] = (mean, count)
    return out


def correction(T: Table, obs: Table, var: str) -> tuple[np.ndarray, np.ndarray]:
    """(mean error to subtract, pair count) for every row of T, one variable."""
    return corrections(T, obs, (var,))[var]


def corrected_table(
    T: Table, obs: Table, rows: np.ndarray | None = None
) -> tuple[Table, dict[str, float]]:
    """Bias-corrected copy of raw table T, restricted to ``rows`` if given.

    Columns that are not corrected (keys, wdir, oi) are shared with T when
    ``rows`` is None, else they are (new) subsets.
    """
    corr = corrections(T, obs, CORRECTED_VARS, rows)
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
    return out, applied


def _circ_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular mean of two direction arrays (NaN-aware), chunked to bound memory."""
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), CHUNK):
        ra = np.deg2rad(a[i : i + CHUNK], dtype=np.float32)
        rb = np.deg2rad(b[i : i + CHUNK], dtype=np.float32)
        s = np.nan_to_num(np.sin(ra)) + np.nan_to_num(np.sin(rb))
        c = np.nan_to_num(np.cos(ra)) + np.nan_to_num(np.cos(rb))
        d = np.rad2deg(np.arctan2(s, c)) % np.float32(360.0)
        d[np.isnan(ra) & np.isnan(rb)] = np.nan
        out[i : i + CHUNK] = d
    return out


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


def blend_table(a: Table | None, b: Table | None) -> Table | None:
    """Real-time blend of two corrected tables (``ecmwf_bc``, ``gfs_bc``).

    Blend runs are the union of both models' run times. For blend run R each
    component comes from that model's latest run <= R (at most 12 h older, valid
    time still within its 72 h), matched on (station, valid); ``lead = valid - R``.
    Values are the mean of the available components (circular mean for wdir),
    falling back to the single available one. Uses no run later than R, and each
    component's correction is itself look-ahead free, so the blend is too.
    """
    tabs = [t for t in (a, b) if t is not None and len(t["sid"])]
    if not tabs:
        return None
    blend_runs = np.unique(np.concatenate([t["run"] for t in tabs]))
    tabs = [r for r in (_rebase(t, blend_runs) for t in tabs) if len(r["sid"])]
    return _join_mean(tabs) if tabs else None


def _join_mean(tabs: list[Table]) -> Table:
    """Row-wise mean of one or two tables on (sid, run, lead)."""

    def key(t: Table) -> np.ndarray:
        return (
            (t["sid"].astype(np.int64) << 40)
            | (t["run"].astype(np.int64) << 8)
            | t["lead"].astype(np.int64)
        )

    a = tabs[0]
    if len(tabs) == 1:
        return dict(a)
    b = tabs[1]
    ka, kb = key(a), key(b)
    ob = np.argsort(kb)
    kbs = kb[ob]
    del kb
    pos = np.minimum(np.searchsorted(kbs, ka), len(kbs) - 1)
    hit = kbs[pos] == ka
    del kbs, ka
    ia_b = ob[pos[hit]]  # b row for each a row that has one
    del pos, ob
    b_only = np.ones(len(b["sid"]), dtype=bool)
    b_only[ia_b] = False
    out: Table = {}
    for c in ("sid", "run", "lead", "valid"):
        out[c] = np.concatenate([a[c], b[c][b_only]])
    oi_a = a["oi"].copy()
    oi_a[hit & (oi_a < 0)] = b["oi"][ia_b][oi_a[hit] < 0]
    out["oi"] = np.concatenate([oi_a, b["oi"][b_only]])
    for var in FC_VARS:
        va = a[var]
        vb = np.full(len(va), np.nan, np.float32)
        vb[hit] = b[var][ia_b]
        if var == "wdir":
            m = _circ_mean(va, vb)
        else:
            with np.errstate(invalid="ignore"):
                m = np.where(
                    np.isfinite(va) & np.isfinite(vb),
                    (va + vb) / 2,
                    np.where(np.isfinite(va), va, vb),
                ).astype(np.float32)
        out[var] = np.concatenate([m, b[var][b_only]])
    return out


def build_models(
    raw: dict[str, Table],
    obs: Table,
    keep_from: int | None = None,
    on_table: Callable[[str, Table], None] | None = None,
    trim_from: int | None = None,
) -> tuple[dict[str, Table], dict]:
    """Raw tables -> all models (raw, *_bc, blend). Raw tables must have ``oi``.

    Corrections are learned from all raw pairs. Corrected tables (and the blend)
    are produced for rows with ``valid > keep_from`` (all rows if None).

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
        if m in ("ecmwf", "gfs"):
            tables[f"{m}_bc"], info[f"{m}_bc"] = corrected_table(T, obs, rows_after(T, keep_from))
        tables[m] = done(m, T)
        del T
    bl = blend_table(tables.get("ecmwf_bc"), tables.get("gfs_bc"))
    for m in ("ecmwf_bc", "gfs_bc"):
        if m in tables:
            tables[m] = done(m, tables[m])
    if bl is not None:
        tables["blend"] = done("blend", bl)
        del bl
    order = ("ecmwf", "gfs", "ecmwf_bc", "gfs_bc", "blend")
    return {m: tables[m] for m in order if m in tables}, info
