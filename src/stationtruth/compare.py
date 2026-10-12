"""Fair head-to-head model comparisons (docs/DATA.md, "compare.json").

Scoreboard numbers (``scores.json``) give every model its own sample: a model
with more runs, or one that skips hard hours (missing gust, missing SLP), is
scored on different weather than its rival. This module compares models only
on their **common sample** and puts an honest uncertainty on the difference.

Common sample
    A forecast–observation pair of model A and one of model B are paired when
    they have the same station, the same valid time and the same lead time
    (hence the same initialisation time, so both had the same information) and
    both are verifiable for the variable. Matching on the exact lead is
    stricter than matching on the lead bucket; it never pairs a 3 h forecast of
    one model with a 21 h forecast of the other. Both errors then refer to the
    same observation.

Uncertainty: block bootstrap
    Forecast errors are strongly correlated in space (one synoptic system spoils
    a whole region's forecast) and in time (a regime persists for hours to
    days), so the hundreds of thousands of pairs are far from independent and a
    textbook standard error would be much too narrow. Pairs are therefore
    grouped into blocks of **one 24-hour period (counted back from the build
    time) x one region** and whole blocks are resampled:

    * a region's CI resamples its days with replacement (as many days as it
      has with common data);
    * the global CI resamples days independently within every region
      (stratified by region, so each region keeps its weight) and adds the
      regions up. Regions are thousands of km apart, beyond the synoptic
      decorrelation length, so treating them as independent is reasonable,
      and it gives ~9x more blocks than resampling global days.

    Remaining approximation: errors on consecutive days are still somewhat
    correlated (regimes last a few days), which day blocks ignore, so the CIs
    are, if anything, a little optimistic. ``lag1`` in the output is the lag-1
    autocorrelation of the global daily MSE difference, to judge this.

    For speed the pairs are reduced once to sums per day x region x lead
    bucket (n, sum e_A^2, sum e_B^2, sum |e_A|, sum |e_B|); each of the
    ``RESAMPLES`` bootstrap replicates sums resampled blocks, never rows.
    95 % CIs are bootstrap percentiles. The random stream is fixed per pair,
    variable and window, so a rebuild on the same data gives the same intervals
    (also when models are added).

Per-station wins
    Among stations with at least ``STATION_MIN_N`` common pairs, the fraction
    where A has the lower RMSE (ties count half).

All-models common sample
    For each variable, the pairs verifiable for **every** included model, and
    each model's n/bias/MAE/RMSE on it: an apples-to-apples ranking. A model is
    included when it has at least ``COMMON_MIN_SHARE`` of the largest model's
    pair count in the window (a model that started yesterday would otherwise
    shrink everyone's sample to one day); excluded models are listed.

Errors are stored as int16 hundredths (forecast - observed, see
``stationtruth.verify``); the quantisation changes an RMSE by < 1e-4.
"""

from __future__ import annotations

import logging
import time
import zlib

import numpy as np

from .regions import REGIONS
from .verify import Table, errors_at, iso

log = logging.getLogger(__name__)

VARS: tuple[str, ...] = ("t", "td", "wspd", "gust", "mslp")
BUCKETS: tuple[str, ...] = ("0-24", "27-48", "51-72", "all")
WINDOWS: dict[str, int] = {"7d": 7, "30d": 30}
MAX_LEAD = 72
RESAMPLES = 2000
SEED = 20261012
CI = 0.95
STATION_MIN_N = 10
#: minimum common pairs and days with data before a CI is given
MIN_N = 30
MIN_DAYS = 3
COMMON_MIN_SHARE = 0.25
LAG1_MIN_DAYS = 10
FIELDS: tuple[str, ...] = (
    "n",
    "stations",
    "rmse_a",
    "rmse_b",
    "d_rmse",
    "d_rmse_lo",
    "d_rmse_hi",
    "mae_a",
    "mae_b",
    "d_mae",
    "d_mae_lo",
    "d_mae_hi",
    "frac_a_better",
    "days",
)
COMMON_FIELDS: tuple[str, ...] = ("n", "bias", "mae", "rmse")

_Q = 100.0  # errors stored in 1/_Q units
_NA = np.int16(-32768)
_NL = MAX_LEAD + 1
_NB = 3  # lead buckets without "all"


def lead_bucket(lead: np.ndarray) -> np.ndarray:
    """0-24 -> 0, 27-48 -> 1, 51-72 -> 2 (as stationtruth.correct)."""
    lead = lead.astype(np.int64)
    return np.where(lead <= 24, 0, np.where(lead <= 48, 1, 2))


def _r(x, nd: int) -> float | None:
    if x is None:
        return None
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, nd) + 0.0


# ------------------------------------------------------------------ bootstrap


def block_length(n_days: int) -> int:
    """Moving-block length in days: n^(1/3) rounded (1 day up to 3 days, 2 up to 15 ...)."""
    return max(1, int(round(n_days ** (1 / 3))))


def resample_sums(
    blocks: np.ndarray, rng: np.random.Generator, b: int = RESAMPLES, length: int = 1
) -> np.ndarray:
    """Moving-block bootstrap of summed per-day aggregates.

    ``blocks`` is ``(n_days, ...)`` in time order. Each of the ``b`` replicates
    is the sum of ``round(n_days / length)`` runs of ``length`` consecutive days,
    each run starting at a uniformly drawn day; returns ``(b, ...)``. With
    ``length=1`` this is the ordinary (iid) block bootstrap.
    """
    nb = len(blocks)
    if nb == 0:
        return np.zeros((b, *blocks.shape[1:]))
    length = max(1, min(length, nb))
    flat = blocks.reshape(nb, -1)
    cs = np.concatenate([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    runs = cs[length:] - cs[:-length]  # (nb - length + 1, ...) sums of each run
    ns = len(runs)
    k = max(1, int(round(nb / length)))
    idx = rng.integers(0, ns, size=(b, k))
    # how often each run is drawn per replicate, then one matrix product
    w = np.bincount((np.arange(b)[:, None] * ns + idx).ravel(), minlength=b * ns)
    return (w.reshape(b, ns).astype(np.float64) @ runs).reshape(b, *blocks.shape[1:])


def pair_stats(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d_rmse, d_mae) from sums ``[..., (n, ssa, ssb, saa, sab)]``; NaN where n == 0."""
    n = S[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        d_rmse = np.sqrt(S[..., 1] / n) - np.sqrt(S[..., 2] / n)
        d_mae = (S[..., 3] - S[..., 4]) / n
    d_rmse = np.where(n > 0, d_rmse, np.nan)
    d_mae = np.where(n > 0, d_mae, np.nan)
    return d_rmse, d_mae


def percentile_ci(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CI along axis 0 ignoring NaN replicates; NaN if > 5 % are NaN.

    Same as ``np.nanquantile`` (linear interpolation), vectorised over the
    other axes.
    """
    a = (1 - CI) / 2
    s = np.sort(boot, axis=0)  # NaN sort last
    cnt = np.isfinite(boot).sum(axis=0)
    out = []
    for q in (a, 1 - a):
        pos = q * np.maximum(cnt - 1, 0)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, np.maximum(cnt - 1, 0))
        f = pos - i0
        v0 = np.take_along_axis(s, i0[None], axis=0)[0]
        v1 = np.take_along_axis(s, i1[None], axis=0)[0]
        v = v0 + f * (v1 - v0)
        out.append(np.where(cnt >= 0.95 * len(boot), v, np.nan))
    return out[0], out[1]


def bootstrap_regions(
    A: np.ndarray, rng: np.random.Generator, b: int = RESAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """Region-stratified moving-block bootstrap over days.

    ``A``: ``(days, regions, ...)`` sums. Every region's days with data are
    resampled on their own (``resample_sums``, block length ``block_length``).
    Returns ``(boot, days)``: ``(b, regions, ...)`` replicate sums per region
    (sum over axis 1 = the global replicate) and the days with data per region.
    """
    nd, nr = A.shape[:2]
    out = np.zeros((b, *A.shape[1:]))
    days = np.zeros(nr, dtype=np.int64)
    for r in range(nr):
        Ar = A[:, r]
        act = np.flatnonzero(Ar[..., 0].reshape(nd, -1).sum(axis=1) > 0)
        days[r] = len(act)
        if len(act):
            out[:, r] = resample_sums(Ar[act[::-1]], rng, b, block_length(len(act)))
    return out, days


# ------------------------------------------------------------------ collector


class Comparer:
    """Collects compact per-model pairs (``add``), then compares all models (``result``).

    ``add(model, table)`` has the signature of ``correct.build_models``'
    ``on_table`` hook: it is called with each model's complete table (``oi``
    attached) and keeps, for pairs with lead <= 72 h verifying in the last 30
    days, a sorted int key (station, valid hour, lead) and int16 errors.
    """

    def __init__(
        self,
        obs: Table,
        region: np.ndarray,
        now: int,
        resamples: int = RESAMPLES,
        seed: int = SEED,
    ):
        self.obs, self.now = obs, int(now)
        self.region = np.asarray(region, dtype=np.int64)
        self.nreg = len(REGIONS) + 1  # slot 0 = unknown region
        self.days = max(WINDOWS.values())
        self.nh = self.days * 24 + 2
        self.base = self.now - self.days * 1440 - 60  # minutes; hour 0 of the key space
        self.resamples, self.seed = resamples, seed
        self.data: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        self.timings: dict[str, float] = {}

    # -- keys
    def _key(self, sid: np.ndarray, valid: np.ndarray, lead: np.ndarray) -> np.ndarray:
        hour = (valid.astype(np.int64) - self.base) // 60
        return (sid.astype(np.int64) * self.nh + hour) * _NL + lead.astype(np.int64)

    def _decode(self, key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """key -> (sid, age day 0..days-1, lead)."""
        key = key.astype(np.int64)
        lead = key % _NL
        t = key // _NL
        hour = t % self.nh
        sid = t // self.nh
        valid = self.base + hour * 60
        day = (self.now - valid) // 1440
        return sid, day, lead

    def add(self, model: str, T: Table | None) -> None:
        t0 = time.perf_counter()
        if T is None or len(T.get("sid", ())) == 0 or "oi" not in T:
            return
        rows = np.flatnonzero(T["oi"] >= 0)
        lead = T["lead"][rows].astype(np.int64)
        age = self.now - T["valid"][rows].astype(np.int64)
        keep = (lead >= 0) & (lead <= MAX_LEAD) & (age >= 0) & (age < self.days * 1440)
        keep &= T["sid"][rows] < len(self.region)
        rows = rows[keep]
        key = self._key(T["sid"][rows], T["valid"][rows], T["lead"][rows])
        order = np.argsort(key, kind="stable")
        key = key[order]
        rows = rows[order]
        first = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, bool)
        key, rows = key[first], rows[first]
        errs: dict[str, np.ndarray] = {}
        for v in VARS:
            if v not in T:
                continue
            e = errors_at(T, self.obs, v, rows).astype(np.float64)
            q = np.full(len(e), _NA, dtype=np.int16)
            ok = np.isfinite(e)
            q[ok] = np.clip(np.rint(e[ok] * _Q), -32767, 32767).astype(np.int16)
            if ok.any():
                errs[v] = q
        if key.size and key.max() < 2**31:
            key = key.astype(np.int32)
        self.data[model] = (key, errs)
        self.timings[f"add:{model}"] = round(time.perf_counter() - t0, 2)

    # -- result
    def result(self, order: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        models = [m for m in (order or []) if m in self.data]
        models += [m for m in self.data if m not in models]
        n_sid = len(self.region)
        reg_slot = np.clip(self.region + 1, 0, self.nreg - 1)
        sreg = reg_slot  # per sid

        out: dict = {
            "generated": iso(self.now),
            "models": models,
            "vars": list(VARS),
            "buckets": list(BUCKETS),
            "regions": list(REGIONS),
            "fields": list(FIELDS),
            "common_fields": list(COMMON_FIELDS),
            "method": {
                "match": "same station, valid time and lead time (same initialisation)",
                "block": "24 h period (counted back from the build time) x region",
                "global": "days resampled independently within each region (stratified)",
                "resamples": self.resamples,
                "ci": CI,
                "seed": self.seed,
                "station_min_n": STATION_MIN_N,
                "min_n": MIN_N,
                "min_days": MIN_DAYS,
                "block_days": "round(days ** (1/3)): 1 for <= 3 days, 2 for 4-15, 3 for 16-42",
                "common_min_share": COMMON_MIN_SHARE,
            },
            "windows": {
                w: {
                    "from": iso(self.now - d * 1440),
                    "to": iso(self.now),
                    "days": d,
                    "global": {},
                    "regions": {r: {} for r in REGIONS},
                    "common": {},
                    "lag1": {},
                }
                for w, d in WINDOWS.items()
            },
        }
        if not models:
            return out

        # union of keys over all models; each model's keys -> positions in it
        keys = [self.data[m][0] for m in models]
        allk = np.concatenate([k.astype(np.int64) for k in keys])
        allk.sort(kind="stable")  # merges the sorted runs
        U = allk[np.r_[True, allk[1:] != allk[:-1]]] if len(allk) else allk
        del allk
        pos = {m: np.searchsorted(U, self.data[m][0].astype(np.int64)) for m in models}
        sid, day, lead = self._decode(U)
        bucket = lead_bucket(lead)
        reg = sreg[sid]
        # block cell (day, region, bucket) and station cell (sid, week, bucket)
        cb = (day * self.nreg + reg) * _NB + bucket
        nbc = self.days * self.nreg * _NB
        cs = (sid * 2 + (day >= WINDOWS["7d"])) * _NB + bucket
        nsc = n_sid * 2 * _NB
        del lead

        for v in VARS:
            have = [m for m in models if v in self.data[m][1]]
            E: dict[str, np.ndarray] = {}
            for m in have:
                e = np.full(len(U), np.nan, dtype=np.float32)
                q = self.data[m][1][v]
                ok = q != _NA
                e[pos[m][ok]] = q[ok].astype(np.float32) / _Q
                E[m] = e
            fin = {m: np.isfinite(E[m]) for m in have}
            for i, a in enumerate(have):
                for b in have[i + 1 :]:
                    self._pair(out, a, b, v, E, fin, cb, nbc, cs, nsc, sreg, n_sid)
            self._common(out, have, v, E, fin, cb, nbc, day)
            del E, fin
        self.timings["result"] = round(time.perf_counter() - t0, 2)
        out["timings"] = self.timings
        return out

    def _pair(self, out, a, b, v, E, fin, cb, nbc, cs, nsc, sreg, n_sid) -> None:
        m = fin[a] & fin[b]
        if not m.any():
            return
        ea = E[a][m].astype(np.float64)
        eb = E[b][m].astype(np.float64)
        c = cb[m]
        A = np.stack(
            [
                np.bincount(c, minlength=nbc),
                np.bincount(c, weights=ea * ea, minlength=nbc),
                np.bincount(c, weights=eb * eb, minlength=nbc),
                np.bincount(c, weights=np.abs(ea), minlength=nbc),
                np.bincount(c, weights=np.abs(eb), minlength=nbc),
            ],
            axis=-1,
        ).reshape(self.days, self.nreg, _NB, 5)
        s = cs[m]
        St = np.stack(
            [
                np.bincount(s, minlength=nsc),
                np.bincount(s, weights=ea * ea, minlength=nsc),
                np.bincount(s, weights=eb * eb, minlength=nsc),
            ],
            axis=-1,
        ).reshape(n_sid, 2, _NB, 3)
        A = np.concatenate([A, A.sum(axis=2, keepdims=True)], axis=2)  # + "all" bucket
        St = np.concatenate([St, St.sum(axis=2, keepdims=True)], axis=2)
        St = np.cumsum(St, axis=1)  # [:, 0] = 7d, [:, 1] = 30d
        pk = f"{a}|{b}"
        for wi, (w, nd) in enumerate(WINDOWS.items()):
            win = out["windows"][w]
            Aw = A[:nd]
            if Aw[..., 0].sum() == 0:
                continue
            # own stream per pair/variable/window: adding a model never moves other CIs
            rng = np.random.default_rng([self.seed, zlib.crc32(f"{pk}|{v}|{w}".encode())])
            boot, days_r = bootstrap_regions(Aw, rng, self.resamples)
            # slot 0 = global (all regions incl. unknown), 1.. = REGIONS
            P = np.concatenate([Aw.sum(axis=(0, 1))[None], Aw.sum(axis=0)[1:]])
            B = np.concatenate([boot.sum(axis=1)[:, None], boot[:, 1:]], axis=1)
            d_rmse, d_mae = pair_stats(P)
            ci_r = percentile_ci(pair_stats(B)[0])
            ci_m = percentile_ci(pair_stats(B)[1])
            g_days = int((Aw[..., 0].reshape(nd, -1).sum(axis=1) > 0).sum())
            Sw = St[:, wi]  # (n_sid, buckets, 3)
            for slot in range(len(P)):
                if P[slot, 3, 0] <= 0:
                    continue
                rows = _entries(
                    P[slot],
                    d_rmse[slot],
                    d_mae[slot],
                    (ci_r[0][slot], ci_r[1][slot]),
                    (ci_m[0][slot], ci_m[1][slot]),
                    g_days if slot == 0 else int(days_r[slot]),
                    Sw if slot == 0 else Sw[sreg == slot],
                )
                dest = win["global"] if slot == 0 else win["regions"][REGIONS[slot - 1]]
                dest.setdefault(pk, {})[v] = rows
            # day-to-day persistence of the global MSE difference ("all" leads)
            g = Aw[:, :, 3].sum(axis=1)  # (days, 5)
            okd = g[:, 0] > 0
            if okd.sum() >= LAG1_MIN_DAYS:
                dm = ((g[okd, 1] - g[okd, 2]) / g[okd, 0])[::-1]  # chronological
                if np.std(dm[:-1]) > 0 and np.std(dm[1:]) > 0:
                    lag = float(np.corrcoef(dm[:-1], dm[1:])[0, 1])
                    win["lag1"].setdefault(pk, {})[v] = _r(lag, 2)

    def _common(self, out, have, v, E, fin, cb, nbc, day) -> None:
        for w, nd in WINDOWS.items():
            inwin = day < nd
            counts = {m: int((fin[m] & inwin).sum()) for m in have}
            top = max(counts.values(), default=0)
            if top == 0:
                continue
            inc = [m for m in have if counts[m] >= COMMON_MIN_SHARE * top]
            exc = [m for m in have if m not in inc]
            mask = inwin.copy()
            for m in inc:
                mask &= fin[m]
            block: dict = {"models": inc, "excluded": exc, "global": {}, "regions": {}}
            if mask.any():
                c = cb[mask]
                for m in inc:
                    e = E[m][mask].astype(np.float64)
                    S = np.stack(
                        [
                            np.bincount(c, minlength=nbc),
                            np.bincount(c, weights=e, minlength=nbc),
                            np.bincount(c, weights=np.abs(e), minlength=nbc),
                            np.bincount(c, weights=e * e, minlength=nbc),
                        ],
                        axis=-1,
                    ).reshape(self.days, self.nreg, _NB, 4)
                    S = S.sum(axis=0)  # (nreg, NB, 4)
                    S = np.concatenate([S, S.sum(axis=1, keepdims=True)], axis=1)
                    block["global"][m] = [_score(x) for x in S.sum(axis=0)]
                    for ri, rname in enumerate(REGIONS):
                        if S[ri + 1, 3, 0] > 0:
                            block["regions"].setdefault(rname, {})[m] = [
                                _score(x) for x in S[ri + 1]
                            ]
            out["windows"][w]["common"][v] = block


def _score(x: np.ndarray) -> list:
    n, s, sa, ss = (float(y) for y in x)
    if n <= 0:
        return [0, None, None, None]
    return [int(n), _r(s / n, 3), _r(sa / n, 3), _r(np.sqrt(ss / n), 3)]


def _entries(S, d_rmse, d_mae, ci_r, ci_m, days: int, St: np.ndarray) -> list:
    """One FIELDS row per bucket (None without pairs). ``S``: (buckets, 5) sums,
    ``d_*``/``ci_*``: per bucket, ``St``: (stations, buckets, 3) per-station sums."""
    rows = []
    ci_ok = days >= MIN_DAYS
    for k in range(S.shape[0]):
        n = float(S[k, 0])
        if n <= 0:
            rows.append(None)
            continue
        good = St[:, k, 0] >= STATION_MIN_N
        ns = int(good.sum())
        frac = None
        if ns:
            a_, b_ = St[good, k, 1], St[good, k, 2]
            frac = (float((a_ < b_).sum()) + 0.5 * float((a_ == b_).sum())) / ns
        ok = ci_ok and n >= MIN_N
        rows.append(
            [
                int(n),
                ns,
                _r(np.sqrt(S[k, 1] / n), 3),
                _r(np.sqrt(S[k, 2] / n), 3),
                _r(d_rmse[k], 3),
                _r(ci_r[0][k], 3) if ok else None,
                _r(ci_r[1][k], 3) if ok else None,
                _r(S[k, 3] / n, 3),
                _r(S[k, 4] / n, 3),
                _r(d_mae[k], 3),
                _r(ci_m[0][k], 3) if ok else None,
                _r(ci_m[1][k], 3) if ok else None,
                _r(frac, 3),
                days,
            ]
        )
    return rows


def compare_tables(
    tables: dict[str, Table],
    obs: Table,
    region: np.ndarray,
    now: int,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict:
    """Convenience: ``Comparer`` over ready tables (tests, scripts)."""
    c = Comparer(obs, region, now, resamples=resamples, seed=seed)
    for m, T in tables.items():
        c.add(m, T)
    return c.result(list(tables))
