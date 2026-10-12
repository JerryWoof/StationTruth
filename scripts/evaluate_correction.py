"""Out-of-sample evaluation of bias-correction schemes on a local data directory.

Usage::

    python scripts/evaluate_correction.py --data-dir data [--now 2026-10-12T18:00]
        [--json out.json] [--production]

For every raw model run R the correction of each candidate scheme is learned
exactly as production does it (docs/DATA.md, "Bias correction"): only verified
pairs with ``valid <= R``, ``run < R`` and ``valid > R - 14 d`` are used. The
corrected forecasts of run R are then scored against the observations, so every
score is out of sample. Rows that a scheme cannot correct keep the raw value
(same as production), hence every scheme is scored on exactly the same pairs.

Scores: RMSE / MAE / bias of corrected minus raw, per model x variable x lead
bucket (0-24, 27-48, 51-72 h) and per "history" bin = number of pairs ``n`` in
the station x lead-bucket x hour-bucket cell at correction time.

Candidates (est = estimate from the cell's pairs, w = n / (n + k)):

* ``raw``            no correction
* ``current``        cell mean if n >= 8, else nothing (pre-2026-10 production)
* ``shrink_k``       w * cell mean (shrink toward zero)
* ``poolA_k1_k2``    cell -> station (all lead/hour buckets) -> 0
* ``poolB_k1_k2``    cell -> station -> region x lead x hour bucket (``_rK``: region
                     mean shrunk with K instead of 20 pseudo-pairs)
* ``poolG_k1_k2``    as poolB with one global prior per lead x hour bucket
* ``poolC_k``        cell -> region x lead x hour bucket
* ``poolD_k1_k2_k3`` cell -> station x lead bucket -> station -> region
* ``median_k``       w * cell median (robust to outliers)
* ``medpoolB_k1_k2`` as poolB but with medians at cell and station level
* ``*_ratio``        wind speed / gust: multiplicative, factor = sum(obs)/sum(fcst)
                     pooled like the additive scheme (toward 1)
* ``*_tdclip``       additionally caps corrected td at corrected t

Pooled estimate: ``(S + k * prior) / (n + k)`` where S, n are the error sum and
count at that level, i.e. a pseudo-count (empirical-Bayes style) shrinkage.

``--production`` additionally runs ``stationtruth.correct.corrected_table`` and
scores it (``production``: with the region map, ``production_global``: without),
which must reproduce the rows of the scheme implemented there
(``poolB_10_5_tdclip`` and ``poolG_10_5_tdclip``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stationtruth import correct, verify

VARS = ("t", "td", "wspd", "gust", "mslp")
WIND = ("wspd", "gust")
WINDOW = 14 * 1440
NBINS = (0, 1, 8, 16, 32)  # history bins: n == 0, 1-7, 8-15, 16-31, >= 32
NBIN_LABEL = ("n=0", "1-7", "8-15", "16-31", "32+")
LB_LABEL = ("0-24h", "27-48h", "51-72h")
KS = (5, 10, 20, 40)
K2S = (1, 2, 5, 10, 20)
RATIO_CLIP = (1 / 3, 3.0)


def latest_obs_time(data_dir: Path) -> datetime:
    obs_files, _ = verify.list_files(data_dir)
    if not obs_files:
        sys.exit("no obs files")
    t = pq.read_table(obs_files[-1][1], columns=["time"]).column("time").to_pandas()
    return pd.Timestamp(t.max()).to_pydatetime().replace(second=0, microsecond=0)


def shrink(S, n, k, prior=0.0):
    """Pseudo-count shrinkage of a mean toward ``prior``: (S + k*prior) / (n + k)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (S + k * prior) / (n + k)


def wshrink(est, n, k, prior=0.0):
    """Same for a non-additive estimate (median, ratio): w*est + (1-w)*prior."""
    w = n / (n + k)
    return np.where(n > 0, w * np.nan_to_num(est) + (1 - w) * prior, prior)


class Levels:
    """Group ids of every forecast row at the pooling levels."""

    def __init__(self, T: verify.Table, region: np.ndarray):
        sid = T["sid"].astype(np.int64)
        lb = correct.lead_bucket(T["lead"])
        hb = correct.hour_bucket(T["valid"])
        self.lb = lb
        self.cell = correct.bucket_group(T)
        self.stn = sid
        self.stnlb = sid * correct.N_LEAD_BUCKETS + lb
        self.glob = lb * correct.N_HOUR_BUCKETS + hb
        reg = region[sid].astype(np.int64) + 1  # 0 = unknown
        self.reg = (reg * correct.N_LEAD_BUCKETS + lb) * correct.N_HOUR_BUCKETS + hb
        self.size = {
            "cell": int(self.cell.max()) + 1,
            "stn": int(sid.max()) + 1,
            "stnlb": int(self.stnlb.max()) + 1,
            "glob": correct.N_LEAD_BUCKETS * correct.N_HOUR_BUCKETS,
            "reg": int(self.reg.max()) + 1,
        }


def group_stats(gid, size, e, f, o):
    n = np.bincount(gid, minlength=size).astype(np.float64)
    return {
        "n": n,
        "S": np.bincount(gid, weights=e, minlength=size),
        "Sf": np.bincount(gid, weights=f, minlength=size),
        "So": np.bincount(gid, weights=o, minlength=size),
    }


def group_median(gid, size, e):
    out = np.full(size, np.nan)
    if len(gid):
        med = pd.Series(e).groupby(gid).median()
        out[med.index.to_numpy()] = med.to_numpy()
    return out


def candidates(var: str, st: dict) -> dict[str, tuple[str, np.ndarray]]:
    """name -> ("add", c) [corrected = raw - c] or ("mul", r) [corrected = raw * r].

    ``st`` holds stats gathered at the test rows: n_cell, S_cell, ... (arrays).
    """
    n_c, S_c = st["n_cell"], st["S_cell"]
    n_s, S_s = st["n_stn"], st["S_stn"]
    n_r, S_r = st["n_reg"], st["S_reg"]
    out: dict[str, tuple[str, np.ndarray]] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = S_c / n_c
    out["current"] = ("add", np.where(n_c >= 8, mean_c, 0.0))
    for k in KS:
        out[f"shrink_{k}"] = ("add", shrink(S_c, n_c, k))
    reg_prior = shrink(S_r, n_r, 20)  # region x lead x hour: large n, light shrink
    glob_prior = shrink(st["S_glob"], st["n_glob"], 20)
    for k1 in KS:
        for k2 in K2S:
            stnR = shrink(S_s, n_s, k2, reg_prior)
            out[f"poolB_{k1}_{k2}"] = ("add", shrink(S_c, n_c, k1, stnR))
        out[f"poolC_{k1}"] = ("add", shrink(S_c, n_c, k1, reg_prior))
    for k1, k2 in ((10, 2), (10, 5), (20, 5), (20, 20)):
        out[f"poolA_{k1}_{k2}"] = ("add", shrink(S_c, n_c, k1, shrink(S_s, n_s, k2)))
        stnG = shrink(S_s, n_s, k2, glob_prior)
        out[f"poolG_{k1}_{k2}"] = ("add", shrink(S_c, n_c, k1, stnG))
        for kr in (5, 100):
            stnR = shrink(S_s, n_s, k2, shrink(S_r, n_r, kr))
            out[f"poolB_{k1}_{k2}_r{kr}"] = ("add", shrink(S_c, n_c, k1, stnR))
    # cell -> station x lead bucket -> station -> region
    for k1, k2, k3 in ((10, 5, 2), (10, 10, 2), (10, 10, 5), (20, 10, 5), (5, 10, 2)):
        stnR = shrink(S_s, n_s, k3, reg_prior)
        slb = shrink(st["S_stnlb"], st["n_stnlb"], k2, stnR)
        out[f"poolD_{k1}_{k2}_{k3}"] = ("add", shrink(S_c, n_c, k1, slb))
    for k in KS:
        out[f"median_{k}"] = ("add", wshrink(st["med_cell"], n_c, k))
    for k1 in (10, 20):
        for k2 in (10, 20, 40):
            stnR = wshrink(st["med_stn"], n_s, k2, reg_prior)
            out[f"medpoolB_{k1}_{k2}"] = ("add", wshrink(st["med_cell"], n_c, k1, stnR))
    if var in WIND:
        with np.errstate(invalid="ignore", divide="ignore"):

            def ratio(So, Sf):
                r = np.where(Sf > 0, So / Sf, 1.0)
                return np.clip(np.nan_to_num(r, nan=1.0), *RATIO_CLIP)

            r_c = ratio(st["So_cell"], st["Sf_cell"])
            r_s = ratio(st["So_stn"], st["Sf_stn"])
            r_r = ratio(st["So_reg"], st["Sf_reg"])
        for k in KS:
            out[f"shrink_{k}_ratio"] = ("mul", wshrink(r_c, n_c, k, 1.0))
        reg_prior_r = wshrink(r_r, n_r, 20, 1.0)
        for k1 in (10, 20):
            for k2 in (2, 5, 10):
                stnR = wshrink(r_s, n_s, k2, reg_prior_r)
                out[f"poolB_{k1}_{k2}_ratio"] = ("mul", wshrink(r_c, n_c, k1, stnR))
    return out


class Acc:
    """Sums n, s, sa, ss per (candidate, var, lead bucket, n bin)."""

    def __init__(self):
        self.d: dict[tuple[str, str], np.ndarray] = {}

    def add(self, cand: str, var: str, lb: np.ndarray, nb: np.ndarray, e: np.ndarray):
        idx = lb * len(NBINS) + nb
        size = correct.N_LEAD_BUCKETS * len(NBINS)
        a = self.d.setdefault((cand, var), np.zeros((4, size)))
        a[0] += np.bincount(idx, minlength=size)
        a[1] += np.bincount(idx, weights=e, minlength=size)
        a[2] += np.bincount(idx, weights=np.abs(e), minlength=size)
        a[3] += np.bincount(idx, weights=e * e, minlength=size)

    def score(self, cand, var, lb=None, nb=None) -> dict:
        a = self.d[(cand, var)].reshape(4, correct.N_LEAD_BUCKETS, len(NBINS))
        if lb is not None:
            a = a[:, lb : lb + 1]
        if nb is not None:
            a = a[:, :, nb] if np.ndim(nb) == 0 else a[:, :, nb]
        n, s, sa, ss = (x.sum() for x in a)
        if n == 0:
            return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
        return {"n": int(n), "rmse": np.sqrt(ss / n), "mae": sa / n, "bias": s / n}


def evaluate_model(ds, model: str, acc: Acc, half_acc: list[Acc], production: bool) -> None:
    T = ds.fc[model]
    obs = ds.obs
    L = Levels(T, ds.region)
    runs = np.unique(T["run"])
    order = np.argsort(T["run"], kind="stable")
    bounds = np.searchsorted(T["run"][order], runs)
    bounds = np.append(bounds, len(order))
    pairs = {}
    for var in VARS:
        rows, e = verify.pair_errors(T, obs, var)
        f = T[var][rows].astype(np.float64)
        pairs[var] = {
            "rows": rows,
            "run": T["run"][rows].astype(np.int64),
            "valid": T["valid"][rows].astype(np.int64),
            "e": e.astype(np.float64),
            "f": f,
            "o": f - e,
        }
    prod = {}
    if production:
        prod["production"], _ = correct.corrected_table(T, obs, region=ds.region)
        prod["production_global"], _ = correct.corrected_table(T, obs)
    for ri, R in enumerate(runs.tolist()):
        test = order[bounds[ri] : bounds[ri + 1]]
        half = half_acc[0] if ri < len(runs) // 2 else half_acc[1]
        per_var = {}
        for var in VARS:
            P = pairs[var]
            tr = (P["valid"] <= R) & (P["run"] < R) & (P["valid"] > R - WINDOW)
            e, f, o, prow = P["e"][tr], P["f"][tr], P["o"][tr], P["rows"][tr]
            st = {}
            for lev in ("cell", "stnlb", "stn", "reg", "glob"):
                gid = getattr(L, lev)
                g = group_stats(gid[prow], L.size[lev], e, f, o)
                for key, arr in g.items():
                    st[f"{key}_{lev}"] = arr[gid[test]]
            for lev in ("cell", "stn"):
                gid = getattr(L, lev)
                st[f"med_{lev}"] = group_median(gid[prow], L.size[lev], e)[gid[test]]
            per_var[var] = (st, candidates(var, st))
        # corrected values of all candidates at the test rows
        raw = {v: T[v][test].astype(np.float64) for v in VARS}
        vals: dict[str, dict[str, np.ndarray]] = {}
        for var in VARS:
            vals[var] = {"raw": raw[var]}
            for name, (kind, c) in per_var[var][1].items():
                v = raw[var] * c if kind == "mul" else raw[var] - c
                vals[var][name] = np.maximum(v, 0.0) if var in correct.NONNEG else v
            for name, P in prod.items():
                vals[var][name] = P[var][test].astype(np.float64)
        # a wind-only (ratio) scheme is its additive base scheme for other variables
        for name in vals["wspd"]:
            if name.endswith("_ratio"):
                for var in VARS:
                    if var not in WIND:
                        vals[var][name] = vals[var][name.removesuffix("_ratio")]
        # td <= t variants (the other variables are unchanged)
        for name in [n for n in vals["td"] if n != "raw" and not n.startswith("production")]:
            for var in VARS:
                if var == "td":
                    with np.errstate(invalid="ignore"):
                        clip = np.fmin(vals["td"][name], vals["t"][name])
                    vals["td"][name + "_tdclip"] = np.where(
                        np.isnan(vals["t"][name]), vals["td"][name], clip
                    )
                else:
                    vals[var][name + "_tdclip"] = vals[var][name]
        oi = T["oi"][test]
        has = oi >= 0
        lb = L.lb[test]
        for var in VARS:
            ovals = np.full(len(test), np.nan)
            ovals[has] = obs[verify.OBS_COL[var]][oi[has]]
            nb = np.searchsorted(NBINS, per_var[var][0]["n_cell"], side="right") - 1
            ok = np.isfinite(raw[var] - ovals)
            for name, v in vals[var].items():
                err = (v - ovals)[ok]
                acc.add(name, var, lb[ok], nb[ok], err)
                half.add(name, var, lb[ok], nb[ok], err)


def fmt(x, nd=3):
    return "  nan" if not np.isfinite(x) else f"{x:.{nd}f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--now", help="ISO time (default: latest observation)")
    ap.add_argument("--json", type=Path, help="write all scores here")
    ap.add_argument("--production", action="store_true", help="also score correct.py")
    ap.add_argument("--top", type=int, default=12, help="candidates in the summary table")
    a = ap.parse_args(argv)
    t0 = time.time()
    now = (
        datetime.fromisoformat(a.now).replace(tzinfo=timezone.utc)
        if a.now
        else latest_obs_time(a.data_dir)
    )
    ds = verify.load(a.data_dir, now)
    for T in ds.fc.values():
        verify.match_table(T, ds.obs)
    results: dict[str, Acc] = {}
    halves: dict[str, list[Acc]] = {}
    for model in verify.RAW_MODELS:
        if model not in ds.fc:
            continue
        results[model] = Acc()
        halves[model] = [Acc(), Acc()]
        evaluate_model(ds, model, results[model], halves[model], a.production)
    print(f"# data {a.data_dir}, now {now:%Y-%m-%dT%H:%MZ}, {time.time() - t0:.0f} s\n")
    report(results, halves, a.top, a.json)


def report(results: dict[str, Acc], halves: dict[str, list[Acc]], top: int, out: Path | None):
    models = list(results)
    cands = sorted({c for acc in results.values() for c, _ in acc.d})
    # skill = 1 - rmse/rmse_raw per (model, var); rank by mean skill, penalise worst case
    rows = []
    for c in cands:
        sk, worst_low = [], []
        for m in models:
            for v in VARS:
                if (c, v) not in results[m].d:
                    break
                r0 = results[m].score("raw", v)["rmse"]
                sk.append(1 - results[m].score(c, v)["rmse"] / r0)
                for nb in range(len(NBINS)):
                    s0 = results[m].score("raw", v, nb=nb)
                    if s0["n"] > 1000:
                        worst_low.append(1 - results[m].score(c, v, nb=nb)["rmse"] / s0["rmse"])
            else:
                continue
            break
        else:
            rows.append((c, float(np.mean(sk)), float(np.min(sk)), float(np.min(worst_low))))
    rows.sort(key=lambda r: -r[1])
    print("## Ranking (RMSE skill vs raw, % ; mean / worst model-variable / worst n-bin)\n")
    print(f"| {'scheme':<24} | mean | worst var | worst n-bin |")
    print("|---|---:|---:|---:|")
    fixed = ("raw", "current", "production", "production_global")

    def family(c: str) -> str:
        f = c.split("_")[0]
        return f + ("_ratio" if "_ratio" in c else "") + ("_tdclip" if "_tdclip" in c else "")

    best: dict[str, str] = {}
    for c, *_ in rows:  # sorted: first seen = best of its family
        best.setdefault(family(c), c)
    keep = set(c for c, *_ in rows[:top]) | set(best.values()) | set(fixed)
    shown = [r for r in rows if r[0] in keep]
    for c, mean, worst, wl in shown:
        print(f"| {c:<24} | {100 * mean:5.2f} | {100 * worst:5.2f} | {100 * wl:5.2f} |")
    focus = [
        c
        for c in (
            "raw",
            "current",
            "shrink_10",
            "shrink_20",
            "median_20",
            "poolC_20",
            "poolA_20_20",
            best.get("poolB_ratio", ""),
            rows[0][0],
            "production",
            "production_global",
        )
        if c in cands
    ]
    focus = list(dict.fromkeys(focus))
    print("\n## RMSE per model x variable (all leads)\n")
    print("| scheme | " + " | ".join(f"{m} {v}" for m in models for v in VARS) + " |")
    print("|---|" + "---:|" * (len(models) * len(VARS)))
    for c in focus:
        cells = [fmt(results[m].score(c, v)["rmse"]) for m in models for v in VARS]
        print(f"| {c} | " + " | ".join(cells) + " |")
    print("\n## Bias per model x variable (all leads)\n")
    print("| scheme | " + " | ".join(f"{m} {v}" for m in models for v in VARS) + " |")
    print("|---|" + "---:|" * (len(models) * len(VARS)))
    for c in focus:
        cells = [fmt(results[m].score(c, v)["bias"], 2) for m in models for v in VARS]
        print(f"| {c} | " + " | ".join(cells) + " |")
    for m in models:
        print(f"\n## {m}: RMSE by lead bucket and by cell history n (pairs available)\n")
        cols = [(f"{lab}", dict(lb=i)) for i, lab in enumerate(LB_LABEL)] + [
            (f"{lab}", dict(nb=i)) for i, lab in enumerate(NBIN_LABEL)
        ]
        for v in VARS:
            print(
                f"\n**{v}** (n pairs: "
                + ", ".join(f"{lab} {results[m].score('raw', v, **kw)['n']}" for lab, kw in cols)
                + ")\n"
            )
            print("| scheme | " + " | ".join(lab for lab, _ in cols) + " |")
            print("|---|" + "---:|" * len(cols))
            for c in focus:
                cells = [fmt(results[m].score(c, v, **kw)["rmse"]) for _, kw in cols]
                print(f"| {c} | " + " | ".join(cells) + " |")
    print("\n## Stability: RMSE skill % in first vs second half of runs\n")
    print("| scheme | " + " | ".join(f"{m} {v}" for m in models for v in VARS) + " |")
    print("|---|" + "---:|" * (len(models) * len(VARS)))
    for c in focus:
        cells = []
        for m in models:
            for v in VARS:
                h = [
                    100 * (1 - acc.score(c, v)["rmse"] / acc.score("raw", v)["rmse"])
                    for acc in halves[m]
                ]
                cells.append(f"{h[0]:.1f}/{h[1]:.1f}")
        print(f"| {c} | " + " | ".join(cells) + " |")
    if out is not None:
        dump = {
            m: {
                c: {
                    v: {
                        "all": results[m].score(c, v),
                        "lead": [results[m].score(c, v, lb=i) for i in range(3)],
                        "nbin": [results[m].score(c, v, nb=i) for i in range(len(NBINS))],
                    }
                    for v in VARS
                    if (c, v) in results[m].d
                }
                for c in cands
            }
            for m in models
        }
        out.write_text(json.dumps(dump, default=float, indent=1))


if __name__ == "__main__":
    main()
