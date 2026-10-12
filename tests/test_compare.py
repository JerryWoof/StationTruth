from __future__ import annotations

import json
from datetime import timezone

import numpy as np
import pytest
from synth import make_dataset

from stationtruth import compare, sitegen
from stationtruth.compare import (
    FIELDS,
    Comparer,
    block_length,
    bootstrap_regions,
    pair_stats,
    percentile_ci,
    resample_sums,
)
from stationtruth.regions import REGIONS

UTC = timezone.utc
F = {f: i for i, f in enumerate(FIELDS)}
NOW = 29_000_000  # epoch minutes, a multiple of 60 (2025-02-20)


# ------------------------------------------------------------------ bootstrap


def test_block_length():
    assert [block_length(n) for n in (1, 3, 4, 7, 15, 16, 30)] == [1, 1, 2, 2, 2, 3, 3]


def test_resample_sums_shapes_and_totals():
    rng = np.random.default_rng(0)
    blocks = np.arange(10, dtype=float).reshape(5, 2)
    for length in (1, 2):
        out = resample_sums(blocks, rng, b=300, length=length)
        assert out.shape == (300, 2)
        # every replicate sums round(5 / length) runs of `length` days
        k = round(5 / length)
        assert np.all(out[:, 0] % 2 == 0)  # sums of even numbers
        assert out[:, 0].min() >= 0 and out[:, 0].max() <= 8 * k * length
    # iid case: the mean replicate equals the total
    out = resample_sums(blocks, rng, b=4000, length=1)
    assert out.mean(axis=0) == pytest.approx(blocks.sum(axis=0), rel=0.03)


def test_percentile_ci_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 3, 4))
    x[:50, 0, 0] = np.nan  # 2.5 % NaN: still a CI
    x[:300, 1, 1] = np.nan  # 15 % NaN: no CI
    lo, hi = percentile_ci(x)
    assert np.isnan(lo[1, 1]) and np.isnan(hi[1, 1])
    ok = ~np.isnan(lo)
    with np.errstate(all="ignore"):
        assert np.allclose(lo[ok], np.nanquantile(x, 0.025, axis=0)[ok])
        assert np.allclose(hi[ok], np.nanquantile(x, 0.975, axis=0)[ok])


def _synthetic_blocks(rng, days=20, regions=4, n=400, sa=1.0, sb=1.3, sd_day=0.8, rho=0.5):
    """Day x region sums of squared/absolute errors for two models.

    Both models share an AR(1) day-and-region error component (correlated
    errors, like weather) plus independent noise of sd ``sa`` / ``sb``.
    """
    A = np.zeros((days, regions, 1, 5))
    for r in range(regions):
        common = np.zeros(days)
        x = rng.normal()
        for d in range(days):
            x = rho * x + np.sqrt(1 - rho**2) * rng.normal()
            common[d] = sd_day * x
        for d in range(days):
            ea = common[d] + rng.normal(0, sa, n)
            eb = common[d] + rng.normal(0, sb, n)
            A[d, r, 0] = [n, (ea**2).sum(), (eb**2).sum(), np.abs(ea).sum(), np.abs(eb).sum()]
    return A


def test_bootstrap_ci_covers_known_difference():
    """95 % CIs of dRMSE and dMAE contain the true difference in most replications."""
    sa, sb, sd_day = 1.0, 1.3, 0.8
    # population values: common part has variance sd_day^2 (stationary AR(1))
    true_d_rmse = np.sqrt(sa**2 + sd_day**2) - np.sqrt(sb**2 + sd_day**2)
    true_d_mae = np.sqrt(2 / np.pi) * (np.sqrt(sa**2 + sd_day**2) - np.sqrt(sb**2 + sd_day**2))
    rng = np.random.default_rng(42)
    hits_r = hits_m = 0
    reps = 60
    for _ in range(reps):
        A = _synthetic_blocks(rng, sa=sa, sb=sb, sd_day=sd_day)
        boot, days = bootstrap_regions(A, rng, b=500)
        assert (days == 20).all()
        d_r, d_m = pair_stats(boot.sum(axis=1))
        lo_r, hi_r = percentile_ci(d_r)
        lo_m, hi_m = percentile_ci(d_m)
        hits_r += lo_r[0] <= true_d_rmse <= hi_r[0]
        hits_m += lo_m[0] <= true_d_mae <= hi_m[0]
        assert hi_r[0] < 0  # the difference is clearly significant
    assert hits_r / reps >= 0.8
    assert hits_m / reps >= 0.8


def test_bootstrap_ci_no_difference_straddles_zero():
    rng = np.random.default_rng(7)
    A = _synthetic_blocks(rng, sa=1.0, sb=1.0)
    boot, _ = bootstrap_regions(A, rng, b=1000)
    lo, hi = percentile_ci(pair_stats(boot.sum(axis=1))[0])
    assert lo[0] < 0 < hi[0]


# ------------------------------------------------------------------ common sample


def _tables(n_sid=12, days=10, seed=0):
    """Obs (all zero) + two model tables whose forecast value is the error."""
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(NOW - days * 1440, NOW, 720):
        for lead in range(0, 73, 3):
            valid = run + lead * 60
            if valid > NOW:
                continue
            for s in range(n_sid):
                rows.append((s, run, lead, valid))
    a = np.array(rows, dtype=np.int64)
    keys = np.unique(a[:, 0] * 10**9 + a[:, 3])
    o_sid, o_time = keys // 10**9, keys % 10**9
    n_obs = len(keys)
    obs = {v: np.zeros(n_obs, np.float32) for v in ("t", "td", "wspd", "gust", "slp", "wdir")}
    obs["wspd"][:] = 10.0
    obs["sid"], obs["time"] = o_sid.astype(np.int32), o_time

    def table(sd, bias=0.0):
        n = len(a)
        T = {
            "sid": a[:, 0].astype(np.int32),
            "run": a[:, 1].astype(np.int32),
            "lead": a[:, 2].astype(np.int16),
            "valid": a[:, 3].astype(np.int32),
            "oi": np.searchsorted(keys, a[:, 0] * 10**9 + a[:, 3]).astype(np.int32),
        }
        for v in ("t", "td", "gust", "mslp", "wdir"):
            T[v] = (bias + rng.normal(0, sd, n)).astype(np.float32)
        T["wspd"] = (10.0 + bias + rng.normal(0, sd, n)).astype(np.float32)
        return T

    return obs, table


def test_common_sample_filtering():
    obs, table = _tables()
    A, B = table(1.0), table(2.0)
    n = len(A["sid"])
    # B misses some rows (NaN t), lacks a station entirely and has extra leads > 72
    B["t"][::7] = np.nan
    gone = B["sid"] == 3
    B = {k: v[~gone] for k, v in B.items()}
    A["oi"][::11] = -1  # A unmatched rows
    A["lead"][A["lead"] == 72] = 75  # A: lead outside the compared range
    region = np.array([i % 3 for i in range(12)], dtype=np.int8)  # Europe, N Am, C Am
    c = Comparer(obs, region, NOW, resamples=200)
    c.add("a", A)
    c.add("b", B)
    res = c.result(["a", "b"])
    row = res["windows"]["30d"]["global"]["a|b"]["t"][3]  # "all" leads
    # expected common sample, by hand
    ka = {
        (s, v, ld): A["t"][i]
        for i, (s, v, ld) in enumerate(zip(A["sid"], A["valid"], A["lead"], strict=True))
        if A["oi"][i] >= 0 and ld <= 72 and NOW - v < 30 * 1440
    }
    kb = {
        (s, v, ld): B["t"][i]
        for i, (s, v, ld) in enumerate(zip(B["sid"], B["valid"], B["lead"], strict=True))
        if np.isfinite(B["t"][i])
    }
    common = sorted(set(ka) & set(kb))
    assert row[F["n"]] == len(common) < n
    ea = np.array([ka[k] for k in common], dtype=np.float64)
    eb = np.array([kb[k] for k in common], dtype=np.float64)
    assert row[F["rmse_a"]] == pytest.approx(np.sqrt((ea**2).mean()), abs=2e-3)
    assert row[F["rmse_b"]] == pytest.approx(np.sqrt((eb**2).mean()), abs=2e-3)
    assert row[F["d_mae"]] == pytest.approx(np.abs(ea).mean() - np.abs(eb).mean(), abs=2e-3)
    assert row[F["d_rmse_lo"]] <= row[F["d_rmse"]] <= row[F["d_rmse_hi"]] < 0
    assert row[F["stations"]] == 11  # station 3 has no common pairs
    assert row[F["frac_a_better"]] == 1.0
    assert row[F["days"]] == 11  # valid times span NOW - 10 d .. NOW
    # regions add up to the global sample
    by_reg = [res["windows"]["30d"]["regions"][r]["a|b"]["t"][3][F["n"]] for r in REGIONS[:3]]
    assert sum(by_reg) == row[F["n"]]
    assert "a|b" not in res["windows"]["30d"]["regions"]["Asia"]
    # buckets add up to "all"
    rows = res["windows"]["30d"]["global"]["a|b"]["t"]
    assert sum(r[F["n"]] for r in rows[:3]) == row[F["n"]]
    # the 7d window is a subset
    assert 0 < res["windows"]["7d"]["global"]["a|b"]["t"][3][F["n"]] < row[F["n"]]
    # all-models common sample: both models scored on the same pairs
    cm = res["windows"]["30d"]["common"]["t"]
    assert cm["models"] == ["a", "b"] and cm["excluded"] == []
    assert cm["global"]["a"][3][0] == cm["global"]["b"][3][0] == len(common)
    assert cm["global"]["a"][3][3] == pytest.approx(row[F["rmse_a"]], abs=1e-3)


def test_common_sample_excludes_sparse_model_and_is_deterministic():
    obs, table = _tables(seed=3)
    A, B, C = table(1.0), table(1.5), table(1.2)
    late = C["valid"] < NOW - 1440  # C only has the last day: < 25 % of the others
    C = {k: v[~late] for k, v in C.items()}
    region = np.zeros(12, dtype=np.int8)
    tabs = {"a": A, "b": B, "c": C}
    r1 = compare.compare_tables(tabs, obs, region, NOW, resamples=300)
    r2 = compare.compare_tables(tabs, obs, region, NOW, resamples=300)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    cm = r1["windows"]["30d"]["common"]["t"]
    assert cm["models"] == ["a", "b"] and cm["excluded"] == ["c"]
    # c vs a is still compared head to head, on c's one day: too few days for a CI
    row = r1["windows"]["30d"]["global"]["a|c"]["t"][3]
    assert row[F["days"]] <= 2 and row[F["d_rmse_lo"]] is None
    assert set(r1["windows"]["30d"]["global"]) == {"a|b", "a|c", "b|c"}


# ------------------------------------------------------------------ sitegen


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("cmp")
    return make_dataset(d / "data", n_stations=8, days=12, seed=11), d


def test_build_writes_compare_and_health(synth):
    info, d = synth
    out = d / "out"
    sitegen.build(info.data_dir, out, now=info.now)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["optional_errors"] == {}
    res = json.loads((out / "compare.json").read_text())
    assert res["models"] == meta["models"]
    w = res["windows"]["7d"]
    row = w["global"]["ecmwf|gfs"]["t"][3]
    # synth: ECMWF (bias +0.5, noise 0.6) beats GFS (bias -1.0, noise 0.8)
    assert row[F["d_rmse"]] < 0 and row[F["d_rmse_hi"]] < 0
    assert row[F["frac_a_better"]] > 0.5
    assert set(w["common"]["t"]["global"]) == set(meta["models"])
    health = json.loads((out / "health.json").read_text())
    assert health["status"] in ("ok", "degraded", "down")
    assert set(health["runs"]["models"]) == {"ecmwf", "gfs"}
    assert health["build"]["timings"]["compare"] >= 0


def test_compare_failure_does_not_break_build(synth, monkeypatch):
    info, d = synth
    out = d / "out_fail"

    def boom(self, order=None):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(compare.Comparer, "result", boom)
    sitegen.build(info.data_dir, out, now=info.now)
    meta = json.loads((out / "meta.json").read_text())
    assert "synthetic failure" in meta["optional_errors"]["compare"]
    assert not (out / "compare.json").exists()
    assert (out / "scores.json").exists() and (out / "latest.json").exists()
    health = json.loads((out / "health.json").read_text())
    assert any(r["component"] == "build" for r in health["reasons"])
