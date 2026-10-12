from __future__ import annotations

import json

import numpy as np
import pytest
from synth import make_dataset

from stationtruth import correct, sitegen, verify


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthc")
    info = make_dataset(d / "data", n_stations=6, days=20, seed=3)
    return info


def _tables(info, mutate_obs=None, use_region=False):
    ds = verify.load(info.data_dir, info.now)
    if mutate_obs is not None:
        mutate_obs(ds.obs)
    for T in ds.fc.values():
        verify.match_table(T, ds.obs)
    region = ds.region if use_region else None
    tables, _ = correct.build_models(ds.fc, ds.obs, region=region)
    return ds, tables


def test_correction_reduces_error(synth, tmp_path):
    sitegen.build(synth.data_dir, tmp_path, now=synth.now)
    g = json.loads((tmp_path / "scores.json").read_text())["windows"]["7d"]["global"]
    for m in ("ecmwf", "gfs"):
        for v in ("t", "td", "wspd", "mslp"):
            raw, bc = g[m][v]["all"], g[f"{m}_bc"][v]["all"]
            assert bc["n"] == raw["n"]
            assert bc["rmse"] < raw["rmse"], (m, v)
            # capping td at t only removes positive td errors: allow a little more
            assert abs(bc["bias"]) < (0.25 if v == "td" else 0.15), (m, v)
        assert g[f"{m}_bc"]["wdir"] == g[m]["wdir"]  # direction is not corrected
    assert g["blend"]["t"]["all"]["rmse"] < min(g["ecmwf_bc"]["t"]["all"]["rmse"], 1e9)


@pytest.mark.parametrize("use_region", [False, True])
def test_no_look_ahead(synth, use_region):
    """Corrections for run R must not change when pairs with valid > R change."""
    ds0, t0 = _tables(synth, use_region=use_region)
    R = sorted(set(t0["ecmwf"]["run"].tolist()))[len(set(t0["ecmwf"]["run"].tolist())) // 2]

    def poison(obs):
        late = obs["time"] > R
        for v in ("t", "td", "wspd", "gust", "slp"):
            obs[v][late] += np.float32(7.0)

    ds1, t1 = _tables(synth, poison, use_region)
    for m in ("ecmwf_bc", "gfs_bc", "blend"):
        a, b = t0[m], t1[m]
        upto = a["run"] <= R
        assert upto.sum() > 0
        for v in ("t", "td", "wspd", "gust", "mslp"):
            np.testing.assert_array_equal(a[v][upto], b[v][upto], err_msg=f"{m} {v}")
        # sanity: the poison does change later corrections, so the test has teeth
        later = a["run"] > R + 3 * 1440
        assert not np.allclose(a["t"][later], b["t"][later])


def _single_station_table(n_runs: int):
    """Daily 00Z runs, lead 0 only; error of run k is exactly k."""
    day = 1440
    base = verify.to_minutes(synth_start())
    run = base + np.arange(n_runs) * day
    T = {
        "sid": np.zeros(n_runs, np.int32),
        "run": run.astype(np.int32),
        "lead": np.zeros(n_runs, np.int16),
        "valid": run.astype(np.int32),
        "t": np.arange(n_runs, dtype=np.float32) + 10,
        "oi": np.arange(n_runs, dtype=np.int32),
    }
    for v in ("td", "wspd", "gust", "wdir", "mslp"):
        T[v] = np.full(n_runs, np.nan, np.float32)
    obs = {v: np.full(n_runs, np.nan, np.float32) for v in verify.OBS_VARS}
    obs["t"] = np.full(n_runs, 10.0, np.float32)
    return T, obs


def synth_start():
    from synth import START

    return START


def _expected_single_station(errors: list[float]) -> float:
    """Three-level estimate when cell, station and region hold the same pairs."""
    S, n = float(np.sum(errors)), len(errors)
    b = S / (n + correct.K_REGION)
    b = (S + correct.K_STATION * b) / (n + correct.K_STATION)
    return (S + correct.K_CELL * b) / (n + correct.K_CELL)


def test_window_and_strictly_earlier_runs():
    T, obs = _single_station_table(30)
    corr, n = correct.correction(T, obs, "t")
    # run k sees errors of runs j < k (its own lead-0 pair is not available) within 14 days
    assert n[:14].tolist() == list(range(14))
    assert np.isnan(corr[0])  # nothing known at all -> no correction
    for k in (1, 2, 8, 13):
        assert corr[k] == pytest.approx(_expected_single_station(list(range(k))), rel=1e-5)
    assert n[20] == 13
    assert corr[20] == pytest.approx(_expected_single_station(list(range(7, 20))), rel=1e-5)


def test_shrinkage_small_n():
    """Few pairs move the correction only a little; many pairs give the plain mean."""
    n_runs = 60
    T, obs = _single_station_table(n_runs)
    T["t"] = np.full(n_runs, 13.0, np.float32)  # constant +3 error
    corr, n = correct.correction(T, obs, "t")
    # one pair: well below the observed error, and growing monotonically with n
    assert 0 < corr[1] < 0.5 * 3.0
    assert (np.diff(corr[1:14]) > 0).all()
    # window full (13-14 pairs): most of the way to the mean
    assert corr[20] == pytest.approx(_expected_single_station([3.0] * 13), rel=1e-5)
    assert corr[20] > 0.6 * 3.0
    # one wild first pair (a gust-spike-like outlier) yields only a fraction of it, even
    # here where the "region" is this one station (in reality it holds many)
    T["t"][0] = 40.0
    corr2, _ = correct.correction(T, obs, "t")
    assert corr2[1] == pytest.approx(_expected_single_station([30.0]), rel=1e-5)
    assert corr2[1] < 30.0 / 3


def _two_level_table():
    """Stations 0, 1 (region 0) and 2 (region 1); daily runs, leads 0 and 60 h.

    Station 0 verifies only at lead 0 (error +2), station 2 at lead 0 (error -4),
    station 1 never verifies. Leads 60 h are never matched.
    """
    day = 1440
    base = verify.to_minutes(synth_start())
    rows = []
    for k in range(10):
        for s in (0, 1, 2):
            for lead in (0, 60):
                rows.append((s, base + k * day, lead))
    sid, run, lead = (np.array(x) for x in zip(*rows, strict=True))
    n = len(sid)
    valid = run + lead * 60
    T = {
        "sid": sid.astype(np.int32),
        "run": run.astype(np.int32),
        "lead": lead.astype(np.int16),
        "valid": valid.astype(np.int32),
        "t": np.where(sid == 0, 12.0, np.where(sid == 2, 6.0, 10.0)).astype(np.float32),
        "oi": np.where((lead == 0) & (sid != 1), np.arange(n), -1).astype(np.int32),
    }
    for v in ("td", "wspd", "gust", "wdir", "mslp"):
        T[v] = np.full(n, np.nan, np.float32)
    obs = {v: np.full(n, np.nan, np.float32) for v in verify.OBS_VARS}
    obs["t"] = np.full(n, 10.0, np.float32)
    return T, obs


def test_fallback_station_then_region():
    T, obs = _two_level_table()
    region = np.array([0, 0, 1], np.int8)
    corr, n = correct.correction(T, obs, "t", region=region)
    last = T["run"] == T["run"].max()  # 9 earlier runs verified
    k0 = (T["sid"] == 0) & last
    # station 0, lead 0: own cell (9 pairs of +2) -> close to +2
    c00 = corr[k0 & (T["lead"] == 0)][0]
    assert n[k0 & (T["lead"] == 0)][0] == 9 and 1.5 < c00 < 2.0
    # station 0, lead 60 h: empty cell, falls back to the station's own bias (whose
    # regional prior for that bucket is empty, i.e. 0): 18 / (9 + K_STATION)
    c060 = corr[k0 & (T["lead"] == 60)][0]
    assert n[k0 & (T["lead"] == 60)][0] == 0
    assert c060 == pytest.approx(18.0 / (9 + correct.K_STATION), rel=1e-5)
    # station 1 (never verified): the region 0 prior for its lead/hour bucket, which
    # only holds station 0's lead-0 pairs -> positive; nothing for lead 60 h
    k1 = (T["sid"] == 1) & last
    r_l0 = 18.0 / (9 + correct.K_REGION)
    assert corr[k1 & (T["lead"] == 0)][0] == pytest.approx(r_l0, rel=1e-5)
    assert np.isnan(corr[k1 & (T["lead"] == 60)][0])
    # station 2 lives in another region and never leaks into region 0
    k2 = (T["sid"] == 2) & last
    assert corr[k2 & (T["lead"] == 0)][0] < -3.0
    # without a region map the prior is global: station 1 sees both stations
    g, _ = correct.correction(T, obs, "t")
    assert g[k1 & (T["lead"] == 0)][0] == pytest.approx(
        (18.0 - 36.0) / (18 + correct.K_REGION), rel=1e-5
    )


def test_corrected_values_and_clipping():
    T, obs = _single_station_table(12)
    T["wspd"] = np.full(12, 1.0, np.float32)
    obs["wspd"] = np.zeros(12, np.float32)
    T["wspd"][:] = 5.0  # +5 kt bias ...
    T["wspd"][11] = 2.0  # ... so 2 kt corrected by about -4.5 kt must clip to 0
    T["td"] = np.full(12, 11.0, np.float32)
    obs["td"] = np.full(12, 9.0, np.float32)  # +2 td bias ...
    T["td"][11] = 22.0  # ... and a raw td far above t must be capped at corrected t
    bc, applied = correct.corrected_table(T, obs)
    assert np.isfinite(bc["t"][1:]).all() and bc["t"][0] == T["t"][0]
    assert bc["t"][11] == pytest.approx(
        T["t"][11] - _expected_single_station(list(range(11))), rel=1e-5
    )
    assert bc["wspd"][11] == 0 and (bc["wspd"] >= 0).all()
    assert bc["td"][11] == bc["t"][11]
    assert (bc["td"] <= bc["t"]).all()
    assert bc["wdir"] is T["wdir"]
    assert applied["t"] == pytest.approx(11 / 12)


def test_blend_mean_fallback_and_circular():
    def tab(sid, wdir, t):
        n = len(sid)
        T = {
            "sid": np.array(sid, np.int32),
            "run": np.full(n, 1000, np.int32),
            "lead": np.zeros(n, np.int16),
            "valid": np.full(n, 1000, np.int32),
            "oi": np.arange(n, dtype=np.int32),
            "wdir": np.array(wdir, np.float32),
            "t": np.array(t, np.float32),
        }
        for v in ("td", "wspd", "gust", "mslp"):
            T[v] = np.full(n, np.nan, np.float32)
        return T

    b = correct.blend_table(tab([0, 1], [350, 90], [10, 20]), tab([0, 2], [10, 180], [12, 5]))
    assert b["sid"].tolist() == [0, 1, 2]
    assert b["t"].tolist() == [11, 20, 5]
    assert b["wdir"][0] == pytest.approx(0, abs=1e-3) or b["wdir"][0] == pytest.approx(360)
    assert b["wdir"][1:] == pytest.approx([90, 180], abs=1e-3)
    assert correct.blend_table(None, None) is None


def _runs_table(runs_h: list[int], value_of_run, leads=range(0, 73, 3)) -> dict:
    """One station; t = value_of_run(run hour) + lead/100 so components are identifiable."""
    sid, run, lead = [], [], []
    for r in runs_h:
        for ld in leads:
            sid.append(0)
            run.append(r * 60)
            lead.append(ld)
    run = np.array(run, np.int32)
    lead = np.array(lead, np.int16)
    T = {
        "sid": np.array(sid, np.int32),
        "run": run,
        "lead": lead,
        "valid": (run + lead.astype(np.int32) * 60).astype(np.int32),
        "oi": np.full(len(run), -1, np.int32),
        "t": np.array(
            [value_of_run(r // 60) + ld / 100 for r, ld in zip(run, lead, strict=True)], np.float32
        ),
        "wdir": np.full(len(run), 90, np.float32),
    }
    for v in ("td", "wspd", "gust", "mslp"):
        T[v] = np.full(len(run), np.nan, np.float32)
    return T


def _blend_rows(b, run_h):
    m = b["run"] == run_h * 60
    order = np.argsort(b["lead"][m])
    return b["lead"][m][order], b["valid"][m][order], b["t"][m][order]


def test_blend_uses_older_ecmwf_run_when_same_time_missing():
    base = 1000  # hours
    ec = _runs_table([base], lambda r: 10.0)  # ECMWF 00Z only
    gf = _runs_table([base, base + 12], lambda r: 20.0 if r == base else 30.0)  # GFS 00Z + 12Z
    b = correct.blend_table(ec, gf)
    assert sorted(set((b["run"] // 60).tolist())) == [base, base + 12]
    lead, valid, t = _blend_rows(b, base + 12)
    assert lead.tolist() == list(range(0, 73, 3))  # lead = valid - R
    assert (valid == (base + 12) * 60 + lead.astype(np.int32) * 60).all()
    # valid <= ECMWF 00Z + 72 h (blend lead <= 60): mean of GFS 12Z (lead L) and ECMWF 00Z
    # (lead L + 12); beyond that ECMWF no longer covers the valid time: GFS only
    for L, x in zip(lead.tolist(), t.tolist(), strict=True):
        gfs = 30.0 + L / 100
        want = (gfs + 10.0 + (L + 12) / 100) / 2 if L <= 60 else gfs
        assert x == pytest.approx(want, abs=1e-4), L
    # same-time run present: plain mean, unchanged behaviour
    lead, _, t = _blend_rows(b, base)
    assert t.tolist() == pytest.approx([(20 + 10) / 2 + L / 100 for L in lead.tolist()], abs=1e-4)


def test_blend_never_uses_newer_or_too_old_runs():
    base = 1000
    # ECMWF has a 12Z run that must not leak into blend 00Z; GFS 00Z and 18Z
    ec = _runs_table([base, base + 12], lambda r: 10.0 if r == base else 99.0)
    gf = _runs_table([base, base + 18], lambda r: 20.0 if r == base else 40.0)
    b = correct.blend_table(ec, gf)
    assert sorted(set((b["run"] // 60).tolist())) == [base, base + 12, base + 18]
    _, _, t = _blend_rows(b, base)
    assert (t < 50).all()  # no ECMWF 12Z (99) in blend 00Z
    # blend 12Z: ECMWF 12Z + GFS 00Z (12 h old, allowed)
    lead, _, t = _blend_rows(b, base + 12)
    assert t[0] == pytest.approx((99.0 + 20.0 + 0.12) / 2, abs=1e-4)
    # blend 18Z: GFS 18Z + ECMWF 12Z (6 h old); ECMWF 00Z (18 h old) never used
    lead, _, t = _blend_rows(b, base + 18)
    assert t[0] == pytest.approx((40.0 + 99.0 + 0.06) / 2, abs=1e-4)
    # only a too-old component available -> single-model fallback
    gf2 = _runs_table([base + 18], lambda r: 40.0)
    b2 = correct.blend_table(_runs_table([base], lambda r: 10.0), gf2)
    _, _, t = _blend_rows(b2, base + 18)
    assert t[0] == pytest.approx(40.0)
    assert (b2["run"] <= b2["valid"]).all() and (b["run"] <= b["valid"]).all()


def test_blend_n_models_mixed_steps_and_missing_vars():
    """Three components: two 3-hourly, one 6-hourly AI model without gust."""
    base = 1000
    a = _runs_table([base], lambda r: 10.0)
    b = _runs_table([base], lambda r: 20.0)
    c = _runs_table([base], lambda r: 40.0, leads=range(0, 73, 6))
    a["gust"] = np.full(len(a["sid"]), 30, np.float32)
    b["gust"] = np.full(len(b["sid"]), 20, np.float32)  # c has no gust (NaN)
    a["wdir"][:] = 350
    b["wdir"][:] = 10
    c["wdir"][:] = 0
    bl = correct.blend_table(a, None, b, c)
    lead, _, t = _blend_rows(bl, base)
    assert lead.tolist() == list(range(0, 73, 3))
    for L, x in zip(lead.tolist(), t.tolist(), strict=True):
        want = (10 + 20 + 40) / 3 if L % 6 == 0 else (10 + 20) / 2
        assert x == pytest.approx(want + L / 100, abs=1e-4), L
    assert np.allclose(bl["gust"], 25)  # mean of the models that have gust
    d = bl["wdir"]
    assert np.all((d < 1e-3) | (d > 360 - 1e-3))  # circular mean of 350, 10, 0
    # order-independent and identical to the 2-model result for 2 components
    bl2 = correct.blend_table(c, b, a)
    assert np.array_equal(bl["t"], bl2["t"]) and np.array_equal(bl["sid"], bl2["sid"])
    two = correct.blend_table(a, b)
    assert two["t"].tolist() == pytest.approx(
        [(10 + 20) / 2 + L / 100 for L in range(0, 73, 3)], abs=1e-4
    )


def test_blend_n_models_keeps_age_rule_per_component():
    base = 1000
    a = _runs_table([base, base + 12], lambda r: 10.0 if r == base else 11.0)
    b = _runs_table([base], lambda r: 20.0)  # 12 h old at blend 12Z: allowed
    c = _runs_table([base - 6], lambda r: 99.0)  # 18 h old at 12Z: never used there
    bl = correct.blend_table(a, b, c)
    assert sorted(set((bl["run"] // 60).tolist())) == [base - 6, base, base + 12]
    _, _, t = _blend_rows(bl, base + 12)
    assert t[0] == pytest.approx((11.0 + 20.0 + 0.12) / 2, abs=1e-4)
    _, _, t = _blend_rows(bl, base)  # c is 6 h old at 00Z: used
    assert t[0] == pytest.approx((10.0 + 20.0 + 99.06) / 3, abs=1e-4)
    _, _, t = _blend_rows(bl, base - 6)  # only c exists: no later run leaks in
    assert t[0] == pytest.approx(99.0)


def test_build_models_corrects_every_registry_model(synth):
    ds = verify.load(synth.data_dir, synth.now)
    raw = {"aifs": ds.fc.pop("gfs"), "ecmwf": ds.fc.pop("ecmwf")}  # any registry ids
    for T in raw.values():
        verify.match_table(T, ds.obs)
    n_aifs = len(raw["aifs"]["sid"])
    out, info = correct.build_models(raw, ds.obs)
    # registry order: raw models, corrected models, blend
    assert list(out) == ["ecmwf", "aifs", "ecmwf_bc", "aifs_bc", "blend"]
    assert set(info) == {"ecmwf_bc", "aifs_bc"}
    assert len(out["aifs_bc"]["sid"]) == n_aifs
