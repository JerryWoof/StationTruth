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


def _tables(info, mutate_obs=None):
    ds = verify.load(info.data_dir, info.now)
    if mutate_obs is not None:
        mutate_obs(ds.obs)
    for T in ds.fc.values():
        verify.match_table(T, ds.obs)
    tables, _ = correct.build_models(ds.fc, ds.obs)
    return ds, tables


def test_correction_reduces_error(synth, tmp_path):
    sitegen.build(synth.data_dir, tmp_path, now=synth.now)
    g = json.loads((tmp_path / "scores.json").read_text())["windows"]["7d"]["global"]
    for m in ("ecmwf", "gfs"):
        for v in ("t", "td", "wspd", "mslp"):
            raw, bc = g[m][v]["all"], g[f"{m}_bc"][v]["all"]
            assert bc["n"] == raw["n"]
            assert bc["rmse"] < raw["rmse"], (m, v)
            assert abs(bc["bias"]) < 0.15, (m, v)
        assert g[f"{m}_bc"]["wdir"] == g[m]["wdir"]  # direction is not corrected
    assert g["blend"]["t"]["all"]["rmse"] < min(g["ecmwf_bc"]["t"]["all"]["rmse"], 1e9)


def test_no_look_ahead(synth):
    """Corrections for run R must not change when pairs with valid > R change."""
    ds0, t0 = _tables(synth)
    R = sorted(set(t0["ecmwf"]["run"].tolist()))[len(set(t0["ecmwf"]["run"].tolist())) // 2]

    def poison(obs):
        late = obs["time"] > R
        for v in ("t", "td", "wspd", "gust", "slp"):
            obs[v][late] += np.float32(7.0)

    ds1, t1 = _tables(synth, poison)
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


def test_window_min_pairs_and_strictly_earlier_runs():
    T, obs = _single_station_table(30)
    corr, n = correct.correction(T, obs, "t")
    # run k sees errors of runs j < k (its own lead-0 pair is not available) within 14 days
    assert n[:9].tolist() == list(range(9))
    assert np.isnan(corr[:8]).all()  # fewer than 8 pairs -> no correction
    assert corr[8] == pytest.approx(np.mean(range(8)))
    assert n[20] == 13 and corr[20] == pytest.approx(np.mean(range(7, 20)))


def test_corrected_values_and_clipping():
    T, obs = _single_station_table(12)
    T["wspd"] = np.full(12, 1.0, np.float32)
    obs["wspd"] = np.zeros(12, np.float32)
    T["wspd"][:] = 5.0  # +5 kt bias ...
    T["wspd"][11] = 2.0  # ... so 2 kt corrected by -5 kt must clip to 0
    bc, applied = correct.corrected_table(T, obs)
    assert np.isfinite(bc["t"]).all()
    assert bc["t"][11] == pytest.approx(T["t"][11] - np.mean(range(11)))
    assert bc["wspd"][11] == 0 and (bc["wspd"] >= 0).all()
    assert bc["wdir"] is T["wdir"]
    assert applied["t"] == pytest.approx(4 / 12)


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
