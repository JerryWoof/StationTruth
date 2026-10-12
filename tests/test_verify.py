from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth import make_dataset

from stationtruth import regions, sitegen, verify

UTC = timezone.utc


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    info = make_dataset(d / "data", n_stations=6, days=20)
    out = d / "site"
    summary = sitegen.build(d / "data", out, now=info.now)
    scores = json.loads((out / "scores.json").read_text())
    return info, out, summary, scores


def _key(sid, minutes):
    return (np.asarray(sid, dtype=np.int64) << 32) | np.asarray(minutes, dtype=np.int64)


def test_match_nearest_rules():
    # station 0: METAR 10:50, SPECI 11:10 -> query 11:00 tie, METAR wins
    # station 1: SPECI 10:50, METAR 11:10 -> tie, METAR wins (later one)
    # station 2: METAR 10:50, METAR 11:10 -> tie of same type, earlier wins
    # station 3: only 10:29 -> 31 min away, no match
    # station 4: SPECI 11:05 and METAR 10:50 -> nearest (SPECI) wins when not a tie
    obs = [
        (0, 650, True),
        (0, 670, False),
        (1, 650, False),
        (1, 670, True),
        (2, 650, True),
        (2, 670, True),
        (3, 629, True),
        (4, 650, True),
        (4, 665, False),
    ]
    okey = _key([o[0] for o in obs], [o[1] for o in obs])
    metar = np.array([o[2] for o in obs])
    q = _key([0, 1, 2, 3, 4, 5], [660] * 6)
    idx = verify.match_nearest(okey, metar, q)
    assert idx.tolist() == [0, 3, 4, -1, 8, -1]


def test_match_never_crosses_stations():
    okey = _key([0], [1000])
    idx = verify.match_nearest(okey, np.array([True]), _key([1], [1000]))
    assert idx.tolist() == [-1]


def test_wrap_dir():
    d = verify.wrap_dir(np.array([10 - 350, 350 - 10, 180.0, -180.0, 0.0]))
    assert d.tolist() == [20.0, -20.0, -180.0, -180.0, 0.0]


def test_scores_recover_known_biases(built):
    info, _, _, scores = built
    g = scores["windows"]["30d"]["global"]
    for model, spec in info.models.items():
        for var in ("t", "td", "wspd", "mslp", "wdir"):
            got = g[model][var]["all"]["bias"]
            tol = 2.0 if var == "wdir" else 0.1
            assert got == pytest.approx(spec.bias[var], abs=tol), (model, var)
    # mslp only where SLP is reported (every other synthetic station)
    assert g["ecmwf"]["mslp"]["all"]["n"] == pytest.approx(
        g["ecmwf"]["t"]["all"]["n"] / 2, rel=0.05
    )
    # gust only when the observation reports a gust
    assert 0 < g["ecmwf"]["gust"]["all"]["n"] < g["ecmwf"]["t"]["all"]["n"] / 3
    # RMSE >= |bias|, MAE <= RMSE
    for m in g.values():
        for v in m.values():
            a = v["all"]
            if a["n"] == 0:  # registry model absent from the synthetic data
                continue
            assert a["rmse"] + 1e-9 >= abs(a["bias"]) and a["mae"] <= a["rmse"] + 1e-9


def test_scores_by_lead_and_regions(built):
    _, _, _, scores = built
    w = scores["windows"]["7d"]
    by_lead = w["global"]["gfs"]["t"]["by_lead"]
    assert [b["lead_h"] for b in by_lead] == list(range(0, 73, 3))
    assert sum(b["n"] for b in by_lead) == w["global"]["gfs"]["t"]["all"]["n"]
    assert set(w["regions"]) == set(regions.REGIONS)
    n_reg = sum(r["gfs"]["t"]["all"]["n"] for r in w["regions"].values())
    assert n_reg == w["global"]["gfs"]["t"]["all"]["n"]  # every synthetic station has a region
    assert w["regions"]["Polar"]["gfs"]["t"]["all"]["n"] > 0  # Svalbard
    # error grows with lead
    assert by_lead[-1]["rmse"] > by_lead[0]["rmse"]
    assert (
        w["global"]["ecmwf"]["t"]["all"]["n"]
        < scores["windows"]["30d"]["global"]["ecmwf"]["t"]["all"]["n"]
    )


def _write_tiny(d: Path, obs_rows, fc_rows, run="2026101000"):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(obs_rows).to_parquet(d / "obs-2026-10-10.parquet", index=False)
    pd.DataFrame(fc_rows).to_parquet(d / f"fcst-ecmwf-{run}.parquet", index=False)


def test_wdir_only_with_wind_and_qc(tmp_path):
    t0 = pd.Timestamp("2026-10-10T00:00Z")
    obs, fc = [], []
    for h in range(12):
        ts = t0 + pd.Timedelta(hours=h)
        obs.append(
            {
                "station": "AAAA",
                "time": ts,
                "type": "METAR",
                "t": 10.0 if h != 5 else 99.0,  # 99 C is rejected by QC
                "wspd": 3.0 if h < 6 else 10.0,
                "wdir": 350.0,
            }
        )
        fc.append(
            {
                "station": "AAAA",
                "run": t0,
                "valid": ts,
                "lead_h": h,
                "t": 11.0,
                "wspd": 3.0,
                "wdir": 20.0,
            }
        )
    _write_tiny(tmp_path / "d", obs, fc)
    ds = verify.load(tmp_path / "d", datetime(2026, 10, 10, 12, tzinfo=UTC))
    T = ds.fc["ecmwf"]
    verify.match_table(T, ds.obs)
    rows, e = verify.pair_errors(T, ds.obs, "wdir")
    assert len(rows) == 6 and np.allclose(e, 30.0)  # wspd >= 5 only, wrapped 20-350
    rows, e = verify.pair_errors(T, ds.obs, "t")
    assert len(rows) == 11 and np.allclose(e, 1.0)
    for var in ("gust", "mslp", "td"):  # columns missing entirely
        assert len(verify.pair_errors(T, ds.obs, var)[0]) == 0


def test_load_ignores_future_files(tmp_path):
    info = make_dataset(tmp_path / "d", n_stations=2, days=4)
    early = datetime(2026, 9, 22, tzinfo=UTC)
    ds = verify.load(tmp_path / "d", early)
    assert ds.obs["time"].max() <= verify.to_minutes(early)
    assert max(max(r) for r in ds.runs.values()) <= verify.to_minutes(early)
    full = verify.load(tmp_path / "d", info.now)
    assert len(full.obs["time"]) > len(ds.obs["time"])


def test_regions():
    r = regions.region_for
    assert r("CZ", 50.1, 14.3) == "Europe"
    assert r("NO", 78.2, 15.5) == "Polar"
    assert r("AQ", -77.8, 166.7) == "Polar"
    assert r("RU", 55.6, 37.4) == "Europe"
    assert r("RU", 43.4, 132.1) == "Asia"
    assert r("US", 21.3, -157.9) == "Oceania"
    assert r("US", 40.6, -73.8) == "North America"
    assert r("PR", 18.4, -66.0) == "Central America & Caribbean"
    assert r("AE", 25.2, 55.4) == "Middle East"
    assert r("", 48.0, 2.0) == "Europe"
    assert r(None, float("nan"), float("nan")) is None
    assert len(regions.COUNTRY_REGION) >= 249
    assert set(regions.COUNTRY_REGION.values()) == set(regions.REGIONS)
