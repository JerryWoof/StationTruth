from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from synth import make_dataset

from stationtruth import sitegen

UTC = timezone.utc
MODELS = ["ecmwf", "gfs", "ecmwf_bc", "gfs_bc", "blend"]
VARS = {"t", "td", "wspd", "gust", "mslp", "wdir"}
META_KEYS = {
    "generated",
    "first_obs",
    "obs_count",
    "station_count",
    "runs",
    "latest_run",
    "models",
    "pairs",
}


def _load(path):
    def bad(x):
        raise AssertionError(f"non-JSON constant {x} in {path}")

    return json.loads(path.read_text(), parse_constant=bad)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    d = tmp_path_factory.mktemp("site")
    info = make_dataset(d / "data", n_stations=8, days=16, seed=5)
    out = d / "out"
    summary = sitegen.build(info.data_dir, out, now=info.now)
    return info, out, summary


def _check_score_shape(scores):
    assert set(scores["windows"]) == {"7d", "30d"}
    for w in scores["windows"].values():
        for block in (w["global"], *w["regions"].values()):
            assert set(block) == set(MODELS)
            for m in block.values():
                assert set(m) == VARS
                for v in m.values():
                    assert set(v) == {"by_lead", "all"}
                    assert set(v["all"]) == {"n", "bias", "mae", "rmse"}


def test_full_build_schema(site):
    info, out, summary = site
    meta = _load(out / "meta.json")
    assert META_KEYS <= set(meta)
    assert meta["models"] == MODELS
    assert meta["pairs"] == summary["pairs"] > 0
    assert meta["status"] == "ok"
    assert meta["latest_run"]["ecmwf"] == meta["runs"]["ecmwf"][-1]
    assert meta["first_obs"].startswith("2026-09-20T00:")
    _check_score_shape(_load(out / "scores.json"))

    latest = _load(out / "latest.json")
    assert len(latest) == 8
    e = latest[0]
    assert set(e) >= {"id", "name", "country", "lat", "lon", "time", "obs", "fc", "bias7", "best"}
    assert set(e["obs"]) == VARS
    assert set(e["fc"]) == set(MODELS)
    assert set(e["fc"]["ecmwf"]) == {"run", "lead_h"} | VARS
    assert e["best"] in MODELS
    assert set(e["bias7"]["gfs"]) == {"t", "wspd"}
    assert e["bias7"]["gfs"]["t"] < 0 < e["bias7"]["ecmwf"]["t"]

    files = sorted(p.name for p in (out / "stations").iterdir())
    assert len(files) == 8 == summary["station_files"]
    st = _load(out / "stations" / "LKPR.json")
    assert set(st) == {
        "id",
        "name",
        "country",
        "lat",
        "lon",
        "elev_m",
        "obs",
        "past",
        "future",
        "scores",
    }
    assert set(st["obs"]) == {"time"} | VARS
    assert 160 <= len(st["obs"]["time"]) <= 168
    assert all(len(v) == len(st["obs"]["time"]) for v in st["obs"].values())
    assert set(st["past"]) == set(MODELS) == set(st["future"])
    assert set(st["past"]["ecmwf"]["lead"]) <= {21, 24, 27}
    fut = st["future"]["blend"]
    assert fut["run"] == meta["latest_run"]["blend"] and len(fut["valid"]) == 25
    assert set(st["scores"]) == {"7d", "30d"}
    assert set(st["scores"]["7d"]["gfs"]) == VARS
    # 1-decimal rounding
    assert all(v is None or round(v, 1) == v for v in st["obs"]["t"] + st["past"]["gfs"]["t"])
    raw = (out / "stations" / "LKPR.json").read_text()
    assert ": " not in raw and "NaN" not in raw


def test_latest_forecast_is_interpolated(site):
    info, out, _ = site
    e = next(x for x in _load(out / "latest.json") if x["id"] == "LKPR")
    t = datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    fc = e["fc"]["ecmwf"]
    run = datetime.strptime(fc["run"], "%Y%m%d%H").replace(tzinfo=UTC)
    assert run <= t <= run + timedelta(hours=72)
    assert fc["lead_h"] == int((t - run).total_seconds() // 3600)
    # ecmwf bias +0.5 (+-0.5 diurnal) and noise: forecast is within a few degrees of obs
    assert abs(fc["t"] - e["obs"]["t"]) < 5


def test_empty_and_missing_dir(tmp_path):
    for src in (tmp_path / "empty", tmp_path / "does-not-exist"):
        (tmp_path / "empty").mkdir(exist_ok=True)
        out = tmp_path / ("out-" + src.name)
        s = sitegen.build(src, out, now=datetime(2026, 10, 12, 16, tzinfo=UTC))
        meta = _load(out / "meta.json")
        assert META_KEYS <= set(meta)
        assert meta["status"] == "collecting" and meta["pairs"] == 0 and meta["first_obs"] is None
        _check_score_shape(_load(out / "scores.json"))
        assert _load(out / "latest.json") == []
        assert list((out / "stations").iterdir()) == []
        assert s["pairs"] == 0


def test_obs_only(tmp_path):
    info = make_dataset(tmp_path / "d", n_stations=3, days=2, models={})
    out = tmp_path / "out"
    sitegen.build(info.data_dir, out, now=info.now)
    meta = _load(out / "meta.json")
    assert meta["status"] == "collecting" and meta["obs_count"] > 0
    latest = _load(out / "latest.json")
    assert len(latest) == 3 and latest[0]["fc"] == {} and latest[0]["best"] is None
    st = _load(out / "stations" / "LKPR.json")
    assert st["past"] == {} and st["future"] == {} and len(st["obs"]["time"]) > 40


def test_forecasts_without_obs(tmp_path):
    info = make_dataset(tmp_path / "d", n_stations=3, days=2)
    for p in info.data_dir.glob("obs-*.parquet"):
        p.unlink()
    out = tmp_path / "out"
    sitegen.build(info.data_dir, out, now=info.now)
    meta = _load(out / "meta.json")
    assert meta["pairs"] == 0 and meta["latest_run"]["gfs"]
    assert _load(out / "latest.json") == []
    _check_score_shape(_load(out / "scores.json"))


def test_forecasts_not_yet_verified(tmp_path):
    """A run newer than every observation: no pairs, but it is the 'future' forecast."""
    info = make_dataset(tmp_path / "d", n_stations=2, days=2)
    now = info.now + timedelta(hours=1)
    df = pd.read_parquet(info.data_dir / "fcst-gfs-2026092112.parquet")
    df["run"] = pd.Timestamp(info.now)
    df["valid"] = df["run"] + pd.to_timedelta(df["lead_h"].astype(int), unit="h")
    df.to_parquet(info.data_dir / f"fcst-gfs-{info.now:%Y%m%d%H}.parquet", index=False)
    out = tmp_path / "out"
    sitegen.build(info.data_dir, out, now=now)
    st = _load(out / "stations" / "LKPR.json")
    assert st["future"]["gfs"]["run"] == f"{info.now:%Y%m%d%H}"


def test_missing_columns_and_nans(tmp_path):
    info = make_dataset(tmp_path / "d", n_stations=3, days=3)
    for p in info.data_dir.glob("obs-*.parquet"):
        df = pd.read_parquet(p).drop(columns=["gust", "slp", "type"])
        df.loc[df.index[::3], "t"] = np.nan
        df.to_parquet(p, index=False)
    for p in info.data_dir.glob("fcst-ecmwf-*.parquet"):
        df = pd.read_parquet(p).drop(columns=["gust", "mslp", "valid"])
        df.loc[df.index[::2], "wspd"] = np.nan
        df.to_parquet(p, index=False)
    (info.data_dir / "stations.parquet").unlink()
    (info.data_dir / "obs-1999-01-01.parquet").write_bytes(b"not parquet")
    out = tmp_path / "out"
    stale = out / "stations" / "ZZZZ.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    sitegen.build(info.data_dir, out, now=info.now)
    meta = _load(out / "meta.json")
    assert meta["pairs"] > 0
    sc = _load(out / "scores.json")["windows"]["7d"]["global"]
    assert sc["ecmwf"]["gust"]["all"]["n"] == 0 and sc["ecmwf"]["t"]["all"]["n"] > 0
    st = _load(out / "stations" / "LKPR.json")
    assert st["name"] == "" and st["lat"] is None
    assert None in st["obs"]["t"] or len(st["obs"]["t"]) > 0
    assert not stale.exists()
    latest = _load(out / "latest.json")
    assert latest and all(x["obs"]["gust"] is None for x in latest)


def test_forecast_at_picks_latest_covering_run_and_interpolates():
    nan = np.nan

    def rows(run, leads, t, gust):
        n = len(leads)
        lead = np.array(leads, np.int16)
        return {
            "sid": np.zeros(n, np.int32),
            "run": np.full(n, run, np.int32),
            "lead": lead,
            "valid": (run + lead.astype(np.int32) * 60).astype(np.int32),
            "t": np.array(t, np.float32),
            "td": np.full(n, nan, np.float32),
            "wspd": np.full(n, nan, np.float32),
            "gust": np.array(gust, np.float32),
            "wdir": np.array([350, 10, 30, 50][:n], np.float32),
            "mslp": np.full(n, nan, np.float32),
        }

    r0, r1 = 1_000_000, 1_000_000 + 720
    a, b = (
        rows(r0, [12, 15, 18, 21], [1, 2, 3, 4], [5, 6, 7, 8]),
        rows(r1, [0, 3, 6], [10, 13, 16], [nan, 9, 9]),
    )
    T = {k: np.concatenate([a[k], b[k]]) for k in a}
    # 30 min after r1: run r1 covers it (newest), between lead 0 and 3
    f = sitegen.forecast_at(T, np.array([0]), np.array([r1 + 30]), 1).loc[0]
    assert f["run"] == r1 and f["lead_h"] == 0
    assert f["t"] == pytest.approx(10.5)
    assert np.isnan(f["gust"])  # lead-0 gust missing: no interpolation across it
    assert f["wdir"] == pytest.approx(350 + 20 / 6)  # 350 -> 10 through north
    # exactly at lead 3 the missing lead-0 gust does not matter
    f = sitegen.forecast_at(T, np.array([0]), np.array([r1 + 180]), 1).loc[0]
    assert f["gust"] == pytest.approx(9.0) and f["t"] == pytest.approx(13.0)
    # beyond r1's last step only the older run r0 covers
    f = sitegen.forecast_at(T, np.array([0]), np.array([r1 + 400]), 1).loc[0]
    assert f["run"] == r0 and f["lead_h"] == 18 and f["t"] == pytest.approx(3 + 40 / 180)
