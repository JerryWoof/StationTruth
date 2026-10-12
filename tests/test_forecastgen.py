from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from synth import make_dataset

from stationtruth import forecastgen as fg
from stationtruth import sitegen
from stationtruth.verify import NAT


def _load(path):
    def bad(x):
        raise AssertionError(f"non-JSON constant {x} in {path}")

    return json.loads(path.read_text(), parse_constant=bad)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """24 days of synthetic data; the forecast is built at day 20 (no look-ahead)."""
    d = tmp_path_factory.mktemp("fc")
    info = make_dataset(d / "data", n_stations=8, days=24, seed=11)
    now = info.now - timedelta(days=4)
    out = d / "out"
    summary = fg.build(info.data_dir, out, now=now)
    return info, now, out, summary


def _obs(info) -> pd.DataFrame:
    files = sorted(info.data_dir.glob("obs-*.parquet"))
    o = pd.concat([pd.read_parquet(p) for p in files])
    return o[o["type"] == "METAR"]


def _times(blk: dict, n: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(blk["start"]), periods=n, freq=f"{blk['step']}min")


def test_files_and_schema(built):
    info, now, out, summary = built
    fdir = out / "forecast"
    files = sorted(p.name for p in fdir.glob("*.json"))
    assert "index.json" in files
    assert summary["stations"] == 8 == len(files) - 1
    idx = _load(fdir / "index.json")
    assert idx["count"] == 8
    assert {s["id"] for s in idx["stations"]} == set(info.stations)
    assert set(idx["stations"][0]) == {"id", "name", "country", "lat", "lon"}

    d = _load(fdir / "LKPR.json")
    assert d["generated"] == now.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert d["scale"] == 10
    b = d["blend"]
    assert b["step"] == 60
    assert b["start"] == now.strftime("%Y-%m-%dT%H:00:00Z")
    for v in ("t", "td", "wspd", "wdir", "gust", "mslp", "tcc", "rh", "feels"):
        assert v in b, v
    for v in fg.BAND_VARS:
        assert f"{v}_lo" in b and f"{v}_hi" in b
    assert set(d["models"]) == {"ecmwf", "gfs", "ecmwf_bc", "gfs_bc"}
    assert d["models"]["ecmwf"]["step"] == 180
    assert "tcc" in d["models"]["ecmwf"] and "tcc" not in d["models"]["ecmwf_bc"]
    # hourly blend covers the run's remaining hours (run + 72 h)
    vals = fg.decode_block(b)
    run = pd.Timestamp(b["run"][:8] + "T" + b["run"][8:], tz="UTC")
    end = _times(b, len(vals["t"]))[-1]
    assert end == run + pd.Timedelta(hours=72)
    assert pd.Timestamp(b["start"]) >= run
    # bands bracket the forecast
    for v in fg.BAND_VARS:
        lo, mid, hi = (np.array(vals[k], float) for k in (f"{v}_lo", v, f"{v}_hi"))
        ok = np.isfinite(lo) & np.isfinite(hi)
        assert ok.mean() > 0.9
        assert (lo[ok] <= hi[ok]).all()
    assert d["band"]["t"]["src"] == ["station"] * 6
    assert d["band"]["q"] == [0.1, 0.9]
    # derived fields
    t, td, rh = (np.array(vals[k], float) for k in ("t", "td", "rh"))
    assert np.all((rh > 0) & (rh <= 100))
    assert np.allclose(rh, fg.rel_humidity(t, td), atol=0.6)
    s = d["skill"]
    assert set(s["models"]) == {"ecmwf", "gfs", "ecmwf_bc", "gfs_bc", "blend"}
    assert s["summary"]["text"].startswith("Blend temperature error here")
    # the synthetic models are biased; the blend of corrected models beats both raw ones
    assert s["summary"]["better_pct"]["ecmwf"] > 0 and s["summary"]["better_pct"]["gfs"] > 0
    assert d["obs"]["time"] <= d["generated"]


def test_band_covers_about_80_percent(built):
    """Out of sample: hourly observations after ``now`` fall inside the band ~80 %."""
    info, now, out, _ = built
    obs = _obs(info)
    obs = obs.assign(hour=obs["time"].dt.round("h"))
    obs = obs.drop_duplicates(["station", "hour"]).set_index(["station", "hour"])
    inside, n = 0, 0
    width = []
    for sid in info.stations:
        d = _load(out / "forecast" / f"{sid}.json")
        b = d["blend"]
        vals = fg.decode_block(b)
        times = _times(b, len(vals["t"]))
        for k, tm in enumerate(times):
            if (sid, tm) not in obs.index:
                continue
            lo, hi = vals["t_lo"][k], vals["t_hi"][k]
            if lo is None or hi is None:
                continue
            o = obs.loc[(sid, tm), "t"]
            n += 1
            inside += lo <= o <= hi
            width.append(hi - lo)
    assert n > 300
    cover = inside / n
    assert 0.68 <= cover <= 0.92, cover
    assert 0.5 < np.median(width) < 6


def test_band_in_sample_quantiles():
    """The band's quantiles are the empirical 10th/90th percentiles of the errors."""
    rng = np.random.default_rng(0)
    e = rng.normal(0.5, 2.0, 5000)
    g = rng.integers(0, 3, 5000)
    n, q = fg.group_quantiles(g, e, 4)
    assert n.tolist()[:3] == np.bincount(g).tolist() and n[3] == 0
    for k in range(3):
        assert np.allclose(q[k], np.quantile(e[g == k], [0.1, 0.9]))
        inside = ((e[g == k] >= q[k, 0]) & (e[g == k] <= q[k, 1])).mean()
        assert abs(inside - 0.8) < 0.01
    assert np.isnan(q[3]).all()


def test_interpolation():
    run = np.array([0, 60, NAT], np.int64)  # minutes; second run starts an hour later
    dense = np.full((3, fg.N_STEPS), np.nan, np.float32)
    dense[0, :3] = [0.0, 3.0, 9.0]
    dense[1, :3] = [10.0, 13.0, np.nan]
    h = fg.to_hourly(dense, run, start=60, n_hours=7)
    assert h[0, :6].tolist() == pytest.approx([1, 2, 3, 5, 7, 9])
    assert np.isnan(h[0, 6])
    # run 1 starts at 60: exact on steps, NaN next to a missing step
    assert h[1, 0] == 10 and h[1, 3] == 13
    assert h[1, 1] == pytest.approx(11)
    assert np.isnan(h[1, 4])
    assert np.isnan(h[2]).all()
    # wind direction goes the short way round
    dd = np.full((1, fg.N_STEPS), np.nan, np.float32)
    dd[0, :2] = [350.0, 20.0]
    w = fg.to_hourly(dd, np.array([0]), start=0, n_hours=4, circular=True)
    assert w[0].tolist()[:4] == pytest.approx([350, 0, 10, 20])
    # band weights: constant beyond the outer bucket centres, linear in between
    W = fg.interp_weights(np.array([0.0, fg.BUCKET_CENTRES[1], 72.0]))
    assert W.sum(axis=-1) == pytest.approx(1)
    assert W[0, 0] == 1 and W[1, 1] == 1 and W[2, -1] == 1


def test_encoding_round_trip():
    a = np.array([14.04, 12.0, np.nan, 13.0, -2.55, np.nan])
    b = np.array([np.nan] * 6)
    blk = fg.block(0, 60, {"t": a, "x": b}, run="1970010100")
    assert "x" not in blk  # variables without values are left out
    assert blk["t"] == sitegen._tenths(a[:5])  # trailing empty slot trimmed
    got = fg.decode_block(blk)["t"]
    assert got == [14.0, 12.0, None, 13.0, -2.6] or got == [14.0, 12.0, None, 13.0, -2.5]
    assert fg.block(0, 60, {"t": b}) is None


def test_feels_like():
    t = np.array([-5.0, 30.0, 15.0, np.nan])
    rh = np.array([80.0, 70.0, 50.0, 50.0])
    wind = np.array([20.0, 5.0, 30.0, 5.0])
    f = fg.feels_like(t, rh, wind)
    assert f[0] < -10  # wind chill
    assert f[1] > 33  # heat index
    assert f[2] == 15  # neither
    assert np.isnan(f[3])


def test_empty_data(tmp_path):
    (tmp_path / "data").mkdir()
    out = tmp_path / "out"
    summary = fg.build(tmp_path / "data", out)
    assert summary["stations"] == 0
    idx = _load(out / "forecast" / "index.json")
    assert idx["stations"] == [] and idx["count"] == 0


def test_stale_files_removed_and_cli(built, tmp_path):
    info, now, _, _ = built
    out = tmp_path / "out"
    (out / "forecast").mkdir(parents=True)
    (out / "forecast" / "GONE.json").write_text("{}")
    rc = fg.main(["--data-dir", str(info.data_dir), "--out", str(out), "--now", now.isoformat()])
    assert rc == 0
    names = {p.name for p in (out / "forecast").glob("*.json")}
    assert "GONE.json" not in names and "LKPR.json" in names
