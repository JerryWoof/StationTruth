from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from stationtruth import cli, grib, obs, stations

UTC = timezone.utc
NOW = datetime(2026, 10, 12, 16, 7, tzinfo=UTC)


@pytest.fixture
def offline(monkeypatch):
    st = stations.to_frame(
        [
            {
                "station": "LKPR",
                "name": "Prague",
                "country": "CZ",
                "lat": 50.1,
                "lon": 14.26,
                "elev_m": 364,
            },
            {
                "station": "KJFK",
                "name": "JFK",
                "country": "US",
                "lat": 40.6,
                "lon": -73.8,
                "elev_m": 3,
            },
        ]
    )
    monkeypatch.setattr(cli, "utcnow", lambda: NOW)
    monkeypatch.setattr(cli.stations_mod, "fetch_stations", lambda: st)
    reports = [
        ("METAR LKPR 121600Z 29005KT CAVOK 12/05 Q1022", NOW, None),
        ("METAR KJFK 112351Z 35010KT 10SM FEW250 18/13 A2981", NOW, None),  # yesterday
        ("METAR KJFK 102351Z 35010KT 10SM FEW250 18/13 A2981", NOW, None),  # too old
    ]
    monkeypatch.setattr(
        cli.obs_mod, "fetch_latest", lambda now=None: obs.decode_reports(reports, now=NOW)
    )
    return st


def test_collect_obs_local(offline, tmp_path):
    assert cli.main(["collect-obs", "--data-dir", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["obs-2026-10-11.parquet", "obs-2026-10-12.parquet", "stations.parquet"]
    today = obs.read_obs(tmp_path / "obs-2026-10-12.parquet")
    assert today.station.tolist() == ["LKPR"]
    # running again merges without duplicates
    assert cli.main(["collect-obs", "--data-dir", str(tmp_path)]) == 0
    assert len(obs.read_obs(tmp_path / "obs-2026-10-12.parquet")) == 1


def test_collect_fcst_local(offline, tmp_path, monkeypatch):
    fetched = []

    def fake_fetch_run(source, run, st, log=None, **kw):
        fetched.append((source.name, run))
        df = pd.DataFrame(
            {"station": st.station.repeat(2).tolist(), "lead_h": [0, 3] * len(st), "t": 1.0}
        )
        df["run"] = pd.Timestamp(run)
        return grib.to_fcst_frame(df), {
            "bytes": 1,
            "requests": 1,
            "seconds": 0.0,
            "failed_steps": [],
        }

    available = {datetime(2026, 10, 12, 0, tzinfo=UTC), datetime(2026, 10, 11, 12, tzinfo=UTC)}
    monkeypatch.setattr(cli.grib, "fetch_run", fake_fetch_run)
    monkeypatch.setattr(cli.grib, "run_available", lambda source, run: run in available)

    assert (
        cli.main(["collect-fcst", "--model", "all", "--runs", "1", "--data-dir", str(tmp_path)])
        == 0
    )
    assert sorted(fetched) == [
        ("ecmwf", datetime(2026, 10, 12, tzinfo=UTC)),
        ("gfs", datetime(2026, 10, 12, tzinfo=UTC)),
    ]
    assert (tmp_path / "fcst-ecmwf-2026101200.parquet").exists()
    assert (tmp_path / "stations.parquet").exists()  # fetched because none was stored
    fetched.clear()
    assert (
        cli.main(["collect-fcst", "--model", "gfs", "--runs", "2", "--data-dir", str(tmp_path)])
        == 0
    )
    assert fetched == [("gfs", datetime(2026, 10, 11, 12, tzinfo=UTC))]
    df = pd.read_parquet(tmp_path / "fcst-gfs-2026101112.parquet")
    assert (df.valid - df.run).max() == timedelta(hours=3)


def test_backfill_obs_only(offline, tmp_path, monkeypatch):
    seen = {}

    def fake_history(ids, start, end, **kw):
        seen["ids"], seen["start"] = list(ids), start
        return obs.decode_reports(
            [("METAR LKPR 101200Z 29005KT CAVOK 12/05 Q1022", NOW, None)], now=NOW
        )

    monkeypatch.setattr(cli.obs_mod, "fetch_history", fake_history)
    rc = cli.main(["backfill", "--days", "3", "--models", "", "--data-dir", str(tmp_path)])
    assert rc == 0
    assert seen["ids"] == ["KJFK", "LKPR"] and seen["start"] == NOW - timedelta(days=3)
    assert (tmp_path / "obs-2026-10-10.parquet").exists()


def test_sync_from_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "utcnow", lambda: NOW)
    src = tmp_path / "src"
    src.mkdir()
    for name in ["obs-2026-10-12.parquet", "obs-2026-08-01.parquet", "fcst-gfs-2026101100.parquet"]:
        (src / name).write_bytes(b"x")
    rc = cli.main(["sync", "--days", "3", "--to", str(tmp_path / "out"), "--from-dir", str(src)])
    assert rc == 0
    assert sorted(p.name for p in (tmp_path / "out").glob("*.parquet")) == [
        "fcst-gfs-2026101100.parquet",
        "obs-2026-10-12.parquet",
    ]


def test_parser_help_lists_commands():
    p = cli.build_parser()
    text = p.format_help()
    for cmd in ["collect-obs", "collect-fcst", "backfill", "sync", "build-site"]:
        assert cmd in text
