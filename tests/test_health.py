from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from stationtruth import health

UTC = timezone.utc
NOW = datetime(2026, 10, 12, 18, 20, tzinfo=UTC)


def _write_obs(data_dir, skip_hours=(), low_hours=(), per_hour=100, until=NOW):
    """Hourly obs for 3 days up to ``until``; ``skip_hours``/``low_hours`` are hours
    before NOW's hour (1 = previous hour) with no / few reports."""
    h_now = NOW.replace(minute=0)
    rows = []
    t = h_now - timedelta(hours=72)
    while t <= until:
        back = int((h_now - t).total_seconds() // 3600)
        n = 0 if back in skip_hours else (per_hour // 5 if back in low_hours else per_hour)
        for i in range(n):
            rows.append((f"S{i:03d}", t + timedelta(minutes=50 if i % 2 else 20)))
        t += timedelta(hours=1)
    df = pd.DataFrame(rows, columns=["station", "time"])
    df = df[df["time"] <= until]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["type"] = "METAR"
    df["t"] = np.float32(10.0)
    for day, g in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        g.to_parquet(data_dir / f"obs-{day}.parquet", index=False)


def _write_runs(data_dir, model, skip=(), days=8):
    first = (NOW - timedelta(days=days)).replace(hour=0, minute=0)
    run = first
    names = []
    while run <= NOW - timedelta(hours=6):
        rid = run.strftime("%Y%m%d%H")
        if rid not in skip:
            name = f"fcst-{model}-{rid}.parquet"
            pd.DataFrame({"station": ["S000"], "lead_h": [0], "t": [1.0]}).to_parquet(
                data_dir / name, index=False
            )
            names.append((name, run))
        run += timedelta(hours=12)
    return names


def test_obs_gap_detection(tmp_path):
    # hours 2-3 before now: no reports (recent -> degraded); 20 h back: few reports
    _write_obs(tmp_path, skip_hours=(2, 3), low_hours=(20,))
    _write_runs(tmp_path, "gfs")
    doc = health.compute(tmp_path, tmp_path / "site", NOW)
    o = doc["obs"]
    st = o["hours"]["state"]
    assert len(st) == 48 == len(o["hours"]["reports"])
    assert st[-1] == "partial"  # current hour
    assert st[-2] == "partial"  # previous hour not settled yet (ended 20 min ago)
    assert st[-3] == "gap" and st[-4] == "gap" and st[-5] == "ok"
    assert st[-21] == "low"
    assert o["median_reports"] == 100
    kinds = sorted((g["kind"], g["hours"]) for g in o["gaps"])
    assert kinds == [("gap", 2), ("low", 1)]
    assert doc["status"] == "degraded"
    levels = {(r["component"], r["level"]) for r in doc["reasons"]}
    assert ("obs", "degraded") in levels and ("obs", "info") in levels
    assert o["stations_24h"] == 100


def test_obs_stopped_is_down(tmp_path):
    _write_obs(tmp_path, until=NOW - timedelta(hours=5))
    _write_runs(tmp_path, "gfs")
    doc = health.compute(tmp_path, tmp_path / "site", NOW)
    assert doc["status"] == "down"
    assert doc["obs"]["age_min"] >= 300
    assert doc["obs"]["stations_3h"] == 0


def test_healthy_pipeline_is_ok(tmp_path):
    _write_obs(tmp_path)
    names = _write_runs(tmp_path, "ecmwf") + _write_runs(tmp_path, "gfs")
    # asset listing: each run stored 6 h after initialisation
    sync = {n: f"1|{(r + timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')}" for n, r in names}
    (tmp_path / ".sync.json").write_text(json.dumps(sync))
    site = tmp_path / "site"
    (site / "analysis").mkdir(parents=True)
    (site / "analysis" / "analysis.json").write_text(
        json.dumps({"generated": "2026-10-12T18:00:00Z", "valid": "2026-10-12T17:00:00Z"})
    )
    doc = health.compute(tmp_path, site, NOW)
    assert doc["status"] == "ok", doc["reasons"]
    g = doc["runs"]["models"]["gfs"]
    assert g["stored"] == g["expected"] == 14 and g["missing"] == []
    assert g["median_late_h"] == 6.0
    assert g["cycles"][-1]["status"] == "ok"  # 12Z is due by 18:20 with a 6 h store
    assert doc["analysis"]["status"] == "ok"
    assert doc["runs"]["listing"] is True


def test_missing_runs(tmp_path):
    _write_obs(tmp_path)
    # gfs: one old run missing (info) and the latest due run missing (degraded)
    _write_runs(tmp_path, "gfs", skip=("2026100900", "2026101200", "2026101212"))
    _write_runs(tmp_path, "ecmwf")
    doc = health.compute(tmp_path, tmp_path / "site", NOW)
    g = doc["runs"]["models"]["gfs"]
    assert "2026100900" in g["missing"] and "2026101200" in g["missing"]
    assert g["cycles"][-1] == {"run": "2026101212", "status": "pending"}  # not due yet
    assert doc["status"] == "degraded"
    msgs = [r for r in doc["reasons"] if r["component"] == "runs:gfs"]
    assert {r["level"] for r in msgs} == {"degraded", "info"}
    assert doc["runs"]["models"]["ecmwf"]["missing"] == []
    assert doc["runs"]["models"]["gfs"]["median_late_h"] is None  # no listing


def test_parse_stamp_and_listing(tmp_path):
    assert health._parse_stamp("12|2026-10-12T17:06:03Z") == health.to_minutes(
        datetime(2026, 10, 12, 17, 6, tzinfo=UTC)
    )
    ns = int(datetime(2026, 10, 12, 17, 6, tzinfo=UTC).timestamp() * 1e9)
    assert health._parse_stamp(f"12|{ns}") == health.to_minutes(
        datetime(2026, 10, 12, 17, 6, tzinfo=UTC)
    )
    assert health._parse_stamp("garbage") is None
    assert health.read_listing(tmp_path) == {}


def test_build_writes_json_and_keeps_build_info(tmp_path):
    _write_obs(tmp_path)
    _write_runs(tmp_path, "gfs")
    site = tmp_path / "site"
    meta = {"generated": "2026-10-12T18:17:00Z", "models": ["gfs"], "pairs": 5}
    health.build(tmp_path, site, NOW, meta, {"seconds": 3.2, "errors": {"compare": "boom"}})
    doc = json.loads((site / "health.json").read_text())
    assert doc["build"]["seconds"] == 3.2 and doc["build"]["pairs"] == 5
    assert doc["status"] == "degraded"  # a failed optional step
    # standalone re-run (after analysis/TAFs) keeps the site build details
    (site / "meta.json").write_text(json.dumps(meta))
    health.build(tmp_path, site, NOW)
    doc2 = json.loads((site / "health.json").read_text())
    assert doc2["build"]["seconds"] == 3.2
    assert any(r["component"] == "build" for r in doc2["reasons"])
