from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stationtruth import tafs, tafverify

UTC = timezone.utc
T0 = datetime(2026, 10, 12, 6, tzinfo=UTC)


def obs_frame(rows):
    """rows: (station, time, type, cat)."""
    return pd.DataFrame(rows, columns=["station", "time", "type", "cat"]).assign(
        time=lambda d: pd.to_datetime(d["time"], utc=True)
    )


def taf_rows(station, issued, hours, prev, tempo=None, lead0=1):
    tempo = tempo or [""] * len(prev)
    return [
        {
            "station": station,
            "issued": issued,
            "hour": T0 + timedelta(hours=k),
            "lead_h": lead0 + k,
            "prev_cat": p,
            "prev_vis_m": np.nan,
            "prev_ceil_ft": np.nan,
            "prev_wspd": np.nan,
            "prev_gust": np.nan,
            "tempo_cat": t,
            "tempo_prob": 100 if t else 0,
            "amended": False,
            "raw": "",
        }
        for k, p, t in zip(range(hours), prev, tempo, strict=True)
    ]


def test_event_scores_math():
    s = tafverify.event_scores(30, 10, 20, 940)
    assert s["pod"] == pytest.approx(0.6)
    assert s["far"] == pytest.approx(0.25)
    assert s["csi"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(0.8)
    empty = tafverify.event_scores(0, 0, 0, 100)
    assert empty["pod"] is None and empty["far"] is None and empty["csi"] is None


def test_heidke():
    perfect = np.diag([10, 5, 3, 2])
    assert tafverify.heidke(perfect) == 1.0
    chance = np.outer([1, 1, 1, 1], [1, 1, 1, 1]) * 5  # forecast independent of obs
    assert tafverify.heidke(chance) == pytest.approx(0.0)
    assert tafverify.heidke(np.zeros((4, 4))) is None


def test_observed_hours_worst_and_nominal():
    o = obs_frame(
        [
            ("LKPR", "2026-10-12T05:50Z", "METAR", "VFR"),  # nominal for 06Z (10 min)
            ("LKPR", "2026-10-12T06:20Z", "SPECI", "IFR"),  # worst in 06Z hour
            ("LKPR", "2026-10-12T06:50Z", "METAR", "MVFR"),  # nominal for 07Z
            ("LKPR", "2026-10-12T07:10Z", "SPECI", ""),  # unknown category: ignored
            ("KJFK", "2026-10-12T06:51Z", "METAR", "LIFR"),
        ]
    )
    oh = tafverify.observed_hours(o).set_index(["station", "hour"])
    h6 = oh.loc[("LKPR", pd.Timestamp("2026-10-12T06:00Z"))]
    assert h6.worst == 2 and h6.nominal == 0  # IFR worst, VFR at :50 the hour before
    h7 = oh.loc[("LKPR", pd.Timestamp("2026-10-12T07:00Z"))]
    assert h7.worst == -1 and h7.nominal == 1
    h5 = oh.loc[("LKPR", pd.Timestamp("2026-10-12T05:00Z"))]
    assert h5.worst == 0 and h5.nominal == -1


def test_compute_contingency_and_tempo_credit():
    # Station A, hours 06..09: forecast VFR, VFR, IFR, MVFR (TEMPO IFR on 07Z and 09Z)
    rows = taf_rows(
        "LKPR",
        T0 - timedelta(hours=1),
        4,
        ["VFR", "VFR", "IFR", "MVFR"],
        ["", "IFR", "", "LIFR"],
    )
    # observed worst per hour: VFR, IFR, MVFR, LIFR
    o = obs_frame(
        [
            ("LKPR", "2026-10-12T06:00Z", "METAR", "VFR"),
            ("LKPR", "2026-10-12T07:00Z", "METAR", "VFR"),
            ("LKPR", "2026-10-12T07:30Z", "SPECI", "IFR"),
            ("LKPR", "2026-10-12T08:00Z", "METAR", "MVFR"),
            ("LKPR", "2026-10-12T09:00Z", "METAR", "LIFR"),
            ("LKPR", "2026-10-12T12:00Z", "METAR", "VFR"),
        ]
    )
    st = pd.DataFrame(
        {"station": ["LKPR"], "name": ["Prague"], "country": ["CZ"], "lat": [50.1], "lon": [14.3]}
    )
    now = datetime(2026, 10, 12, 13, tzinfo=UTC)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tafverify, "STATION_LIST_MIN_N", 1)
        doc = tafverify.compute(tafs.to_frame(rows), o, st, now=now)
    assert doc["taf_count"] == 1 and doc["first_issued"] == "2026-10-12T05:00:00Z"
    w = doc["windows"]["7d"]
    g = w["global"]["worst"]
    assert g["n"] == 4
    t = np.array(g["table"])  # rows forecast, cols observed: VFR MVFR IFR LIFR
    assert t[0, 0] == 1 and t[0, 2] == 1 and t[2, 1] == 1 and t[1, 3] == 1 and t.sum() == 4
    assert g["hit_rate"] == 0.25
    # IFR-or-worse, prevailing only: hit none; FA 08Z; misses 07Z, 09Z
    assert (g["ifr"]["a"], g["ifr"]["b"], g["ifr"]["c"], g["ifr"]["d"]) == (0, 1, 2, 1)
    assert g["ifr"]["pod"] == 0.0 and g["ifr"]["far"] == 1.0 and g["ifr"]["csi"] == 0.0
    # with TEMPO: 07Z and 09Z become hits
    it = g["ifr_tempo"]
    assert (it["a"], it["b"], it["c"], it["d"]) == (2, 1, 0, 1)
    assert (
        it["pod"] == 1.0 and it["far"] == pytest.approx(0.333) and it["csi"] == pytest.approx(0.667)
    )
    assert it["bias"] == 1.5
    # nominal: routine METAR at the hour -> VFR, VFR, MVFR, LIFR
    nom = w["global"]["nominal"]
    assert nom["ifr"]["c"] == 1 and nom["n"] == 4
    # lead buckets: leads 1..4 -> all in 0-6
    assert w["global"]["worst"]["by_lead"][0]["n"] == 4
    assert w["global"]["worst"]["by_lead"][1]["n"] == 0
    assert list(w["regions"]) == ["Europe"]
    row = dict(zip(doc["station_fields"], w["stations"][0], strict=True))
    assert row == {
        "id": "LKPR",
        "n": 4,
        "hits": 1,
        "a": 0,
        "b": 1,
        "c": 2,
        "a_t": 2,
        "b_t": 1,
        "c_t": 0,
    }
    assert doc["stations"] == {"LKPR": ["Prague", "CZ", "Europe"]}


def test_compute_empty_and_lead_bucket():
    doc = tafverify.compute(tafs.empty_frame(), obs_frame([]), pd.DataFrame())
    assert doc["windows"] == {} and doc["taf_count"] == 0 and doc["first_issued"] is None
    lb = tafverify.lead_bucket(np.array([0, 5, 6, 11, 12, 23, 24, 29, 31]))
    assert lb.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]


def test_build_writes_json(tmp_path):
    rows = taf_rows("LKPR", T0 - timedelta(hours=1), 3, ["VFR", "IFR", "VFR"])
    tafs.write_tafs(tafs.to_frame(rows), tmp_path / "taf-2026-10-12.parquet")
    o = obs_frame(
        [
            ("LKPR", f"2026-10-12T{h:02d}:00Z", "METAR", c)
            for h, c in [(6, "VFR"), (7, "IFR"), (8, "VFR"), (10, "VFR")]
        ]
    )
    o.assign(t=np.nan).to_parquet(tmp_path / "obs-2026-10-12.parquet")
    out = tmp_path / "site"
    now = datetime(2026, 10, 12, 11, tzinfo=UTC)
    summary = tafverify.build(tmp_path, out, now=now)
    doc = json.loads((out / "taf_scores.json").read_text())
    assert summary["7d"]["csi"] == 1.0
    assert doc["windows"]["7d"]["global"]["worst"]["hit_rate"] == 1.0
    assert doc["last_hour"] == "2026-10-12T08:00:00Z"
    assert tafverify.main(["--data-dir", str(tmp_path), "--out", str(out)]) == 0
