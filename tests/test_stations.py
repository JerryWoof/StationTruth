from __future__ import annotations

import gzip
import json

import numpy as np

from stationtruth import stations

RECORDS = [
    {
        "icaoId": "LKPR",
        "site": "Prague/Havel Arpt",
        "lat": 50.101,
        "lon": 14.26,
        "elev": 364,
        "country": "CZ",
        "siteType": ["METAR", "TAF"],
    },
    {
        "icaoId": "PADK",
        "site": "Adak",
        "lat": 51.88,
        "lon": 180.0,
        "elev": None,
        "country": "US",
        "siteType": ["METAR"],
    },
    {
        "icaoId": None,
        "site": "Buoy",
        "lat": 30,
        "lon": -80,
        "elev": 0,
        "country": None,
        "siteType": [],
    },
    {
        "icaoId": "XTAF",
        "site": "TAF only",
        "lat": 1,
        "lon": 1,
        "elev": 0,
        "country": "XX",
        "siteType": ["TAF"],
    },
    {
        "icaoId": "XBAD",
        "site": "No coords",
        "lat": None,
        "lon": 1,
        "elev": 0,
        "country": "XX",
        "siteType": ["METAR"],
    },
    {
        "icaoId": "ZZZZ",
        "site": None,
        "lat": -45.0,
        "lon": -170.5,
        "elev": 12,
        "country": None,
        "siteType": ["METAR"],
    },
]


def test_parse_stations_filters_and_types():
    df = stations.parse_stations(RECORDS)
    assert list(df.columns) == stations.COLUMNS
    assert df.station.tolist() == ["LKPR", "PADK", "ZZZZ"]
    assert df.lat.dtype == np.float64 and df.lon.dtype == np.float64
    assert df.elev_m.dtype == np.float32
    r = df.set_index("station")
    assert r.loc["PADK"].lon == -180.0  # 180 wraps into [-180, 180)
    assert np.isnan(r.loc["PADK"].elev_m)
    assert r.loc["ZZZZ"].country == "" and r.loc["ZZZZ"]["name"] == ""
    assert r.loc["LKPR"].elev_m == 364


def test_fetch_stations_decompresses(monkeypatch, tmp_path):
    payload = gzip.compress(json.dumps(RECORDS).encode())
    monkeypatch.setattr(stations.http, "get", lambda url, **kw: payload)
    df = stations.fetch_stations()
    assert len(df) == 3
    path = stations.write_stations(df, tmp_path / stations.STATIONS_FILE)
    back = stations.read_stations(path)
    assert back.station.tolist() == df.station.tolist()
