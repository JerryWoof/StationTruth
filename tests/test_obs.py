from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stationtruth import obs

NOW = datetime(2026, 10, 12, 16, 7, tzinfo=timezone.utc)

CSV = """No errors
No warnings
3 ms
data source=metars
4 results
raw_text,station_id,observation_time,latitude,longitude,temp_c,sky_cover,sky_cover,metar_type
"METAR LKPR 121530Z 29005KT 260V330 CAVOK 12/05 Q1022 NOSIG",LKPR,2026-10-12T15:30:00.000Z,50.1,14.26,12,,,METAR
"KJFK 121551Z 35010G18KT 10SM BKN031 18/13 A2981 RMK AO2 SLP094 T01780128",KJFK,2026-10-12T15:51:00.000Z,40.6,-73.7,18,,,METAR
"SPECI CYBW 121538Z AUTO VRB04KT 1/4SM FZFG VV003 M04/M04 A2996 RMK SLP202",CYBW,2026-10-12T15:38:00.000Z,51.1,-114.3,-4,,,SPECI
"UUEE 121530Z 18004MPS 9999 OVC008 08/07 Q1015 R24L/290050 NOSIG",UUEE,2026-10-12T15:30:00.000Z,55.9,37.4,8,,,METAR
"EGLL 311550Z NIL",EGLL,,51.4,-0.4,,,,METAR
"""  # noqa: E501


def test_parse_cache_csv_rows_and_units():
    df = obs.parse_metar_cache_csv(CSV, now=NOW)
    assert list(df.columns) == obs.COLUMNS
    assert set(df.station) == {"LKPR", "KJFK", "CYBW", "UUEE"}
    assert str(df["time"].dtype) == "datetime64[ns, UTC]"
    for col in obs.FLOAT_COLUMNS:
        assert df[col].dtype == np.float32
    r = df.set_index("station")

    lkpr = r.loc["LKPR"]
    assert lkpr.time == pd.Timestamp("2026-10-12T15:30Z")
    assert lkpr.type == "METAR" and lkpr.t == 12 and lkpr.td == 5
    assert lkpr.vis_m == 10000 and np.isnan(lkpr.ceil_ft) and lkpr.qnh == 1022
    assert lkpr.wdir == 290 and lkpr.wspd == 5 and np.isnan(lkpr.gust)
    assert np.isnan(lkpr.slp) and lkpr["cat"] == "VFR"

    jfk = r.loc["KJFK"]
    assert jfk.t == pytest.approx(17.8) and jfk.td == pytest.approx(12.8)  # T-group wins
    assert jfk.gust == 18 and jfk.slp == pytest.approx(1009.4)
    assert jfk.qnh == pytest.approx(1009.5, abs=0.1) and jfk.ceil_ft == 3100
    assert jfk.vis_m == pytest.approx(16093.4, abs=0.1)

    cybw = r.loc["CYBW"]
    assert cybw.type == "SPECI" and np.isnan(cybw.wdir) and cybw.wspd == 4
    assert cybw.ceil_ft == 300 and cybw["cat"] == "LIFR" and cybw.t == -4

    uuee = r.loc["UUEE"]
    assert uuee.wspd == pytest.approx(7.8, abs=0.05)  # 4 m/s -> kt
    assert uuee.vis_m == 10000 and uuee.ceil_ft == 800


def test_calm_wind_has_no_direction():
    df = obs.decode_reports([("METAR LKPR 121530Z 00000KT 9999 SCT020 12/05 Q1022", NOW, None)])
    assert df.wspd.iloc[0] == 0 and np.isnan(df.wdir.iloc[0])


def test_time_resolution_across_month_boundary():
    ref = datetime(2026, 11, 1, 0, 10, tzinfo=timezone.utc)
    df = obs.decode_reports(
        [("METAR LKPR 312350Z 27005KT 9999 FEW020 05/02 Q1015", ref, None)], now=ref
    )
    assert df.time.iloc[0] == pd.Timestamp("2026-10-31T23:50Z")


def test_future_reports_dropped():
    df = obs.decode_reports(
        [("METAR LKPR 122350Z 27005KT 9999 FEW020 05/02 Q1015", NOW, None)], now=NOW
    )
    assert df.empty


def test_merge_obs_dedupes_new_wins():
    a = obs.decode_reports([("METAR LKPR 121530Z 29005KT CAVOK 12/05 Q1022", NOW, None)])
    b = obs.decode_reports(
        [
            ("METAR LKPR 121530Z 29005KT CAVOK 13/05 Q1022", NOW, None),
            ("METAR LKPR 121600Z 29005KT CAVOK 11/05 Q1022", NOW, None),
        ]
    )
    m = obs.merge_obs(a, b)
    assert len(m) == 2 and m.t.tolist() == [13.0, 11.0]
    assert obs.merge_obs(None, None).empty
    assert len(obs.merge_obs(a, None)) == 1


def test_split_by_day_and_roundtrip(tmp_path):
    ref = datetime(2026, 10, 13, 0, 30, tzinfo=timezone.utc)
    df = obs.decode_reports(
        [
            ("METAR LKPR 122350Z 29005KT CAVOK 12/05 Q1022", ref, None),
            ("METAR LKPR 130020Z 29005KT CAVOK 11/05 Q1022", ref, None),
        ],
        now=ref,
    )
    parts = obs.split_by_day(df)
    assert set(parts) == {date(2026, 10, 12), date(2026, 10, 13)}
    path = obs.write_obs(
        parts[date(2026, 10, 12)], tmp_path / obs.obs_file_name(date(2026, 10, 12))
    )
    assert path.name == "obs-2026-10-12.parquet"
    back = obs.read_obs(path)
    assert len(back) == 1 and str(back.time.dtype) == "datetime64[ns, UTC]"


def _api_line(station, when: datetime, t=10):
    return f"METAR {station} {when:%d%H%M}Z 27005KT 9999 FEW020 {t:02d}/05 Q1015"


def test_fetch_history_paginates_backwards():
    end = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    start = end - timedelta(hours=10)
    ids = [f"S{i:03d}" for i in range(50)]
    # every station reports every 10 minutes -> 50*6*10 = 3000 reports
    all_reports = []
    t = end
    while t > start - timedelta(hours=2):
        for s in ids:
            all_reports.append((t, _api_line(s, t)))
        t -= timedelta(minutes=10)
    calls = []

    def getter(url):
        q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
        wend = datetime.strptime(q["date"], "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
        wstart = wend - timedelta(hours=float(q["hours"]))
        calls.append(url)
        hits = [ln for when, ln in all_reports if wstart <= when <= wend]
        return "\n".join(hits[: obs.API_MAX_RESULTS]).encode()

    stats = {}
    df = obs.fetch_history(ids, start, end, chunk=50, pause=0, log=None, getter=getter, stats=stats)
    expected = sum(1 for when, _ in all_reports if start <= when <= end)
    assert len(df) == expected
    assert df.time.min() >= pd.Timestamp(start) and df.time.max() == pd.Timestamp(end)
    assert stats["requests"] == len(calls) >= expected // obs.API_MAX_RESULTS


def test_fetch_history_chunks_ids():
    end = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    seen = []

    def getter(url):
        ids = url.split("ids=", 1)[1].split("&", 1)[0].split(",")
        seen.append(len(ids))
        if not url.endswith(f"date={end:%Y%m%d_%H%M}"):
            return b""
        return "\n".join(_api_line(s, end) for s in ids).encode()

    df = obs.fetch_history(
        [f"S{i:03d}" for i in range(450)],
        end - timedelta(hours=1),
        end,
        chunk=400,
        pause=0,
        log=None,
        getter=getter,
    )
    # 400 reports in one minute fill a page: one extra request steps past that minute
    assert seen == [400, 400, 50] and len(df) == 450
    with pytest.raises(ValueError):
        obs.fetch_history_chunk([f"S{i}" for i in range(401)], end, end, getter=getter)


def test_fetch_history_parallel_workers_same_result():
    end = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    ids = [f"S{i:03d}" for i in range(30)]

    def getter(url):
        q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
        wend = datetime.strptime(q["date"], "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
        wstart = wend - timedelta(hours=float(q["hours"]))
        lines = []
        t = wend.replace(minute=wend.minute - wend.minute % 10)
        while t >= wstart:
            lines += [_api_line(s, t) for s in q["ids"].split(",")]
            t -= timedelta(minutes=10)
        return "\n".join(lines[: obs.API_MAX_RESULTS]).encode()

    kw = dict(chunk=7, pause=0, log=None, getter=getter)
    one = obs.fetch_history(ids, end - timedelta(hours=6), end, workers=1, **kw)
    stats = {}
    many = obs.fetch_history(ids, end - timedelta(hours=6), end, workers=4, stats=stats, **kw)
    pd.testing.assert_frame_equal(one, many)
    assert len(one) == 30 * 37 and stats["requests"] >= 5
