from __future__ import annotations

import gzip
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from metar_parse import parse_taf

from stationtruth import storage, tafs

UTC = timezone.utc
REF = datetime(2026, 10, 12, 5, 0, tzinfo=UTC)


def expand(raw: str, ref: datetime = REF) -> pd.DataFrame:
    return tafs.to_frame(tafs.expand_taf(parse_taf(raw), ref)).set_index("hour")


def at(df: pd.DataFrame, hh: int, day: int = 12) -> pd.Series:
    return df.loc[pd.Timestamp(datetime(2026, 10, day, hh, tzinfo=UTC))]


def test_base_hours_lead_and_columns():
    df = expand("TAF LKPR 120500Z 1206/1306 24010G20KT 9999 SCT040")
    assert len(df) == 24
    first = at(df, 6)
    assert first.lead_h == 1 and first.prev_cat == "VFR"
    assert first.prev_vis_m == 10000 and np.isnan(first.prev_ceil_ft)
    assert first.prev_wspd == 10 and first.prev_gust == 20
    assert first.tempo_cat == "" and first.tempo_prob == 0 and not first.amended
    assert at(df, 5, day=13).lead_h == 24
    assert str(df.reset_index()["hour"].dtype) == "datetime64[ns, UTC]"
    assert df["lead_h"].dtype == np.int16


def test_becmg_takes_effect_at_end_of_period():
    df = expand("TAF LKPR 120500Z 1206/1306 24010KT 9999 BKN040 BECMG 1210/1212 3000 BR BKN008")
    for hh in range(6, 12):
        assert at(df, hh).prev_cat == "VFR", hh
    after = at(df, 12)
    assert after.prev_cat == "IFR" and after.prev_vis_m == 3000 and after.prev_ceil_ft == 800
    assert after.prev_wspd == 10  # not mentioned in BECMG: kept


def test_becmg_cavok_clears_cloud_and_order_by_time():
    raw = (
        "TAF LKPR 120500Z 1206/1306 24010KT 4000 BR OVC006 "
        "BECMG 1212/1214 CAVOK BECMG 1208/1210 2000 BR"
    )
    df = expand(raw)
    assert at(df, 9).prev_cat == "IFR"
    assert at(df, 10).prev_cat == "IFR" and at(df, 10).prev_vis_m == 2000
    assert at(df, 14).prev_cat == "VFR" and np.isnan(at(df, 14).prev_ceil_ft)
    assert at(df, 14).prev_vis_m == 10000


def test_fm_overrides_everything_before_it():
    raw = (
        "TAF KXYZ 120520Z 1206/1312 24010KT P6SM SCT050 BECMG 1208/1209 BKN020 "
        "FM121500 18015G25KT 1/2SM FG VV002 FM130000 27008KT P6SM SKC"
    )
    df = expand(raw)
    assert at(df, 8).prev_cat == "VFR"
    assert at(df, 9).prev_cat == "MVFR"
    fm = at(df, 15)
    assert fm.prev_cat == "LIFR" and fm.prev_ceil_ft == 200
    assert fm.prev_wspd == 15 and fm.prev_gust == 25
    assert at(df, 14).prev_cat == "MVFR"
    assert at(df, 0, day=13).prev_cat == "VFR" and np.isnan(at(df, 0, day=13).prev_gust)
    assert at(df, 11, day=13).lead_h == 29


def test_tempo_prob_recorded_separately():
    raw = (
        "TAF LKPR 120500Z 1206/1306 24010KT 9999 BKN040 TEMPO 1208/1210 1500 BR OVC005 "
        "PROB30 TEMPO 1209/1211 6000 SHRA BKN012 PROB40 1300/1303 BKN008"
    )
    df = expand(raw)
    assert (df["prev_cat"] == "VFR").all()  # temporary groups never change prevailing
    assert at(df, 7).tempo_cat == "" and at(df, 7).tempo_prob == 0
    assert at(df, 8).tempo_cat == "LIFR" and at(df, 8).tempo_prob == 100
    h9 = at(df, 9)
    assert h9.tempo_cat == "LIFR" and h9.tempo_prob == 100  # worst / max of both groups
    h10 = at(df, 10)  # TEMPO ended at 10Z; PROB30 TEMPO still active
    assert h10.tempo_cat == "MVFR" and h10.tempo_prob == 30
    assert at(df, 11).tempo_cat == ""
    h0 = at(df, 0, day=13)
    assert h0.tempo_cat == "IFR" and h0.tempo_prob == 40


def test_tempo_overlays_prevailing_elements():
    # TEMPO only lowers visibility; the prevailing low ceiling still counts.
    df = expand("TAF LKPR 120500Z 1206/1306 24010KT 9999 OVC015 TEMPO 1206/1208 6000 -RA")
    assert at(df, 6).prev_cat == "MVFR" and at(df, 6).tempo_cat == "MVFR"


def test_amendment_skips_hours_before_issue():
    df = expand("TAF AMD LKPR 120735Z 1206/1306 24010KT 9999 FEW030")
    assert df.index.min() == pd.Timestamp("2026-10-12T08:00Z")
    assert at(df, 8).lead_h == 0 and bool(at(df, 8).amended)


def test_unusable_tafs_give_no_rows():
    assert tafs.expand_taf(parse_taf("TAF LKPR 120500Z NIL"), REF) == []
    assert tafs.expand_taf(parse_taf("TAF LKPR 120500Z 1206/1306 CNL"), REF) == []
    assert tafs.expand_taf(parse_taf("garbage"), REF) == []


def test_worst():
    assert tafs.worst(["VFR", "", "IFR", "MVFR"]) == "IFR"
    assert tafs.worst(["", "x"]) == ""


CACHE_XML = b"""<?xml version="1.0" encoding="UTF-8"?><response><data num_results="2">
<TAF><raw_text><![CDATA[TAF AMD EGKK 121805Z 1218/1324 20009KT 9999 SCT008 BKN030 PROB40 TEMPO 1220/1301 8000 BKN008]]></raw_text>
<station_id>EGKK</station_id><issue_time>2026-10-12T18:05:00.000Z</issue_time></TAF>
<TAF><raw_text><![CDATA[KJFK 121720Z 1218/1324 22013KT P6SM FEW100]]></raw_text>
<station_id>KJFK</station_id><issue_time>2026-10-12T17:20:00.000Z</issue_time></TAF>
</data></response>"""  # noqa: E501


def test_parse_cache_xml_and_expand():
    reps = tafs.parse_cache_xml(CACHE_XML)
    assert [r[1] for r in reps] == [
        datetime(2026, 10, 12, 18, 5, tzinfo=UTC),
        datetime(2026, 10, 12, 17, 20, tzinfo=UTC),
    ]
    assert reps[1][0].startswith("TAF KJFK")
    df = tafs.expand_reports(reps)
    assert set(df.station) == {"EGKK", "KJFK"}
    egkk = df[df.station == "EGKK"].set_index("hour")
    assert egkk.loc[pd.Timestamp("2026-10-12T20:00Z")].tempo_prob == 40
    assert egkk["amended"].all()


def test_fetch_cache_gzip(monkeypatch):
    monkeypatch.setattr(tafs.http, "get", lambda url, **kw: gzip.compress(CACHE_XML))
    df = tafs.fetch_cache()
    assert df["station"].nunique() == 2


RAW_API = """TAF EGKK 101053Z 1012/1118 25009KT 9999 FEW040
  PROB30
  TEMPO 1012/1017 27015G25KT 9000 -SHRA
TAF KJFK 101120Z 1012/1118 22013KT P6SM FEW100
  FM101800 27008KT P6SM SKC
"""


def test_split_raw_tafs_rejoins_prob_lines():
    parts = tafs.split_raw_tafs(RAW_API)
    assert len(parts) == 2
    assert "PROB30 TEMPO 1012/1017" in parts[0]
    assert parts[1].endswith("FM101800 27008KT P6SM SKC")


def test_fetch_history_walks_snapshots_and_dedupes():
    urls = []

    def getter(url):
        urls.append(url)
        return RAW_API.encode()

    df = tafs.fetch_history(
        ["EGKK", "KJFK"],
        datetime(2026, 10, 10, 12, tzinfo=UTC),
        datetime(2026, 10, 10, 15, tzinfo=UTC),
        pause=0,
        workers=1,
        log=None,
        getter=getter,
    )
    assert len(urls) == 4 and all("ids=EGKK,KJFK" in u for u in urls)
    assert "date=20261010_1500" in urls[0]
    assert len(df[["station", "issued"]].drop_duplicates()) == 2
    with pytest.raises(ValueError):
        tafs.fetch_at([f"K{i:03d}" for i in range(401)], REF, getter=getter)


def test_fetch_at_raises_on_api_error():
    with pytest.raises(RuntimeError):
        tafs.fetch_at(["EGKK"], REF, getter=lambda u: b"Unexpected query parameter provided")


def test_merge_replaces_same_taf_and_store_roundtrip(tmp_path):
    a = tafs.expand_reports([("TAF LKPR 120500Z 1206/1306 24010KT 9999 FEW030", REF)])
    b = tafs.expand_reports([("TAF LKPR 120500Z 1206/1212 24010KT 0500 FG", REF)])
    c = tafs.expand_reports([("TAF LKPR 121100Z 1212/1312 24010KT 9999 FEW030", REF)])
    m = tafs.merge(a, b)
    assert len(m) == 6 and (m.prev_cat == "LIFR").all()
    m = tafs.merge(m, c)
    assert m["issued"].nunique() == 2

    st = storage.LocalStorage(tmp_path / "store")
    work = tmp_path / "work"
    work.mkdir()
    assert tafs.store(st, a, work, log=None) == 1
    assert tafs.store(st, tafs.merge(b, c), work, log=None) == 1
    stored = tafs.read_tafs(st.root / tafs.taf_file_name(date(2026, 10, 12)))
    assert stored["issued"].nunique() == 2 and len(stored[stored.prev_cat == "LIFR"]) == 6
    assert tafs.store(st, a, work, days={date(2026, 10, 1)}, log=None) == 0


def test_storage_knows_taf_files():
    assert storage.month_of("taf-2026-10-12.parquet") == "2026-10"
    assert storage.parse_name("taf-2026-10-12.parquet") == {
        "kind": "taf",
        "date": date(2026, 10, 12),
    }
    got = storage.names_for_window(
        ["taf-2026-10-02.parquet", "taf-2026-10-03.parquet", "taf-2026-10-13.parquet"],
        datetime(2026, 10, 5, tzinfo=UTC),
        datetime(2026, 10, 12, 16, tzinfo=UTC),
    )
    assert got == ["taf-2026-10-03.parquet"]


def test_main_collect(tmp_path, monkeypatch):
    monkeypatch.setattr(tafs.http, "get", lambda url, **kw: gzip.compress(CACHE_XML))
    now = datetime(2026, 10, 12, 18, 30, tzinfo=UTC)
    st = storage.LocalStorage(tmp_path)
    assert tafs.collect(st, now=now, log=None) == 2
    assert (tmp_path / "taf-2026-10-12.parquet").exists()
    assert tafs.main(["collect", "--data-dir", str(tmp_path)]) == 0
