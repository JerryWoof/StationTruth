from __future__ import annotations

from datetime import datetime, timezone

from stationtruth import models

RUN = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)

ECMWF_INDEX = "\n".join(
    [
        '{"levtype": "sfc", "step": "3", "param": "tp", "_offset": 0, "_length": 10}',
        '{"levtype": "sfc", "step": "3", "param": "msl", "_offset": 10, "_length": 5}',
        '{"levtype": "pl", "levelist": "850", "param": "t", "_offset": 15, "_length": 5}',
        '{"levtype": "sfc", "step": "3", "param": "2t", "_offset": 20, "_length": 5}',
        '{"levtype": "sfc", "step": "3", "param": "2d", "_offset": 25, "_length": 5}',
        '{"levtype": "sfc", "step": "3", "param": "10u", "_offset": 30, "_length": 5}',
        '{"levtype": "sfc", "step": "3", "param": "10v", "_offset": 35, "_length": 5}',
        '{"levtype": "sfc", "step": "3", "param": "10fg", "_offset": 40, "_length": 5}',
        '{"levtype": "sfc", "step": "3", "param": "tcc", "_offset": 45, "_length": 5}',
        "",
    ]
)

GFS_IDX = """1:0:d=2026101212:PRMSL:mean sea level:3 hour fcst:
2:100:d=2026101212:VIS:surface:3 hour fcst:
3:200:d=2026101212:GUST:surface:3 hour fcst:
4:300:d=2026101212:TMP:2 m above ground:3 hour fcst:
5:400:d=2026101212:DPT:2 m above ground:3 hour fcst:
6:500:d=2026101212:UGRD:10 m above ground:3 hour fcst:
7:600:d=2026101212:VGRD:10 m above ground:3 hour fcst:
8:700:d=2026101212:TCDC:entire atmosphere:0-3 hour ave fcst:
9:800:d=2026101212:TCDC:entire atmosphere:3 hour fcst:
"""


def test_urls():
    assert models.ECMWF.grib_url(RUN, 72) == (
        "https://data.ecmwf.int/forecasts/20261012/12z/ifs/0p25/oper/"
        "20261012120000-72h-oper-fc.grib2"
    )
    assert models.ECMWF.index_url(RUN, 0).endswith("20261012120000-0h-oper-fc.index")
    assert models.GFS.grib_url(RUN, 3) == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261012/12/atmos/gfs.t12z.pgrb2.0p25.f003"
    )
    assert models.GFS.index_url(RUN, 72).endswith(".f072.idx")
    assert models.ECMWF.file_name(RUN) == "fcst-ecmwf-2026101212.parquet"
    assert models.STEPS[0] == 0 and models.STEPS[-1] == 72 and len(models.STEPS) == 25


def test_ecmwf_select():
    entries = models.ECMWF.parse_index(ECMWF_INDEX)
    sel = models.ECMWF.select(entries, 3)
    assert set(sel) == {"t", "td", "u", "v", "gust", "mslp", "tcc"}
    assert sel["t"].offset == 20 and sel["t"].length == 5
    assert "gust" not in models.ECMWF.select(entries, 0)  # no max-gust period at step 0


def test_gfs_select_lengths_and_instantaneous_tcc():
    entries = models.GFS.parse_index(GFS_IDX)
    assert entries[0].length == 100 and entries[-1].length is None
    sel = models.GFS.select(entries, 3)
    assert set(sel) == {"t", "td", "u", "v", "gust", "mslp", "tcc", "vis"}
    assert sel["tcc"].offset == 800  # not the 0-3 hour average


def test_gfs_select_analysis():
    idx = GFS_IDX.replace("3 hour fcst", "anl").replace(
        "9:800:d=2026101212:TCDC:entire atmosphere:anl:\n", ""
    )
    sel = models.GFS.select(models.GFS.parse_index(idx), 0)
    assert "t" in sel and "tcc" not in sel


def test_get_source():
    assert models.get_source("gfs") is models.GFS
    try:
        models.get_source("icon")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError
