from __future__ import annotations

import json
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
        models.get_source("nam")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError


AIFS_INDEX = "\n".join(
    json.dumps({"type": "fc", "levtype": lt, "step": "6", "param": p, "_offset": o, "_length": n})
    if lt == "sfc"
    else json.dumps({"type": "fc", "levtype": lt, "levelist": "850", "param": p, "_offset": o})
    for lt, p, o, n in [
        ("sfc", "tcc", 0, 7),
        ("pl", "t", 7, 3),
        ("sfc", "msl", 10, 5),
        ("sfc", "10u", 15, 5),
        ("sfc", "10v", 20, 5),
        ("sfc", "2t", 25, 5),
        ("sfc", "2d", 30, 5),
        ("sfc", "100u", 35, 5),
    ]
)

# real layout of aigfs.tHHz.sfc.fFFF.grib2.idx (2026-10)
AIGFS_IDX = """1:0:d=2026101212:UGRD:100 m above ground:6 hour fcst:
2:902722:d=2026101212:VGRD:100 m above ground:6 hour fcst:
3:1816569:d=2026101212:UGRD:10 m above ground:6 hour fcst:
4:2702550:d=2026101212:VGRD:10 m above ground:6 hour fcst:
5:3580068:d=2026101212:DPT:2 m above ground:6 hour fcst:
6:4069259:d=2026101212:TMP:2 m above ground:6 hour fcst:
7:4551943:d=2026101212:ACPCP:surface:0-6 hour acc fcst:
8:4552177:d=2026101212:PRMSL:mean sea level:6 hour fcst:
9:5577248:d=2026101212:TMP:surface:6 hour fcst:
10:6163968:d=2026101212:PRES:surface:6 hour fcst:
"""


def test_aifs_source():
    assert models.AIFS.grib_url(RUN, 6) == (
        "https://data.ecmwf.int/forecasts/20261012/12z/aifs-single/0p25/oper/"
        "20261012120000-6h-oper-fc.grib2"
    )
    assert models.AIFS.availability_url(RUN).endswith("20261012120000-72h-oper-fc.index")
    assert models.AIFS.steps == tuple(range(0, 73, 6))
    assert models.AIFS.file_name(RUN) == "fcst-aifs-2026101212.parquet"
    sel = models.AIFS.select(models.AIFS.parse_index(AIFS_INDEX), 6)
    assert set(sel) == {"t", "td", "u", "v", "mslp", "tcc"}  # no 10fg in AIFS
    assert sel["t"].offset == 25 and sel["tcc"].length == 7
    # IFS itself is unchanged
    assert models.ECMWF.grib_url(RUN, 3).split("/")[6] == "ifs"


def test_aigfs_source():
    assert models.AIGFS.grib_url(RUN, 6) == (
        "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/aigfs.20261012/12/model/atmos/"
        "grib2/aigfs.t12z.sfc.f006.grib2"
    )
    assert models.AIGFS.index_url(RUN, 72).endswith("aigfs.t12z.sfc.f072.grib2.idx")
    entries = models.AIGFS.parse_index(AIGFS_IDX)
    sel = models.AIGFS.select(entries, 6)
    assert set(sel) == {"t", "td", "u", "v", "mslp"}
    assert sel["u"].offset == 1816569 and sel["u"].length == 2702550 - 1816569
    assert sel["t"].offset == 4069259  # 2 m, not the surface (skin) temperature
    anl = AIGFS_IDX.replace("6 hour fcst", "anl").replace("0-6 hour acc fcst", "anl")
    assert set(models.AIGFS.select(models.AIGFS.parse_index(anl), 0)) == set(sel)


def test_icon_source():
    assert models.ICON.layout == "files"
    assert models.ICON.file_url(RUN, 3, "TD_2M") == (
        "https://opendata.dwd.de/weather/nwp/icon/grib/12/td_2m/"
        "icon_global_icosahedral_single-level_2026101212_003_TD_2M.grib2.bz2"
    )
    assert models.ICON.coord_url(RUN, "CLAT").endswith(
        "/12/clat/icon_global_icosahedral_time-invariant_2026101212_CLAT.grib2.bz2"
    )
    assert "gust" not in dict(models.ICON.files(0)) and "gust" in dict(models.ICON.files(3))
    assert models.ICON.availability_url(RUN).endswith("_072_CLCT.grib2.bz2")


def test_sources_cover_registry():
    from stationtruth import registry

    assert list(models.SOURCES) == list(registry.RAW_MODELS)
    for m in registry.REGISTRY:
        assert models.SOURCES[m.id].steps == m.steps
        assert models.SOURCES[m.id].name == m.id
