from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import eccodes
import numpy as np
import pandas as pd
import pytest

from stationtruth import grib, models

RUN = datetime(2026, 10, 12, 0, tzinfo=timezone.utc)


def make_grib(
    values: np.ndarray,
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
    di: float,
    dj: float,
    *,
    j_positive: bool = False,
    bitmap_missing: bool = False,
) -> bytes:
    """A small regular_ll GRIB2 message (values shaped (nj, ni), row-major as scanned)."""
    nj, ni = values.shape
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set(h, "Ni", ni)
        eccodes.codes_set(h, "Nj", nj)
        eccodes.codes_set(h, "jScansPositively", int(j_positive))
        eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", lat0)
        eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", lon0)
        eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", lat1)
        eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", lon1)
        eccodes.codes_set(h, "iDirectionIncrementInDegrees", di)
        eccodes.codes_set(h, "jDirectionIncrementInDegrees", dj)
        eccodes.codes_set(h, "bitsPerValue", 24)
        flat = values.astype("float64").ravel()
        if bitmap_missing:
            eccodes.codes_set(h, "bitmapPresent", 1)
            eccodes.codes_set(h, "missingValue", 9999.0)
            flat = np.where(np.isnan(flat), 9999.0, flat)
        eccodes.codes_set_values(h, flat)
        return eccodes.codes_get_message(h)
    finally:
        eccodes.codes_release(h)


def gfs_like(fn, step=1.0):
    """Global grid, lat 90..-90, lon 0..360-step, values = fn(lat, lon)."""
    lats = np.arange(90, -90 - 1e-9, -step)
    lons = np.arange(0, 360 - 1e-9, step)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return make_grib(fn(la, lo), 90, 0, -90, 360 - step, step, step)


def ecmwf_like(fn, step=1.0):
    """Global grid, lat 90..-90, lon -180..180-step."""
    lats = np.arange(90, -90 - 1e-9, -step)
    lons = np.arange(-180, 180 - 1e-9, step)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return make_grib(fn(la, lo), 90, -180, -90, 180 - step, step, step)


def linear(la, lo):
    return 2.0 * la + 0.1 * ((lo + 180.0) % 360.0)  # linear away from lon 180


def test_decode_grid_keys():
    field, grid, _ = grib.decode_message(gfs_like(linear))
    assert field.shape == (181, 360)
    assert grid.ni == 360 and grid.nj == 181
    assert grid.lat0 == 90 and grid.lon0 == 0
    assert grid.di == pytest.approx(1.0) and grid.dj == pytest.approx(1.0)
    assert grid.is_global_lon and not grid.j_positive and not grid.i_negative


@pytest.mark.parametrize("maker", [gfs_like, ecmwf_like])
def test_bilinear_exact_for_linear_field(maker):
    field, grid, _ = grib.decode_message(maker(linear))
    lat = np.array([50.101, -37.3, 0.0, 89.5, -89.9])
    lon = np.array([14.26, 174.79, -73.764, -100.25, 10.0])
    w = grib.bilinear_weights(grid, lat, lon)
    got = w.apply(field)
    np.testing.assert_allclose(got, linear(lat, lon), atol=1e-3)


def test_wraps_longitude_on_0_360_grid():
    # Field equal to the longitude index; interpolating at -0.5 must average 359 and 0.
    def fn(la, lo):
        return np.where(lo == 0, 10.0, np.where(lo == 359, 20.0, 0.0))

    field, grid, _ = grib.decode_message(gfs_like(fn))
    w = grib.bilinear_weights(grid, np.array([45.0, 45.0]), np.array([-0.5, 179.0]))
    got = w.apply(field)
    assert got[0] == pytest.approx(15.0, abs=1e-3)
    assert got[1] == pytest.approx(0.0, abs=1e-3)


def test_south_to_north_and_regional_grid():
    # Regional grid scanning south->north: lat 40..60, lon 0..30, 2 deg.
    lats = np.arange(40, 60 + 1e-9, 2.0)
    lons = np.arange(0, 30 + 1e-9, 2.0)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    msg = make_grib(la * 10 + lo, 40, 0, 60, 30, 2, 2, j_positive=True)
    field, grid, _ = grib.decode_message(msg)
    assert grid.j_positive and not grid.is_global_lon
    w = grib.bilinear_weights(grid, np.array([50.1, 45.0, 70.0]), np.array([14.26, 40.0, 10.0]))
    got = w.apply(field)
    assert got[0] == pytest.approx(501 + 14.26, abs=1e-3)
    assert np.isnan(got[1]) and np.isnan(got[2])  # outside the grid


def test_missing_values_bitmap():
    vals = np.ones((3, 4))
    vals[1, 1] = np.nan
    msg = make_grib(vals, 10, 0, 8, 3, 1, 1, bitmap_missing=True)
    field, _, _ = grib.decode_message(msg)
    assert np.isnan(field[1, 1]) and np.nansum(field) == pytest.approx(11)


def test_wind_dir_convention():
    u = np.array([0.0, -5.0, 0.0, 5.0, 0.0])
    v = np.array([-5.0, 0.0, 5.0, 0.0, 0.0])
    d = grib.wind_dir(u, v)
    np.testing.assert_allclose(d[:4], [0.0, 90.0, 180.0, 270.0], atol=1e-9)  # from N, E, S, W
    assert np.isnan(d[4])


def test_merge_ranges():
    sel = {
        "a": models.IndexEntry(0, 100),
        "b": models.IndexEntry(100, 50),
        "c": models.IndexEntry(1000, 10),
        "d": models.IndexEntry(1015, None),
    }
    rngs = grib.merge_ranges(sel)
    assert [(r.start, r.end) for r in rngs] == [(0, 149), (1000, 1009), (1015, None)]
    rngs = grib.merge_ranges(sel, max_gap=10)
    assert [(r.start, r.end) for r in rngs] == [(0, 149), (1000, None)]
    blob = bytes(range(256)) * 10
    calls = []

    def getter(url, a, b):
        calls.append((a, b))
        return blob[a:] if b is None else blob[a : b + 1]

    out, nreq, nbytes = grib.download_messages("u", sel, max_gap=10, getter=getter)
    assert nreq == 2 and calls == [(0, 149), (1000, None)]
    assert out["b"] == blob[100:150] and out["c"] == blob[1000:1010] and out["d"] == blob[1015:]


def test_fetch_run_units_and_layout():
    stations = pd.DataFrame(
        {"station": ["AAAA", "BBBB"], "lat": [50.0, -10.0], "lon": [14.0, -170.0]}
    )
    const = {
        "t": 288.15,
        "td": 278.15,
        "u": 0.0,
        "v": -10.0,  # from the north
        "gust": 10.0,
        "mslp": 101325.0,
        "tcc": 0.5,  # fraction -> 50 %
    }
    msgs = {
        k: ecmwf_like(lambda la, lo, c=c: np.full(la.shape, c), step=2.0) for k, c in const.items()
    }
    src = dataclasses.replace(models.ECMWF, steps=(0, 3))

    def fake_step(source, run, step, max_gap):
        sel = {k: v for k, v in msgs.items() if not (k == "gust" and step == 0)}
        return grib.StepResult(step, sel, len(sel), sum(map(len, sel.values())))

    df, stats = grib.fetch_run(src, RUN, stations, fetch_step=fake_step, log=None)
    assert list(df.columns) == grib.FCST_COLUMNS
    assert len(df) == 4 and stats["failed_steps"] == []
    assert str(df["run"].dtype) == "datetime64[ns, UTC]"
    assert df["lead_h"].dtype == np.int16 and df["t"].dtype == np.float32
    row = df[(df.station == "AAAA") & (df.lead_h == 3)].iloc[0]
    assert row.valid == pd.Timestamp("2026-10-12T03:00Z")
    assert row.t == pytest.approx(15.0, abs=0.01)
    assert row.td == pytest.approx(5.0, abs=0.01)
    assert row.wspd == pytest.approx(19.44, abs=0.01)
    assert row.wdir == pytest.approx(0.0, abs=0.01) or row.wdir == pytest.approx(360.0, abs=0.01)
    assert row.gust == pytest.approx(19.44, abs=0.01)
    assert row.mslp == pytest.approx(1013.25, abs=0.01)
    assert row.tcc == pytest.approx(50.0, abs=0.01)
    assert np.isnan(row.vis_m)
    step0 = df[df.lead_h == 0]
    assert step0["gust"].isna().all()


def test_fetch_run_reports_failed_steps():
    stations = pd.DataFrame({"station": ["AAAA"], "lat": [0.0], "lon": [0.0]})
    src = dataclasses.replace(models.GFS, steps=(0, 3))

    def fake_step(source, run, step, max_gap):
        return grib.StepResult(step, {}, 0, 0, "boom")

    df, stats = grib.fetch_run(src, RUN, stations, fetch_step=fake_step, log=None)
    assert df.empty and stats["failed_steps"] == [0, 3]


def test_candidate_runs():
    now = datetime(2026, 10, 12, 13, 30, tzinfo=timezone.utc)
    runs = grib.candidate_runs(models.ECMWF, now, lookback_h=48)
    assert runs[0] == datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    assert runs[-1] == datetime(2026, 10, 11, 0, tzinfo=timezone.utc)  # 10th 12Z is > 48 h ago
    assert len(runs) == 4


def test_write_fcst_roundtrip(tmp_path):
    stations = pd.DataFrame({"station": ["AAAA"], "lat": [0.0], "lon": [0.0]})
    msg = ecmwf_like(lambda la, lo: np.full(la.shape, 290.0), step=5.0)
    src = dataclasses.replace(models.GFS, steps=(0,))

    def fake_step(source, run, step, max_gap):
        return grib.StepResult(step, {"t": msg}, 1, len(msg))

    df, _ = grib.fetch_run(src, RUN, stations, fetch_step=fake_step, log=None)
    path = tmp_path / src.file_name(RUN)
    grib.write_fcst(df, path)
    back = pd.read_parquet(path)
    assert path.name == "fcst-gfs-2026101200.parquet"
    assert list(back.columns) == grib.FCST_COLUMNS
    assert str(back["valid"].dtype) == "datetime64[ns, UTC]"
    assert back["t"].iloc[0] == pytest.approx(16.85, abs=0.01)
