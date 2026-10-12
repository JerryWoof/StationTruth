from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stationtruth import analysis as A
from stationtruth.grib import Grid

VALID = datetime(2026, 10, 12, 17, tzinfo=timezone.utc)


def flat_field(value: float = 0.0, d: float = 1.0) -> A.LatLon:
    nlat, nlon = int(round(180 / d)) + 1, int(round(360 / d))
    return A.LatLon(np.full((nlat, nlon), float(value)), -90.0, d)


def obs_frame(lat, lon, value, elev=None) -> pd.DataFrame:
    lat = np.asarray(lat, float)
    return pd.DataFrame(
        {
            "station": [f"S{i:03d}" for i in range(len(lat))],
            "lat": lat,
            "lon": np.asarray(lon, float),
            "elev_m": np.zeros(len(lat)) if elev is None else np.asarray(elev, float),
            "value": np.asarray(value, float),
        }
    )


def cluster(n=30, lat0=45.0, lon0=10.0, spread=3.0, seed=1):
    rng = np.random.default_rng(seed)
    return lat0 + rng.uniform(-spread, spread, n), lon0 + rng.uniform(-spread, spread, n)


def quiet(_msg):
    pass


# --------------------------------------------------------------------------- OI


def test_increments_recovered_near_stations_and_zero_far_away():
    bg = flat_field(10.0)
    lat, lon = cluster()
    res = A.analyse_var(A.VARS["t"], bg, None, obs_frame(lat, lon, np.full(len(lat), 12.0)), quiet)
    assert res.counts["used"] == len(lat)
    inc = res.increment
    # at the cluster centre the analysis is pulled almost all the way to the obs
    centre = A.at_points(inc, np.array([45.0]), np.array([10.0]))[0]
    assert 1.6 < centre <= 2.05
    # far away (beyond the cutoff) the analysis is exactly the background
    far = A.at_points(res.analysis, np.array([-40.0, 45.0]), np.array([-100.0, 40.0]))
    assert np.all(far == 10.0)
    assert np.count_nonzero(inc.data) < inc.data.size * 0.05
    # withheld-station validation: analysis beats the background
    v = res.validation
    assert v["analysis"]["mae"] < v["background"]["mae"] == pytest.approx(2.0)


def test_qc_rejects_outlier_and_orography_mismatch():
    bg = flat_field(10.0)
    orog = flat_field(0.0)
    lat, lon = cluster(40)
    rng = np.random.default_rng(3)
    val = 10.0 + rng.normal(0, 0.5, len(lat))
    val[5] = 30.0  # gross error
    elev = np.zeros(len(lat))
    elev[7] = 1500.0  # mountain station over flat model terrain
    res = A.analyse_var(A.VARS["t"], bg, orog, obs_frame(lat, lon, val, elev), quiet)
    status = res.stations["status"].to_numpy()
    assert status[5] == 1
    assert status[7] == 2
    assert (status == 0).sum() == len(lat) - 2
    assert res.counts["rejected_gross"] == 1 and res.counts["rejected_orography"] == 1
    # the outlier does not leak into the analysis
    assert abs(A.at_points(res.analysis, np.array([lat[5]]), np.array([lon[5]]))[0] - 10) < 1.0


def test_lapse_rate_moves_obs_to_model_height():
    bg = flat_field(10.0)
    orog = flat_field(300.0)
    lat, lon = cluster(20)
    # stations at sea level, model terrain 300 m: the obs is 1.95 K warmer than at 300 m
    res = A.analyse_var(A.VARS["t"], bg, orog, obs_frame(lat, lon, np.full(20, 10.0)), quiet)
    assert res.stations["adj"].to_numpy() == pytest.approx(-1.95)
    assert A.at_points(res.increment, np.array([45.0]), np.array([10.0]))[0] < -1.5


def test_withheld_residuals_match_brute_force():
    rng = np.random.default_rng(7)
    lat = 50 + rng.uniform(-2, 2, 25)
    lon = 5 + rng.uniform(-2, 2, 25)
    lat[1], lon[1] = lat[0] + 0.02, lon[0]  # a twin within 10 km
    d = rng.normal(0, 2, 25)
    cfg = A.VARS["t"].cfg
    alpha, ainv = A.solve_oi(lat, lon, d, cfg)
    fast = A.withheld_residuals(alpha, ainv.astype(float), lat, lon)
    x = A.unit_vectors(lat, lon)
    for k in range(25):
        keep = (x @ x[k]) < np.cos(A.CLUSTER_KM / A.EARTH_R_KM)
        a_k, _ = A.solve_oi(lat[keep], lon[keep], d[keep], cfg)
        r = A.gc_dist_km(lat[k], lon[k], lat[keep], lon[keep])
        incr = np.sum(cfg.sigma_b**2 * np.exp(-r / cfg.corr_length_km) * a_k)
        assert fast[k] == pytest.approx(d[k] - incr, abs=2e-3)


def test_increment_wraps_across_dateline():
    bg = flat_field(0.0, d=0.5)
    res = A.analyse_var(
        A.VARS["t"], bg, None, obs_frame([0.0, 0.5, -0.5], [179.9, 179.8, -179.9], [3.0] * 3), quiet
    )
    east = A.at_points(res.increment, np.array([0.0]), np.array([-179.5]))[0]
    west = A.at_points(res.increment, np.array([0.0]), np.array([179.5]))[0]
    assert east > 1.0 and west > 1.0


# --------------------------------------------------------------------------- grids + times


def test_regularize_handles_ecmwf_and_gfs_layouts():
    lats = np.arange(90, -90.1, -1.0)
    fn = lambda la, lo: la + 0.01 * ((lo + 180) % 360)  # noqa: E731
    lo_e = np.arange(-180, 180, 1.0)
    lo_g = np.arange(0, 360, 1.0)
    fe = fn(*np.meshgrid(lats, lo_e, indexing="ij"))
    fg = fn(*np.meshgrid(lats, lo_g, indexing="ij"))
    ge = Grid(360, 181, 90.0, -180.0, 1.0, 1.0, False, False)
    gg = Grid(360, 181, 90.0, 0.0, 1.0, 1.0, False, False)
    a, b = A.regularize(fe, ge), A.regularize(fg, gg)
    assert a.lat0 == -90 and b.lat0 == -90
    np.testing.assert_allclose(a.data, b.data)
    assert A.at_points(a, np.array([10.0]), np.array([-170.0]))[0] == pytest.approx(10.1)


def test_bracket_steps():
    assert A.bracket_steps(12.0) == ((12,), (1.0,))
    steps, w = A.bracket_steps(11.0)
    assert steps == (9, 12) and w == pytest.approx((1 / 3, 2 / 3))


def test_find_background_picks_newest_available_run():
    seen = []

    def exists(url):
        seen.append(url)
        return "/20261012/06z/" in url and "-oper-" in url

    run, stream, steps, w = A.find_background(VALID, exists=exists, log=quiet)
    assert run == datetime(2026, 10, 12, 6, tzinfo=timezone.utc)
    assert stream == "oper" and steps == (9, 12)
    assert any("/12z/" in u for u in seen)  # tried the newer run first


# --------------------------------------------------------------------------- rendering


def test_mercator_bounds():
    lon, lat = A.mercator_axes(2048)
    assert len(lon) == 2048 and len(lat) == 2041
    assert lon[0] == pytest.approx(-180 + 360 / 4096) and lon[-1] == pytest.approx(180 - 360 / 4096)
    assert 84.9 < lat[0] < 85.0 and -85.0 < lat[-1] < -84.9
    assert np.all(np.diff(lat) < 0)
    # rows equally spaced in Mercator y (what Leaflet's imageOverlay assumes)
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    np.testing.assert_allclose(np.diff(y), np.diff(y)[0])
    ymax = np.log(np.tan(np.pi / 4 + np.radians(85) / 2))
    assert y[0] + (y[0] - y[1]) / 2 == pytest.approx(ymax)
    # equator in the middle row
    assert abs(lat[len(lat) // 2]) < 0.2


def decode_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        chunks[tag] = chunks.get(tag, b"") + body
        pos += 12 + n
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, w + 1)
    out = np.zeros((h, w), np.uint8)
    for j in range(h):
        f, row = raw[j, 0], raw[j, 1:]
        out[j] = row if f == 0 else (row + (out[j - 1] if j else 0)).astype(np.uint8)
    return out, chunks, ctype


def test_png_roundtrip_and_palette():
    spec = A.VARS["t"]
    pal = A.value_palette(spec)
    vals = np.array([[-50.0, -40.0, 0.0], [20.25, 48.0, np.nan]])
    idx = pal.index(vals)
    data = A.png_indexed(idx, pal.rgba())
    out, chunks, ctype = decode_png(data)
    assert ctype == 3
    np.testing.assert_array_equal(out, idx)
    assert idx[1, 2] == 0  # NaN transparent
    assert idx[0, 0] == idx[0, 1] == 1  # clipped to the scale
    rgb = np.frombuffer(chunks[b"PLTE"], np.uint8).reshape(-1, 3)
    assert tuple(rgb[idx[0, 1]]) == (0x3D, 0x0D, 0x6B)  # -40 degC = first stop
    assert chunks[b"tRNS"][0] == 0


def test_increment_palette_transparent_below_threshold():
    spec = A.VARS["t"]
    ip = A.increment_palette(spec)
    rgba = ip.rgba()
    idx = ip.index(np.array([0.0, 0.1, -0.1, 0.5, -3.0]))
    alpha = rgba[idx, 3]
    assert list(alpha[:3]) == [0, 0, 0]
    assert alpha[3] > 0 and alpha[4] == 255
    assert rgba[idx[4], 2] > rgba[idx[4], 0]  # negative = blue


# --------------------------------------------------------------------------- end to end


def test_run_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    lat, lon = cluster(25)
    st = pd.DataFrame(
        {
            "station": [f"S{i:03d}" for i in range(25)],
            "name": "x",
            "country": "",
            "lat": lat,
            "lon": lon,
            "elev_m": np.zeros(25, np.float32),
        }
    )
    st.to_parquet(data / "stations.parquet")
    times = [VALID - timedelta(minutes=10), VALID + timedelta(minutes=40)]  # 2nd outside window
    rows = []
    for i in range(25):
        for k, t in enumerate(times):
            rows.append(
                {
                    "station": f"S{i:03d}",
                    "time": pd.Timestamp(t),
                    "type": "METAR",
                    "t": 13.0 if k == 0 else 99.0,
                    "wspd": 8.0,
                }
            )
    pd.DataFrame(rows).to_parquet(data / f"obs-{VALID:%Y-%m-%d}.parquet")
    f = flat_field(10.0)
    bg = A.Background(
        VALID - timedelta(hours=5),
        (3, 6),
        (1 / 3, 2 / 3),
        {"t": f, "wspd": flat_field(5.0), "orog": flat_field(0.0)},
        123,
        "oper",
    )
    out = tmp_path / "out"
    doc = A.run(data, out, valid=VALID, width=256, background=bg, log=quiet)
    for v in ("t", "wspd"):
        for kind in ("analysis", "background", "increment"):
            assert (out / f"{v}_{kind}.png").exists()
    saved = json.loads((out / "analysis.json").read_text())
    assert saved["valid"] == "2026-10-12T17:00:00Z"
    assert saved["bounds"] == [[-85.0, -180.0], [85.0, 180.0]]
    t = saved["variables"]["t"]
    assert t["counts"]["used"] == 25
    assert t["validation"]["background"]["mae"] == pytest.approx(3.0)
    assert t["validation"]["analysis"]["mae"] < 1.0
    g = t["grid"]
    assert len(g["analysis"]) == g["nlat"] * g["nlon"] == 171 * 360
    assert doc["background"]["lead_h"] == 5.0
    img, _, _ = decode_png((out / "t_analysis.png").read_bytes())
    assert img.shape == (
        int(round(256 * np.log(np.tan(np.pi / 4 + np.radians(42.5))) / np.pi)),
        256,
    )
