"""Station list from the aviationweather.gov station cache (see docs/DATA.md)."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import http

STATIONS_URL = "https://aviationweather.gov/data/cache/stations.cache.json.gz"
STATIONS_FILE = "stations.parquet"
COLUMNS = ["station", "name", "country", "lat", "lon", "elev_m"]


def _valid_coord(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -360 <= lon <= 360


def normalize_lon(lon):
    """Wrap longitudes into [-180, 180)."""
    return (np.asarray(lon, dtype="float64") + 180.0) % 360.0 - 180.0


def parse_stations(records: Iterable[dict]) -> pd.DataFrame:
    """Build the contract ``stations`` frame from station-cache records."""
    rows = []
    for rec in records:
        icao = (rec.get("icaoId") or "").strip().upper()
        site_type = rec.get("siteType") or []
        if isinstance(site_type, str):
            site_type = [site_type]
        if not icao or "METAR" not in site_type:
            continue
        if not _valid_coord(rec.get("lat"), rec.get("lon")):
            continue
        elev = rec.get("elev")
        try:
            elev = float(elev)
        except (TypeError, ValueError):
            elev = float("nan")
        rows.append(
            {
                "station": icao,
                "name": (rec.get("site") or "").strip(),
                "country": (rec.get("country") or "").strip().upper(),
                "lat": float(rec["lat"]),
                "lon": float(rec["lon"]),
                "elev_m": elev,
            }
        )
    return to_frame(rows)


def to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["station"] = df["station"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = normalize_lon(df["lon"].to_numpy()) if len(df) else df["lon"].astype("float64")
    df["elev_m"] = df["elev_m"].astype("float32")
    df = df.drop_duplicates("station", keep="first").sort_values("station")
    return df.reset_index(drop=True)


def fetch_stations(url: str = STATIONS_URL) -> pd.DataFrame:
    """Download the station cache and return METAR-reporting stations."""
    raw = http.get(url, timeout=120)
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass  # server already decompressed it
    return parse_stations(json.loads(raw))


def write_stations(df: pd.DataFrame, path: Path) -> Path:
    df[COLUMNS].to_parquet(path, index=False, compression="zstd")
    return path


def read_stations(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
