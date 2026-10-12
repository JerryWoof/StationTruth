"""METAR observations: current cache, Data API history, decoding and merging."""

from __future__ import annotations

import csv
import gzip
import io
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from metar_parse import Metar, parse_metar

from . import http

METARS_URL = "https://aviationweather.gov/data/cache/metars.cache.csv.gz"
DATA_API_URL = "https://aviationweather.gov/api/data/metar"
# Data API limits (probed October 2026): at most 400 station ids per request,
# at most 400 reports per response (newest first), roughly 15-30 days of history.
API_MAX_IDS = 400
API_MAX_RESULTS = 400

COLUMNS = [
    "station",
    "time",
    "type",
    "t",
    "td",
    "wdir",
    "wspd",
    "gust",
    "vis_m",
    "ceil_ft",
    "qnh",
    "slp",
    "cat",
]
FLOAT_COLUMNS = ["t", "td", "wdir", "wspd", "gust", "vis_m", "ceil_ft", "qnh", "slp"]

NAN = float("nan")


def obs_file_name(day: date) -> str:
    return f"obs-{day:%Y-%m-%d}.parquet"


def _num(value) -> float:
    return NAN if value is None else float(value)


def metar_to_row(metar: Metar, reference: datetime, report_type: str | None = None) -> dict | None:
    """Turn a decoded report into one contract row; ``None`` if unusable."""
    if metar.nil or not metar.station or metar.time is None:
        return None
    when = metar.time.to_datetime(reference)
    if when is None:
        return None
    rmk = metar.remarks

    t = metar.temperature_c
    td = metar.dewpoint_c
    if rmk is not None and rmk.precise_temperature_c is not None:
        t = rmk.precise_temperature_c
    if rmk is not None and rmk.precise_dewpoint_c is not None:
        td = rmk.precise_dewpoint_c

    wdir = wspd = gust = NAN
    wind = metar.wind
    if wind is not None:
        wspd = _num(wind.speed_kt)
        gust = _num(wind.gust_kt)
        if not wind.variable and not wind.calm and wind.direction_deg is not None:
            wdir = float(wind.direction_deg)

    vis = NAN
    if metar.cavok:
        vis = 10000.0
    elif metar.visibility is not None and metar.visibility.value is not None:
        v = metar.visibility
        if v.unit == "M" and v.value >= 9999:
            vis = 10000.0
        else:
            vis = _num(v.meters)

    ceil_ft = NAN if metar.cavok else _num(metar.ceiling_ft)
    qnh = NAN if metar.pressure is None else float(metar.pressure.hpa)
    slp = NAN if rmk is None else _num(rmk.sea_level_pressure_hpa)
    rtype = (report_type or metar.report_type or "METAR").upper()
    if rtype not in ("METAR", "SPECI"):
        rtype = metar.report_type or "METAR"

    return {
        "station": metar.station,
        "time": when,
        "type": rtype,
        "t": _num(t),
        "td": _num(td),
        "wdir": wdir,
        "wspd": wspd,
        "gust": gust,
        "vis_m": vis,
        "ceil_ft": ceil_ft,
        "qnh": qnh,
        "slp": slp,
        "cat": metar.flight_category or "",
    }


def decode_reports(
    reports: Iterable[tuple[str, datetime, str | None]],
    now: datetime | None = None,
) -> pd.DataFrame:
    """Decode ``(raw_text, reference_time, report_type)`` tuples into an obs frame.

    Reports dated more than an hour after ``now`` are dropped as garbage.
    """
    now = now or datetime.now(timezone.utc)
    limit = now + timedelta(hours=1)
    rows = []
    for raw, reference, rtype in reports:
        if not raw or not raw.strip():
            continue
        try:
            metar = parse_metar(raw)
            row = metar_to_row(metar, reference, rtype)
        except Exception:  # never let one odd report kill a collection run
            continue
        if row is not None and row["time"] <= limit:
            rows.append(row)
    return to_frame(rows)


def empty_frame() -> pd.DataFrame:
    return to_frame([])


def to_frame(rows) -> pd.DataFrame:
    """Rows (dicts or a DataFrame) -> contract dtypes, deduplicated and sorted."""
    df = pd.DataFrame(rows, columns=COLUMNS) if not isinstance(rows, pd.DataFrame) else rows
    df = df.reindex(columns=COLUMNS)
    df["station"] = df["station"].astype(str)
    df["time"] = pd.to_datetime(df["time"], utc=True).astype("datetime64[ns, UTC]")
    df["type"] = df["type"].fillna("METAR").astype(str)
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["cat"] = df["cat"].fillna("").astype(str)
    df = df.drop_duplicates(["station", "time"], keep="last")
    return df.sort_values(["station", "time"]).reset_index(drop=True)


def merge_obs(existing: pd.DataFrame | None, new: pd.DataFrame | None) -> pd.DataFrame:
    """Union of two obs frames, deduplicated on (station, time); ``new`` wins."""
    parts = [to_frame(x) for x in (existing, new) if x is not None and len(x)]
    if not parts:
        return empty_frame()
    return to_frame(pd.concat(parts, ignore_index=True))


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split an obs frame by UTC date of ``time``."""
    if df.empty:
        return {}
    days = df["time"].dt.date
    return {d: part.reset_index(drop=True) for d, part in df.groupby(days)}


def write_obs(df: pd.DataFrame, path: Path) -> Path:
    to_frame(df).to_parquet(path, index=False, compression="zstd")
    return path


def read_obs(path: Path) -> pd.DataFrame:
    return to_frame(pd.read_parquet(path))


# --------------------------------------------------------------------------- current cache


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_metar_cache_csv(text: str, now: datetime | None = None) -> pd.DataFrame:
    """Decode the aviationweather.gov ``metars.cache.csv`` text."""
    now = now or datetime.now(timezone.utc)
    reader = csv.reader(io.StringIO(text))
    header = None
    for row in reader:
        if row and row[0] == "raw_text":
            header = row
            break
    if header is None:
        return empty_frame()
    i_raw = header.index("raw_text")
    i_time = header.index("observation_time") if "observation_time" in header else None
    i_type = header.index("metar_type") if "metar_type" in header else None

    def reports():
        for row in reader:
            if len(row) <= i_raw:
                continue
            ref = _parse_iso(row[i_time]) if i_time is not None and len(row) > i_time else None
            rtype = row[i_type] if i_type is not None and len(row) > i_type else None
            yield row[i_raw], ref or now, rtype or None

    return decode_reports(reports(), now=now)


def fetch_latest(url: str = METARS_URL, now: datetime | None = None) -> pd.DataFrame:
    """Download and decode the current METAR cache (latest reports, worldwide)."""
    raw = http.get(url, timeout=120)
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return parse_metar_cache_csv(raw.decode("utf-8", errors="replace"), now=now)


# --------------------------------------------------------------------------- history (Data API)


def _report_type(line: str) -> str | None:
    head = line.split(None, 1)[0] if line.strip() else ""
    return head if head in ("METAR", "SPECI") else None


def fetch_history_chunk(
    ids: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    pause: float = 1.0,
    getter: Callable[[str], bytes] | None = None,
    stats: dict | None = None,
) -> pd.DataFrame:
    """Reports for up to 400 stations between ``start`` and ``end``.

    The API returns at most 400 reports per call, newest first, so the window
    is walked backwards: whenever a response is full, the next request ends at
    the oldest time received (that minute is re-requested; duplicates merge).
    """
    if len(ids) > API_MAX_IDS:
        raise ValueError(f"at most {API_MAX_IDS} ids per request")
    getter = getter or (lambda u: http.get(u, timeout=45, retries=3))
    frames = []
    win_end = end
    first = True
    while win_end > start:
        if not first and pause:
            time.sleep(pause)
        first = False
        hours = max(1, math.ceil((win_end - start).total_seconds() / 3600))
        url = (
            f"{DATA_API_URL}?ids={','.join(ids)}&format=raw&hours={hours}"
            f"&date={win_end:%Y%m%d_%H%M}"
        )
        body = getter(url)
        if stats is not None:
            stats["requests"] = stats.get("requests", 0) + 1
            stats["bytes"] = stats.get("bytes", 0) + len(body)
        lines = [ln.strip() for ln in body.decode("utf-8", "replace").splitlines() if ln.strip()]
        if lines and lines[0].startswith("Maximum number"):
            raise RuntimeError(lines[0])
        df = decode_reports(((ln, win_end, _report_type(ln)) for ln in lines), now=end)
        frames.append(df)
        if len(lines) < API_MAX_RESULTS or df.empty:
            break
        oldest = df["time"].min().to_pydatetime()
        if oldest >= win_end:  # a single minute holds >= 400 reports; step past it
            oldest = win_end - timedelta(minutes=1)
        win_end = oldest
    out = merge_obs(None, pd.concat(frames, ignore_index=True) if frames else None)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return out[(out["time"] >= lo) & (out["time"] <= hi)].reset_index(drop=True)


def fetch_history(
    station_ids: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    chunk: int = 200,
    pause: float = 1.0,
    workers: int = 1,
    log: Callable[[str], None] | None = print,
    getter: Callable[[str], bytes] | None = None,
    stats: dict | None = None,
) -> pd.DataFrame:
    """Reports for all ``station_ids`` in ``[start, end]`` from the Data API.

    Station ids go out in chunks (at most 400 per request); with ``workers``
    > 1 a few chunks are walked concurrently, each pausing ``pause`` seconds
    between its own requests. A failing chunk is logged and skipped.
    """
    ids = sorted({s for s in station_ids if s})
    chunk = max(1, min(chunk, API_MAX_IDS))
    stats = stats if stats is not None else {}
    lock = threading.Lock()
    t0 = time.monotonic()
    parts = [ids[k : k + chunk] for k in range(0, len(ids), chunk)]
    n_chunks = len(parts)
    done = [0]

    def run(k: int) -> pd.DataFrame | None:
        local: dict = {}
        try:
            df = fetch_history_chunk(parts[k], start, end, pause=pause, getter=getter, stats=local)
        except Exception as exc:  # keep going; a later backfill can fill the gap
            df = None
            err = exc
        with lock:
            for key, val in local.items():
                stats[key] = stats.get(key, 0) + val
            done[0] += 1
            if log:
                if df is None:
                    log(f"  history chunk {k + 1}/{n_chunks} failed: {err}")
                else:
                    log(
                        f"  history chunk {k + 1}/{n_chunks} ({done[0]} done): {len(df)} reports, "
                        f"{stats.get('requests', 0)} requests, "
                        f"{stats.get('bytes', 0) / 1e6:.1f} MB, {time.monotonic() - t0:.0f}s"
                    )
        if pause:
            time.sleep(pause)
        return df

    if workers > 1 and n_chunks > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, range(n_chunks)))
    else:
        results = [run(k) for k in range(n_chunks)]
    frames = [df for df in results if df is not None and len(df)]
    if not frames:
        return empty_frame()
    return merge_obs(None, pd.concat(frames, ignore_index=True))


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "0 reports"
    return (
        f"{len(df)} reports, {df['station'].nunique()} stations, "
        f"{df['time'].min():%Y-%m-%d %H:%M} .. {df['time'].max():%Y-%m-%d %H:%M}"
    )


__all__ = [
    "COLUMNS",
    "decode_reports",
    "fetch_history",
    "fetch_latest",
    "merge_obs",
    "metar_to_row",
    "obs_file_name",
    "parse_metar_cache_csv",
    "read_obs",
    "split_by_day",
    "write_obs",
]
