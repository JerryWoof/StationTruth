"""TAFs: collect official airport forecasts and expand them into an hourly table.

Sources
-------
* ``collect``: the aviationweather.gov TAF cache (``tafs.cache.xml.gz``), which
  holds the TAFs currently in force worldwide. Run hourly; every TAF is
  decoded and stored once per ``(station, issued)``.
* ``backfill``: the aviationweather.gov Data API. ``/api/data/taf?ids=..&date=T``
  returns, per station, the latest TAF issued at or before ``T`` (probed
  October 2026: about a month of history; ``hours`` is rejected). Walking ``T``
  hourly therefore recovers every TAF except amendments superseded within the
  step.

Hourly expansion (``expand_taf``)
---------------------------------
A TAF hour ``H`` stands for the clock hour ``[H, H+1h)``. Only hours that start
at or after the issue time and inside the validity are kept, so a TAF never
"forecasts" an hour that had already begun. ``lead_h`` is ``floor(H - issued)``
in hours.

*Prevailing* conditions at ``H`` are the state in effect at the start of the
hour, built from the base forecast and the ``FM`` and ``BECMG`` groups whose
effective time is ``<= H``, applied in order of effective time (ties keep the
order of the text):

* ``FM`` replaces the whole forecast from its time;
* ``BECMG`` changes only the elements it mentions and **takes effect at the end
  of its period** (``BECMG 1214/1216`` applies from 16Z). ICAO allows the
  change at any time inside the period; the end is the time from which the
  forecaster says the new conditions hold, so the TAF is not credited (or
  blamed) for the transition window;
* ``CAVOK`` sets visibility 10 km, no cloud and no weather; ``NSW`` clears weather.

``TEMPO``, ``PROB30/40`` and ``INTER`` groups (including ``PROB40 TEMPO``) are
*temporary*: they do not change the prevailing state. A group is active in hour
``H`` when its period overlaps ``[H, H+1h)``. Each active group is overlaid on
the prevailing state (elements it does not mention are kept) and the worst
resulting flight category is ``tempo_cat``; ``tempo_prob`` is the largest
probability among active groups (100 for a plain ``TEMPO``/``INTER``).

Flight categories use MetarParse's ``derived.flight_category`` (FAA limits:
ceiling < 500 ft or vis < 1 SM LIFR, < 1000 ft / 3 SM IFR, <= 3000 ft / 5 SM
MVFR), exactly like the observed ``cat`` column of the obs files.

``taf-YYYY-MM-DD.parquet`` (UTC day of ``issued``)
--------------------------------------------------
=============  ==========  ============================================================
column         type        notes
=============  ==========  ============================================================
station        str         ICAO id
issued         datetime    TAF issue time (UTC)
hour           datetime    start of the forecast hour (UTC)
lead_h         int16       whole hours from ``issued`` to ``hour``
prev_cat       str         VFR/MVFR/IFR/LIFR of the prevailing conditions, or empty
prev_vis_m     float32     prevailing visibility, m (CAVOK/9999 -> 10000)
prev_ceil_ft   float32     prevailing ceiling, ft; NaN when none
prev_wspd      float32     prevailing wind speed, kt
prev_gust      float32     prevailing gust, kt; NaN when none
tempo_cat      str         worst category among active TEMPO/PROB/INTER groups, or empty
tempo_prob     int8        largest probability of those groups (100 = TEMPO), 0 if none
amended        bool        ``TAF AMD``
raw            str         the TAF text (repeated per hour; dictionary-encoded)
=============  ==========  ============================================================

Rows are unique on ``(station, issued, hour)``; a re-collected TAF (same
station and issue time) replaces the stored one.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from metar_parse import Metar, parse_taf
from metar_parse.models import DayTime, Taf, TafChangeGroup

from . import http
from .storage import Storage, open_storage

TAFS_URL = "https://aviationweather.gov/data/cache/tafs.cache.xml.gz"
DATA_API_URL = "https://aviationweather.gov/api/data/taf"
API_MAX_IDS = 400
MAX_VALIDITY_H = 36

CATEGORIES = ("LIFR", "IFR", "MVFR", "VFR")  # worst first
_CAT_RANK = {c: i for i, c in enumerate(CATEGORIES)}

COLUMNS = [
    "station",
    "issued",
    "hour",
    "lead_h",
    "prev_cat",
    "prev_vis_m",
    "prev_ceil_ft",
    "prev_wspd",
    "prev_gust",
    "tempo_cat",
    "tempo_prob",
    "amended",
    "raw",
]
FLOAT_COLUMNS = ["prev_vis_m", "prev_ceil_ft", "prev_wspd", "prev_gust"]
TEMPORARY_KINDS = ("TEMPO", "PROB", "INTER")

NAN = float("nan")
HOUR = timedelta(hours=1)

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def taf_file_name(day: date) -> str:
    return f"taf-{day:%Y-%m-%d}.parquet"


def worst(cats: Iterable[str]) -> str:
    """Worst flight category of ``cats`` (empty strings ignored); '' if none."""
    known = [c for c in cats if c in _CAT_RANK]
    return min(known, key=_CAT_RANK.__getitem__) if known else ""


# --------------------------------------------------------------------------- state


class _State:
    """Forecast conditions: the elements a TAF group can set."""

    __slots__ = ("wind", "visibility", "weather", "clouds", "cavok")

    def __init__(self, wind=None, visibility=None, weather=(), clouds=(), cavok=False):
        self.wind = wind
        self.visibility = visibility
        self.weather = list(weather)
        self.clouds = list(clouds)
        self.cavok = cavok

    @classmethod
    def from_group(cls, g: TafChangeGroup) -> _State:
        s = cls()
        s.apply(g)
        return s

    def copy(self) -> _State:
        return _State(self.wind, self.visibility, self.weather, self.clouds, self.cavok)

    def apply(self, g: TafChangeGroup) -> None:
        """Change the elements ``g`` mentions (BECMG / TEMPO semantics)."""
        if g.wind is not None:
            self.wind = g.wind
        if g.cavok:
            self.cavok = True
            self.visibility = None
            self.clouds = []
            self.weather = []
        else:
            if g.visibility is not None and g.visibility.value is not None:
                self.visibility = g.visibility
                self.cavok = False
            if g.clouds:
                self.clouds = list(g.clouds)
                self.cavok = False
            if g.weather:
                self.weather = list(g.weather)
        if g.nsw:
            self.weather = []

    def category(self) -> str:
        m = Metar(raw="", visibility=self.visibility, clouds=self.clouds, cavok=self.cavok)
        return m.flight_category or ""

    def vis_m(self) -> float:
        if self.cavok:
            return 10000.0
        v = self.visibility
        if v is None or v.value is None:
            return NAN
        if v.unit == "M" and v.value >= 9999:
            return 10000.0
        return NAN if v.meters is None else float(v.meters)

    def ceil_ft(self) -> float:
        if self.cavok:
            return NAN
        heights = [c.height_ft for c in self.clouds if c.is_ceiling]
        return float(min(heights)) if heights else NAN

    def wind_kt(self) -> tuple[float, float]:
        if self.wind is None:
            return NAN, NAN
        s, g = self.wind.speed_kt, self.wind.gust_kt
        return (NAN if s is None else float(s)), (NAN if g is None else float(g))


# --------------------------------------------------------------------------- expansion


def _resolve(dt: DayTime | None, issued: datetime, start: datetime | None) -> datetime | None:
    """DayTime -> datetime. Old-style times without a day are the first match >= start."""
    if dt is None:
        return None
    if dt.day is None and start is not None:
        when = dt.to_datetime(start)
        if when is not None and when < start:
            when += timedelta(days=1)
        return when
    return dt.to_datetime(issued)


def _floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def _ceil_hour(t: datetime) -> datetime:
    f = _floor_hour(t)
    return f if f == t else f + HOUR


def expand_taf(taf: Taf, reference: datetime) -> list[dict]:
    """Hourly rows (see module docstring) for one decoded TAF; [] if unusable.

    ``reference`` is any time near the issue time (the cache's ``issue_time``
    or the query time) used to resolve day-of-month values.
    """
    if taf.nil or taf.cancelled or not taf.station or taf.issued is None:
        return []
    if not taf.groups or taf.valid_from is None or taf.valid_to is None:
        return []
    issued = taf.issued.to_datetime(reference)
    if issued is None:
        return []
    v_from = _resolve(taf.valid_from, issued, None)
    v_to = _resolve(taf.valid_to, issued, v_from)
    if v_from is None or v_to is None or v_to <= v_from:
        return []
    if v_to - v_from > timedelta(hours=MAX_VALIDITY_H) or v_from - issued > timedelta(hours=30):
        return []

    base = taf.groups[0]
    events: list[tuple[datetime, int, TafChangeGroup]] = []  # (effective time, order, group)
    temporary: list[tuple[datetime, datetime, int, TafChangeGroup]] = []
    for k, g in enumerate(taf.groups[1:], start=1):
        g_from = _resolve(g.valid_from, issued, v_from)
        if g.kind == "FM":
            if g_from is not None:
                events.append((g_from, k, g))
        elif g.kind == "BECMG":
            g_to = _resolve(g.valid_to, issued, g_from or v_from)
            if g_to is not None:
                events.append((g_to, k, g))
        elif g.kind in TEMPORARY_KINDS:
            g_to = _resolve(g.valid_to, issued, g_from or v_from)
            if g_from is not None and g_to is not None and g_to > g_from:
                prob = g.probability if g.probability else 100
                temporary.append((g_from, g_to, prob, g))
    events.sort(key=lambda e: (e[0], e[1]))

    first = max(_ceil_hour(issued), _ceil_hour(v_from))
    rows = []
    state = _State.from_group(base)
    ev = 0
    h = first
    raw = " ".join(taf.raw.split())
    while h < v_to:
        while ev < len(events) and events[ev][0] <= h:
            g = events[ev][2]
            if g.kind == "FM":
                state = _State.from_group(g)
            else:
                state.apply(g)
            ev += 1
        prev_cat = state.category()
        tempo_cats, tempo_prob = [], 0
        h_end = h + HOUR
        for t_from, t_to, prob, g in temporary:
            if t_from < h_end and t_to > h:
                s = state.copy()
                s.apply(g)
                tempo_cats.append(s.category())
                tempo_prob = max(tempo_prob, prob)
        wspd, gust = state.wind_kt()
        rows.append(
            {
                "station": taf.station,
                "issued": issued,
                "hour": h,
                "lead_h": int((h - issued).total_seconds() // 3600),
                "prev_cat": prev_cat,
                "prev_vis_m": state.vis_m(),
                "prev_ceil_ft": state.ceil_ft(),
                "prev_wspd": wspd,
                "prev_gust": gust,
                "tempo_cat": worst(tempo_cats),
                "tempo_prob": tempo_prob,
                "amended": bool(taf.amended),
                "raw": raw,
            }
        )
        h = h_end
    return rows


def expand_reports(reports: Iterable[tuple[str, datetime]]) -> pd.DataFrame:
    """Decode and expand ``(raw_text, reference_time)`` pairs; bad TAFs are skipped."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw, ref in reports:
        if not raw or not raw.strip():
            continue
        key = (" ".join(raw.split()), ref.strftime("%Y%m%d"))
        if key in seen:
            continue
        seen.add(key)
        try:
            rows.extend(expand_taf(parse_taf(raw), ref))
        except Exception:  # never let one odd report kill a run
            continue
    return to_frame(rows)


# --------------------------------------------------------------------------- frames


def to_frame(rows) -> pd.DataFrame:
    """Rows (dicts or DataFrame) -> contract dtypes, deduplicated and sorted."""
    df = pd.DataFrame(rows, columns=COLUMNS) if not isinstance(rows, pd.DataFrame) else rows
    df = df.reindex(columns=COLUMNS)
    df["station"] = df["station"].astype(str)
    for col in ("issued", "hour"):
        df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[ns, UTC]")
    df["lead_h"] = pd.to_numeric(df["lead_h"], errors="coerce").fillna(0).astype("int16")
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in ("prev_cat", "tempo_cat", "raw"):
        df[col] = df[col].fillna("").astype(str)
    df["tempo_prob"] = pd.to_numeric(df["tempo_prob"], errors="coerce").fillna(0).astype("int8")
    df["amended"] = df["amended"].fillna(False).astype(bool)
    df = df.drop_duplicates(["station", "issued", "hour"], keep="last")
    return df.sort_values(["station", "issued", "hour"]).reset_index(drop=True)


def empty_frame() -> pd.DataFrame:
    return to_frame([])


def merge(existing: pd.DataFrame | None, new: pd.DataFrame | None) -> pd.DataFrame:
    """Union of two TAF frames; a TAF (station, issued) in ``new`` replaces the stored one."""
    if existing is None or not len(existing):
        return to_frame(new) if new is not None and len(new) else empty_frame()
    if new is None or not len(new):
        return to_frame(existing)
    existing, new = to_frame(existing), to_frame(new)
    keys = pd.MultiIndex.from_frame(new[["station", "issued"]].drop_duplicates())
    old_keys = pd.MultiIndex.from_frame(existing[["station", "issued"]])
    keep = existing[~old_keys.isin(keys)]
    return to_frame(pd.concat([keep, new], ignore_index=True))


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split by UTC date of ``issued``."""
    if df.empty:
        return {}
    days = df["issued"].dt.date
    return {d: part.reset_index(drop=True) for d, part in df.groupby(days)}


def write_tafs(df: pd.DataFrame, path: Path) -> Path:
    to_frame(df).to_parquet(path, index=False, compression="zstd")
    return path


def read_tafs(path: Path) -> pd.DataFrame:
    return to_frame(pd.read_parquet(path))


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "0 TAFs"
    n = len(df[["station", "issued"]].drop_duplicates())
    return (
        f"{n} TAFs, {df['station'].nunique()} stations, {len(df)} hours, issued "
        f"{df['issued'].min():%Y-%m-%d %H:%M} .. {df['issued'].max():%Y-%m-%d %H:%M}"
    )


# --------------------------------------------------------------------------- cache


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_cache_xml(data: bytes, now: datetime | None = None) -> list[tuple[str, datetime]]:
    """``(raw_text, issue_time)`` for each ``<TAF>`` of the cache XML."""
    now = now or datetime.now(timezone.utc)
    out = []
    for _, el in ET.iterparse(io.BytesIO(data), events=("end",)):
        if el.tag != "TAF":
            continue
        raw = (el.findtext("raw_text") or "").strip()
        ref = _parse_iso(el.findtext("issue_time")) or now
        if raw:
            if not raw.startswith("TAF"):
                raw = "TAF " + raw
            out.append((raw, ref))
        el.clear()
    return out


def fetch_cache(url: str = TAFS_URL, now: datetime | None = None) -> pd.DataFrame:
    """Download, decode and expand the current TAF cache."""
    raw = http.get(url, timeout=120)
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return expand_reports(parse_cache_xml(raw, now=now))


def cache_stations(url: str = TAFS_URL) -> list[str]:
    """Station ids that currently issue TAFs (from the cache)."""
    raw = http.get(url, timeout=120)
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    ids = set()
    for _, el in ET.iterparse(io.BytesIO(raw), events=("end",)):
        if el.tag == "station_id" and el.text:
            ids.add(el.text.strip())
    return sorted(ids)


# --------------------------------------------------------------------------- Data API history


def split_raw_tafs(text: str) -> list[str]:
    """Split the Data API ``format=raw`` body into one string per TAF.

    Each TAF starts with a line beginning ``TAF``; change groups are on
    indented continuation lines (``PROB40`` and its ``TEMPO`` may be on
    separate lines, which joining with spaces restores).
    """
    out: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if line.startswith("TAF") or not out:
            out.append([s])
        else:
            out[-1].append(s)
    return [" ".join(p) for p in out if p and p[0].startswith("TAF")]


def fetch_at(
    ids: Sequence[str],
    when: datetime,
    *,
    getter: Callable[[str], bytes] | None = None,
    stats: dict | None = None,
) -> list[tuple[str, datetime]]:
    """TAFs in force at ``when`` (latest issued at or before it) for up to 400 stations."""
    if len(ids) > API_MAX_IDS:
        raise ValueError(f"at most {API_MAX_IDS} ids per request")
    getter = getter or (lambda u: http.get(u, timeout=60, retries=3))
    url = f"{DATA_API_URL}?ids={','.join(ids)}&format=raw&date={when:%Y%m%d_%H%M}"
    body = getter(url)
    if stats is not None:
        stats["requests"] = stats.get("requests", 0) + 1
        stats["bytes"] = stats.get("bytes", 0) + len(body)
    text = body.decode("utf-8", "replace")
    if text.lstrip().startswith(("Maximum number", "Unexpected", "Invalid")):
        raise RuntimeError(text.strip()[:200])
    return [(raw, when) for raw in split_raw_tafs(text)]


def fetch_history(
    station_ids: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    step_h: float = 1.0,
    chunk: int = API_MAX_IDS,
    pause: float = 0.5,
    workers: int = 2,
    log: Callable[[str], None] | None = log,
    getter: Callable[[str], bytes] | None = None,
    stats: dict | None = None,
) -> pd.DataFrame:
    """Every TAF issued in ``[start, end]`` that the Data API can still return.

    Queries ``date=T`` for ``T`` from ``start`` to ``end`` every ``step_h``
    hours, for each chunk of at most 400 station ids. The result is limited
    to TAFs issued in ``[start - 1 day, end]``, expanded hourly.
    """
    ids = sorted({s for s in station_ids if s})
    chunk = max(1, min(chunk, API_MAX_IDS))
    parts = [ids[k : k + chunk] for k in range(0, len(ids), chunk)]
    times = []
    t = end
    while t >= start:
        times.append(t)
        t -= timedelta(hours=step_h)
    jobs = [(t, p) for t in times for p in parts]
    stats = stats if stats is not None else {}
    lock = threading.Lock()
    reports: dict[str, datetime] = {}
    done = [0]
    t0 = time.monotonic()

    def run(job):
        when, part = job
        local: dict = {}
        try:
            got = fetch_at(part, when, getter=getter, stats=local)
            err = None
        except Exception as exc:  # keep going; neighbouring steps overlap
            got, err = [], exc
        with lock:
            for k, v in local.items():
                stats[k] = stats.get(k, 0) + v
            for raw, ref in got:
                key = " ".join(raw.split())
                reports.setdefault(key, ref)
            done[0] += 1
            if err is not None:
                stats["failed"] = stats.get("failed", 0) + 1
                if log:
                    log(f"  {when:%Y-%m-%d %H:%M}: request failed: {err}")
            elif log and (done[0] % 50 == 0 or done[0] == len(jobs)):
                log(
                    f"  {done[0]}/{len(jobs)} requests, {len(reports)} distinct TAFs, "
                    f"{stats.get('bytes', 0) / 1e6:.1f} MB, {time.monotonic() - t0:.0f}s"
                )
        if pause:
            time.sleep(pause)

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, jobs))
    else:
        for job in jobs:
            run(job)
    df = expand_reports(reports.items())
    lo, hi = pd.Timestamp(start - timedelta(days=1)), pd.Timestamp(end)
    return df[(df["issued"] >= lo) & (df["issued"] <= hi)].reset_index(drop=True)


# --------------------------------------------------------------------------- storage


def store(
    storage: Storage,
    new: pd.DataFrame,
    work: Path,
    days: set[date] | None = None,
    log: Callable[[str], None] | None = log,
) -> int:
    """Merge ``new`` into the stored daily TAF files; returns the number of new TAFs."""
    added = 0
    for day, part in sorted(split_by_day(new).items()):
        if days is not None and day not in days:
            if log:
                log(f"taf: skipping {part['issued'].nunique()} issue times dated {day}")
            continue
        name = taf_file_name(day)
        existing_path = storage.get(name, work)
        existing = read_tafs(existing_path) if existing_path else None

        def n_tafs(df):
            return 0 if df is None else len(df[["station", "issued"]].drop_duplicates())

        before = n_tafs(existing)
        merged = merge(existing, part)
        path = write_tafs(merged, work / name)
        storage.put(path)
        after = n_tafs(merged)
        added += after - before
        if log:
            log(
                f"taf: {name}: {before} -> {after} TAFs (+{after - before}), "
                f"{len(merged)} hours, {path.stat().st_size / 1e6:.2f} MB"
            )
    return added


def collect(
    storage: Storage, now: datetime | None = None, log: Callable[[str], None] | None = log
) -> int:
    """Fetch the TAF cache and merge it into today's/yesterday's files."""
    now = now or datetime.now(timezone.utc)
    t = time.monotonic()
    df = fetch_cache(now=now)
    if log:
        log(f"taf: cache: {summarize(df)} ({time.monotonic() - t:.1f}s)")
    today = now.date()
    days = {today, today - timedelta(days=1), today - timedelta(days=2)}
    with tempfile.TemporaryDirectory() as tmp:
        return store(storage, df, Path(tmp), days=days, log=log)


def backfill(
    storage: Storage,
    days: float,
    *,
    now: datetime | None = None,
    step_h: float = 1.0,
    pause: float = 0.5,
    workers: int = 2,
    station_ids: Sequence[str] | None = None,
    log: Callable[[str], None] | None = log,
) -> int:
    now = now or datetime.now(timezone.utc)
    end = _floor_hour(now)
    start = end - timedelta(days=days)
    ids = list(station_ids) if station_ids else cache_stations()
    if log:
        log(
            f"taf: backfilling {len(ids)} stations, {start:%Y-%m-%d %H:%M}Z .. "
            f"{end:%Y-%m-%d %H:%M}Z every {step_h:g} h"
        )
    stats: dict = {}
    t = time.monotonic()
    df = fetch_history(ids, start, end, step_h=step_h, pause=pause, workers=workers, stats=stats)
    if log:
        log(
            f"taf: history {summarize(df)}; {stats.get('requests', 0)} requests "
            f"({stats.get('failed', 0)} failed), {stats.get('bytes', 0) / 1e6:.1f} MB, "
            f"{time.monotonic() - t:.0f}s"
        )
    with tempfile.TemporaryDirectory() as tmp:
        return store(storage, df, Path(tmp), log=log)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m stationtruth.tafs", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("collect", help="fetch the current TAF cache into taf-YYYY-MM-DD files")
    sp.add_argument("--data-dir", help="local directory instead of GitHub Release assets")

    sp = sub.add_parser("backfill", help="recover past TAFs from the Data API (about 30 days)")
    sp.add_argument("--days", type=float, default=7)
    sp.add_argument("--step-h", type=float, default=1.0, help="hours between API snapshots")
    sp.add_argument("--pause", type=float, default=0.5, help="seconds between API requests")
    sp.add_argument("--workers", type=int, default=2, help="concurrent request streams")
    sp.add_argument("--ids", default="", help="comma list of stations (default: TAF cache)")
    sp.add_argument("--data-dir", help="local directory instead of GitHub Release assets")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = open_storage(args.data_dir, log=log)
    if args.command == "collect":
        collect(storage)
    else:
        ids = [s.strip().upper() for s in args.ids.split(",") if s.strip()]
        backfill(
            storage,
            args.days,
            step_h=args.step_h,
            pause=args.pause,
            workers=args.workers,
            station_ids=ids or None,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COLUMNS",
    "collect",
    "backfill",
    "expand_taf",
    "expand_reports",
    "fetch_cache",
    "fetch_history",
    "merge",
    "parse_cache_xml",
    "read_tafs",
    "split_raw_tafs",
    "taf_file_name",
    "write_tafs",
    "worst",
]
