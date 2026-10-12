"""``stationtruth`` command line: collect, backfill, sync and build the site."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import grib, models, registry
from . import obs as obs_mod
from . import stations as stations_mod
from .storage import STATIONS_FILE, Storage, months_between, open_storage

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- helpers


def refresh_stations(storage: Storage, work: Path) -> pd.DataFrame:
    t = time.monotonic()
    st = stations_mod.fetch_stations()
    path = stations_mod.write_stations(st, work / STATIONS_FILE)
    storage.put(path)
    log(f"stations: {len(st)} METAR stations ({time.monotonic() - t:.1f}s)")
    return st


def load_stations(storage: Storage, work: Path) -> pd.DataFrame:
    """Stored station list, or a fresh download (stored) when there is none."""
    path = storage.get(STATIONS_FILE, work)
    if path is not None:
        st = stations_mod.read_stations(path)
        if len(st):
            return st
    return refresh_stations(storage, work)


def store_obs(
    storage: Storage, new: pd.DataFrame, work: Path, days: set[date] | None = None
) -> int:
    """Merge ``new`` into the stored daily obs files; returns the number of new rows."""
    added = 0
    for day, part in sorted(obs_mod.split_by_day(new).items()):
        if days is not None and day not in days:
            log(f"obs: skipping {len(part)} reports dated {day}")
            continue
        name = obs_mod.obs_file_name(day)
        existing_path = storage.get(name, work)
        existing = obs_mod.read_obs(existing_path) if existing_path else None
        n_before = 0 if existing is None else len(existing)
        merged = obs_mod.merge_obs(existing, part)
        path = obs_mod.write_obs(merged, work / name)
        storage.put(path)
        added += len(merged) - n_before
        log(
            f"obs: {name}: {n_before} -> {len(merged)} reports "
            f"(+{len(merged) - n_before}), {path.stat().st_size / 1e6:.2f} MB"
        )
    return added


def stored_names(storage: Storage, start: datetime, end: datetime) -> set[str]:
    return set(storage.list(months_between(start.date(), end.date())))


def fetch_and_store_run(
    storage: Storage, source: models.ModelSource, run: datetime, st: pd.DataFrame, work: Path
) -> bool:
    name = source.file_name(run)
    log(f"{source.name}: fetching run {run:%Y-%m-%d %HZ} ({len(source.steps)} steps)")
    df, stats = grib.fetch_run(source, run, st, log=log)
    if stats["failed_steps"]:
        log(
            f"{source.name}: run {run:%Y%m%d%H} incomplete (failed steps "
            f"{stats['failed_steps']}); not stored, will retry"
        )
        return False
    path = work / name
    grib.write_fcst(df, path)
    storage.put(path)
    log(
        f"{source.name}: {name}: {len(df)} rows, downloaded {stats['bytes'] / 1e6:.1f} MB "
        f"in {stats['requests']} requests, {stats['seconds']:.0f}s; "
        f"file {path.stat().st_size / 1e6:.2f} MB"
    )
    path.unlink(missing_ok=True)
    return True


def _models(arg: str | None) -> list[models.ModelSource]:
    """``all`` -> every registry model; else a comma list (``ValueError`` if unknown)."""
    return [models.get_source(n) for n in registry.parse_list(arg)]


def _model_list(arg: str) -> str:
    """argparse type: validate a model list early, with the valid choices in the error."""
    try:
        registry.parse_list(arg)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return arg


MODEL_HELP = f"{', '.join(registry.RAW_MODELS)} or all (comma list ok)"


# --------------------------------------------------------------------------- commands


def cmd_collect_obs(args) -> int:
    storage = open_storage(args.data_dir, log=log)
    now = utcnow()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            st = refresh_stations(storage, work)
        except Exception as exc:  # a stale station list must not stop collection
            log(f"stations: refresh failed ({exc}); using the stored list")
            st = load_stations(storage, work)
        t = time.monotonic()
        new = obs_mod.fetch_latest(now=now)
        log(f"obs: METAR cache: {obs_mod.summarize(new)} ({time.monotonic() - t:.1f}s)")
        if args.recent_hours > 0:
            ids = set(new["station"])
            if args.all_stations:
                ids |= set(st["station"])
            stats: dict = {}
            t = time.monotonic()
            hist = obs_mod.fetch_history(
                ids,
                now - timedelta(hours=args.recent_hours),
                now,
                chunk=400,
                pause=args.pause,
                workers=3,
                log=None,
                stats=stats,
            )
            log(
                f"obs: Data API last {args.recent_hours} h: {obs_mod.summarize(hist)} "
                f"({stats.get('requests', 0)} requests, {stats.get('bytes', 0) / 1e6:.1f} MB, "
                f"{time.monotonic() - t:.0f}s)"
            )
            new = obs_mod.merge_obs(hist, new)
        today = now.date()
        store_obs(storage, new, work, days={today, today - timedelta(days=1)})
    return 0


def cmd_collect_fcst(args) -> int:
    storage = open_storage(args.data_dir, log=log)
    now = utcnow()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        st = load_stations(storage, work)
        existing = stored_names(storage, now - timedelta(days=3), now)
        for source in _models(args.model or "all"):
            done = 0
            for run in grib.candidate_runs(source, now, lookback_h=args.lookback_h):
                if done >= args.runs:
                    break
                if source.file_name(run) in existing:
                    continue
                if not grib.run_available(source, run):
                    log(f"{source.name}: run {run:%Y%m%d%H} not available yet")
                    continue
                if fetch_and_store_run(storage, source, run, st, work):
                    done += 1
                else:
                    ok = False
            if done == 0:
                log(f"{source.name}: nothing new to fetch")
    return 0 if ok else 1


def cmd_backfill(args) -> int:
    storage = open_storage(args.data_dir, log=log)
    now = utcnow()
    start = now - timedelta(days=args.days)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        st = refresh_stations(storage, work)
        if not args.no_obs:
            ids = list(st["station"])
            if args.max_stations:
                ids = ids[: args.max_stations]
            log(f"obs: backfilling {len(ids)} stations from {start:%Y-%m-%d %H:%M}Z")
            stats: dict = {}
            t = time.monotonic()
            hist = obs_mod.fetch_history(
                ids,
                start,
                now,
                chunk=args.chunk,
                pause=args.pause,
                workers=args.workers,
                log=log,
                stats=stats,
            )
            log(
                f"obs: history {obs_mod.summarize(hist)}; {stats.get('requests', 0)} requests, "
                f"{stats.get('bytes', 0) / 1e6:.1f} MB, {time.monotonic() - t:.0f}s"
            )
            store_obs(storage, hist, work)
        sources = _models(args.models) if args.models else []
        if sources:
            existing = stored_names(storage, start, now)
            for source in sources:
                runs = grib.candidate_runs(source, now, lookback_h=args.days * 24)
                for run in sorted(runs):
                    if source.file_name(run) in existing:
                        continue
                    if not grib.run_available(source, run):
                        log(f"{source.name}: run {run:%Y%m%d%H} not available")
                        continue
                    ok &= fetch_and_store_run(storage, source, run, st, work)
    return 0 if ok else 1


def cmd_sync(args) -> int:
    storage = open_storage(args.from_dir, log=log)
    now = utcnow()
    start = (now - timedelta(days=args.days)).replace(hour=0, minute=0, second=0, microsecond=0)
    storage.sync(Path(args.to), start, now, prune=args.prune, log=log)
    return 0


def cmd_build_site(args) -> int:
    from . import sitegen  # written separately; imported lazily

    t = time.monotonic()
    summary = sitegen.build(Path(args.data_dir), Path(args.out))
    log(f"site built in {time.monotonic() - t:.1f}s")
    print(json.dumps(summary, default=str, indent=1))
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stationtruth",
        description="Verify global weather models ("
        + ", ".join(m.long for m in registry.REGISTRY)
        + ") against airport METARs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def data_dir(sp):
        sp.add_argument(
            "--data-dir",
            help="work on a local directory instead of GitHub Release assets",
        )

    sp = sub.add_parser("collect-obs", help="fetch latest METARs into today's obs file")
    data_dir(sp)
    sp.add_argument(
        "--recent-hours",
        type=float,
        default=0,
        help="also pull the last N hours from the Data API (catches reports between runs)",
    )
    sp.add_argument(
        "--all-stations",
        action="store_true",
        help="with --recent-hours: query every listed station, not just those in the cache",
    )
    sp.add_argument("--pause", type=float, default=0.5, help="seconds between API requests")
    sp.set_defaults(func=cmd_collect_obs)

    sp = sub.add_parser("collect-fcst", help="fetch the latest model runs not stored yet")
    sp.add_argument("--model", default="all", type=_model_list, help=MODEL_HELP)
    sp.add_argument("--runs", type=int, default=1, help="max new runs per model")
    sp.add_argument("--lookback-h", type=int, default=48, help="how far back to look for runs")
    data_dir(sp)
    sp.set_defaults(func=cmd_collect_fcst)

    sp = sub.add_parser("backfill", help="fill history: obs via the Data API plus model runs")
    sp.add_argument("--days", type=float, default=7)
    sp.add_argument(
        "--models", default="all", type=_model_list, help=MODEL_HELP + "; empty for none"
    )
    sp.add_argument("--no-obs", action="store_true")
    sp.add_argument("--chunk", type=int, default=200, help="station ids per API request")
    sp.add_argument("--pause", type=float, default=0.5, help="seconds between API requests")
    sp.add_argument("--workers", type=int, default=3, help="concurrent Data API request streams")
    sp.add_argument("--max-stations", type=int, default=0, help=argparse.SUPPRESS)
    data_dir(sp)
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("sync", help="download the data needed for the last N days")
    sp.add_argument("--days", type=int, default=35)
    sp.add_argument("--to", required=True, help="destination directory")
    sp.add_argument("--from-dir", help="copy from a local data directory instead of GitHub")
    sp.add_argument("--prune", action="store_true", help="delete data files outside the window")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("build-site", help="verify, correct and write site JSON")
    sp.add_argument("--data-dir", required=True)
    sp.add_argument("--out", default="site/data")
    sp.set_defaults(func=cmd_build_site)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
