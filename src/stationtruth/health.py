"""Pipeline health (docs/DATA.md, "health.json").

Looks only at the data directory (and the site output directory) and answers:
is data still flowing? It reports

* observations: newest report, reports and reporting stations per hour over the
  last 48 h (hours with no or unusually few reports are flagged), stations
  reporting in the last 3 h / 24 h;
* model runs: for every model found in the data directory, the 00Z/12Z cycles
  of the last 7 days that are stored, missing or not yet due, with the time
  each was stored (when known) and how long after initialisation;
* TAF files, the analysis map and the TAF scores;
* the site build (time, step timings, errors of optional steps).

"Stored" times come from the asset listing: ``data_dir/.sync.json`` written by
``Storage.sync`` holds each file's size and update stamp (GitHub asset
``updated_at``, or the source file's mtime for a local store). Local file
mtimes are *not* used: they tell when a file was copied, not when it was
collected. Without a listing, lateness is left out.

The overall status is the worst reason: ``down`` (no recent observations, or no
model run in 36 h), ``degraded`` (a recent gap, a missing latest run, stale
TAFs or analysis) or ``ok``. Older incidents are listed with level ``info``.

Run standalone after the other build steps (so analysis/TAF status is final)::

    python -m stationtruth.health --data-dir data --out site/data
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .verify import _arrow_minutes, from_minutes, iso, run_id, to_minutes

log = logging.getLogger(__name__)

HOURS = 48
#: an hour counts as complete once it ended this many minutes ago (reports are
#: published a few minutes late and collected hourly at :07)
SETTLE_MIN = 70
#: a complete hour with fewer reports than this share of the 48 h median is "low"
LOW_FRAC = 0.5
#: gaps in this many most recent hours make the status "degraded"
RECENT_GAP_H = 6
OBS_DOWN_MIN = 180
RUN_DAYS = 7
CYCLE_HOURS: tuple[int, ...] = (0, 12)
#: a cycle is "due" once it is this many hours old (dissemination + collection)
GRACE_H: dict[str, int] = {"ecmwf": 12, "gfs": 10}
DEFAULT_GRACE_H = 12
RUN_DOWN_H = 36
TAF_STALE_H = 8
ANALYSIS_STALE_MIN = 180

_FCST_RE = re.compile(r"^fcst-([a-z0-9_]+)-(\d{10})\.parquet$")
_OBS_RE = re.compile(r"^obs-(\d{4}-\d{2}-\d{2})\.parquet$")
_TAF_RE = re.compile(r"^taf-(\d{4}-\d{2}-\d{2})\.parquet$")

LEVELS = {"info": 0, "degraded": 1, "down": 2}


# ------------------------------------------------------------------ helpers


def _parse_stamp(stamp) -> int | None:
    """Update stamp of a listing entry -> epoch minutes (ISO string or ns int)."""
    if stamp is None:
        return None
    s = str(stamp)
    if "|" in s:  # .sync.json value "size|updated"
        s = s.split("|", 1)[1]
    if s.isdigit():
        try:
            return int(int(s) // 60_000_000_000)
        except (ValueError, OverflowError):
            return None
    try:
        return to_minutes(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        return None


def read_listing(data_dir: Path) -> dict[str, int]:
    """``{file name: stored time (epoch minutes)}`` from ``data_dir/.sync.json``."""
    try:
        raw = json.loads((Path(data_dir) / ".sync.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for name, stamp in raw.items():
        m = _parse_stamp(stamp)
        if m is not None:
            out[name] = m
    return out


def _rows(path: Path) -> int | None:
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception:
        return None


def _reason(reasons: list, level: str, component: str, msg: str) -> None:
    reasons.append({"level": level, "component": component, "msg": msg})


def _fmt_h(minutes: int) -> str:
    return from_minutes(minutes).strftime("%d %b %H:%M UTC").lstrip("0")


# ------------------------------------------------------------------ obs


def obs_health(data_dir: Path, now: int, listing: dict[str, int], reasons: list) -> dict:
    files = {}
    for p in sorted(data_dir.glob("obs-*.parquet")):
        m = _OBS_RE.match(p.name)
        if m:
            files[m.group(1)] = p
    h_end = now // 60 * 60  # start of the current (partial) hour
    h0 = h_end - (HOURS - 1) * 60
    lo = h0 - 24 * 60  # also the 24 h before, for "stations in the last 24 h"
    times, stations = [], []
    for day, p in files.items():
        d = to_minutes(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc))
        if d + 1440 <= min(lo, now - 1440) or d > now:
            continue
        try:
            tbl = pq.read_table(p, columns=["station", "time"])
        except Exception as exc:
            log.warning("cannot read %s: %s", p, exc)
            continue
        tm = _arrow_minutes(tbl.column("time"))
        st = np.asarray(tbl.column("station").to_numpy(zero_copy_only=False), dtype=object)
        ok = (tm <= now) & (tm >= min(lo, now - 1440))
        times.append(tm[ok])
        stations.append(st[ok])
    tm = np.concatenate(times) if times else np.zeros(0, np.int64)
    st = np.concatenate(stations) if stations else np.zeros(0, object)

    last = int(tm.max()) if len(tm) else None
    nh = HOURS
    reports = np.zeros(nh, dtype=np.int64)
    stat_h = np.zeros(nh, dtype=np.int64)
    sel = tm >= h0
    if sel.any():
        hi = ((tm[sel] - h0) // 60).astype(np.int64)
        reports = np.bincount(hi, minlength=nh)[:nh]
        # unique stations per hour
        codes = np.unique(st[sel], return_inverse=True)[1].astype(np.int64)
        pair = np.unique(hi * (codes.max() + 1) + codes)
        stat_h = np.bincount(pair // (codes.max() + 1), minlength=nh)[:nh]
    hour_start = h0 + np.arange(nh) * 60
    settled = hour_start + 60 + SETTLE_MIN <= now
    med = float(np.median(reports[settled])) if settled.any() else 0.0
    state = []
    for i in range(nh):
        if not settled[i]:
            state.append("partial")
        elif reports[i] == 0:
            state.append("gap")
        elif med > 0 and reports[i] < LOW_FRAC * med:
            state.append("low")
        else:
            state.append("ok")
    # contiguous runs of gap/low hours
    gaps = []
    i = 0
    while i < nh:
        if state[i] in ("gap", "low"):
            j = i
            while j + 1 < nh and state[j + 1] in ("gap", "low"):
                j += 1
            kind = "gap" if all(state[k] == "gap" for k in range(i, j + 1)) else "low"
            gaps.append(
                {
                    "from": iso(int(hour_start[i])),
                    "to": iso(int(hour_start[j]) + 60),
                    "hours": j - i + 1,
                    "kind": kind,
                }
            )
            i = j + 1
        else:
            i += 1
    recent_from = now - RECENT_GAP_H * 60
    for g in gaps:
        end = to_minutes(datetime.fromisoformat(g["to"].replace("Z", "+00:00")))
        what = "no reports" if g["kind"] == "gap" else "unusually few reports"
        span = f"{g['hours']} h from {_fmt_h(end - g['hours'] * 60)}"
        if end > recent_from:
            _reason(reasons, "degraded", "obs", f"Observations: {what} for {span}.")
        else:
            _reason(reasons, "info", "obs", f"Earlier observation gap: {what} for {span}.")

    if last is None:
        _reason(reasons, "down", "obs", "No observations found in the last 48 hours.")
    elif now - last > OBS_DOWN_MIN:
        _reason(
            reasons,
            "down",
            "obs",
            f"Newest observation is {(now - last) / 60:.1f} h old (collection stopped?).",
        )

    s3 = int(len(np.unique(st[tm > now - 180]))) if len(st) else 0
    s24 = int(len(np.unique(st[tm > now - 1440]))) if len(st) else 0
    recent_files = sorted(files)[-RUN_DAYS:]
    file_info = [
        {
            "date": d,
            "rows": _rows(files[d]),
            "stored": iso(listing.get(files[d].name)),
        }
        for d in recent_files
    ]
    collect = [listing[files[d].name] for d in recent_files if files[d].name in listing]
    return {
        "last_obs": iso(last),
        "age_min": None if last is None else int(now - last),
        "last_collect": iso(max(collect)) if collect else None,
        "stations_3h": s3,
        "stations_24h": s24,
        "median_reports": int(round(med)),
        "hours": {
            "start": iso(int(h0)),
            "step": 60,
            "reports": reports.astype(int).tolist(),
            "stations": stat_h.astype(int).tolist(),
            "state": state,
        },
        "gaps": gaps,
        "files": file_info,
    }


# ------------------------------------------------------------------ runs


def runs_health(data_dir: Path, now: int, listing: dict[str, int], reasons: list) -> dict:
    by_model: dict[str, dict[int, Path]] = {}
    for p in data_dir.glob("fcst-*.parquet"):
        m = _FCST_RE.match(p.name)
        if not m:
            continue
        try:
            run = to_minutes(datetime.strptime(m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc))
        except ValueError:
            continue
        if run <= now:
            by_model.setdefault(m.group(1), {})[run] = p
    start = now - RUN_DAYS * 1440
    first_cycle = (start // 720 + 1) * 720  # next 00/12Z after the window start
    cycles = [c for c in range(first_cycle, now + 1, 720) if (c // 60) % 24 in CYCLE_HOURS]
    models: dict = {}
    newest_any = None
    for model in sorted(by_model):
        runs = by_model[model]
        if not any(r >= now - 2 * RUN_DAYS * 1440 for r in runs):
            continue  # not collected for two weeks: retired model, not an outage
        grace = GRACE_H.get(model, DEFAULT_GRACE_H) * 60
        first_run = min(runs)
        rows = []
        for c in cycles:
            p = runs.get(c)
            if p is not None:
                stored = listing.get(p.name)
                rows.append(
                    {
                        "run": run_id(c),
                        "status": "ok",
                        "stored": iso(stored),
                        "late_h": None if stored is None else round((stored - c) / 60, 1),
                        "rows": _rows(p),
                    }
                )
            elif c < first_run:
                rows.append({"run": run_id(c), "status": "before"})  # model not yet collected
            elif now - c < grace:
                rows.append({"run": run_id(c), "status": "pending"})
            else:
                rows.append({"run": run_id(c), "status": "missing"})
        extra = sorted(run_id(r) for r in runs if r >= start and (r // 60) % 24 not in CYCLE_HOURS)
        latest = max(runs)
        newest_any = latest if newest_any is None else max(newest_any, latest)
        due = [r for r in rows if r["status"] in ("ok", "missing")]
        missing = [r["run"] for r in due if r["status"] == "missing"]
        late = [r["late_h"] for r in due if r.get("late_h") is not None]
        # trailing missing cycles = the current outage
        trail = 0
        for r in reversed(due):
            if r["status"] != "missing":
                break
            trail += 1
        if trail:
            _reason(
                reasons,
                "degraded",
                f"runs:{model}",
                f"{model}: latest {trail} due run{'s' if trail > 1 else ''} not stored "
                f"(newest stored {run_id(latest)}).",
            )
        if len(missing) > trail:
            _reason(
                reasons,
                "info",
                f"runs:{model}",
                f"{model}: {len(missing) - trail} earlier run(s) of the last {RUN_DAYS} days "
                "missing.",
            )
        models[model] = {
            "latest": run_id(latest),
            "latest_age_h": round((now - latest) / 60, 1),
            "stored": sum(r["status"] == "ok" for r in due),
            "expected": len(due),
            "missing": missing,
            "extra": extra,
            "median_late_h": round(float(np.median(late)), 1) if late else None,
            "grace_h": grace // 60,
            "cycles": rows,
        }
    if newest_any is None:
        _reason(reasons, "down", "runs", "No model runs found in the data directory.")
    elif now - newest_any > RUN_DOWN_H * 60:
        _reason(
            reasons,
            "down",
            "runs",
            f"Newest model run is {(now - newest_any) / 60:.0f} h old "
            "(forecast collection stopped?).",
        )
    return {
        "days": RUN_DAYS,
        "cycle_hours": list(CYCLE_HOURS),
        "listing": bool(listing),
        "models": models,
    }


# ------------------------------------------------------------------ TAFs, analysis


def taf_health(data_dir: Path, now: int, listing: dict[str, int], reasons: list) -> dict:
    files = []
    for p in sorted(data_dir.glob("taf-*.parquet")):
        m = _TAF_RE.match(p.name)
        if m:
            files.append((m.group(1), p))
    files = files[-RUN_DAYS:]
    info = [{"date": d, "rows": _rows(p), "stored": iso(listing.get(p.name))} for d, p in files]
    last_issued = None
    for _, p in reversed(files):
        try:
            tbl = pq.read_table(p, columns=["issued"])
            tm = _arrow_minutes(tbl.column("issued"))
            tm = tm[(tm <= now) & (tm > 0)]
            if len(tm):
                last_issued = int(tm.max())
                break
        except Exception as exc:
            log.warning("cannot read %s: %s", p, exc)
    if not files:
        _reason(reasons, "info", "tafs", "No TAF files in the data directory.")
    elif last_issued is None or now - last_issued > TAF_STALE_H * 60:
        age = "unknown" if last_issued is None else f"{(now - last_issued) / 60:.0f} h"
        _reason(reasons, "degraded", "tafs", f"Newest stored TAF is {age} old.")
    return {
        "last_issued": iso(last_issued),
        "age_h": None if last_issued is None else round((now - last_issued) / 60, 1),
        "files": info,
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _minutes(s) -> int | None:
    try:
        return to_minutes(datetime.fromisoformat(str(s).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def products_health(out_dir: Path, now: int, reasons: list) -> dict:
    a = _read_json(out_dir / "analysis" / "analysis.json")
    if a is None:
        analysis = {"status": "not_built"}
        _reason(
            reasons,
            "info",
            "analysis",
            "Analysis map not built yet in this build (it is made after the scores).",
        )
    else:
        gen = _minutes(a.get("generated"))
        age = None if gen is None else max(0, now - gen)
        stale = age is None or age > ANALYSIS_STALE_MIN
        analysis = {
            "status": "stale" if stale else "ok",
            "generated": a.get("generated"),
            "valid": a.get("valid"),
            "age_min": age,
            "seconds": (a.get("runtime") or {}).get("seconds"),
        }
        if stale:
            _reason(reasons, "degraded", "analysis", "Analysis map is out of date.")
    t = _read_json(out_dir / "taf_scores.json")
    taf_scores = {"status": "not_built"}
    if t is not None:
        gen = _minutes(t.get("generated"))
        taf_scores = {
            "status": "ok",
            "generated": t.get("generated"),
            "age_min": None if gen is None else max(0, now - gen),
            "last_hour": t.get("last_hour"),
            "taf_count": t.get("taf_count"),
        }
    return {"analysis": analysis, "taf_scores": taf_scores}


# ------------------------------------------------------------------ build


def compute(
    data_dir: Path,
    out_dir: Path,
    now: datetime | None = None,
    meta: dict | None = None,
    build_info: dict | None = None,
    listing: dict[str, int] | None = None,
) -> dict:
    """The ``health.json`` document."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_m = to_minutes(now)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if listing is None:
        listing = read_listing(data_dir)
    reasons: list = []
    doc: dict = {"generated": iso(now_m)}
    doc["obs"] = obs_health(data_dir, now_m, listing, reasons)
    doc["runs"] = runs_health(data_dir, now_m, listing, reasons)
    doc["tafs"] = taf_health(data_dir, now_m, listing, reasons)
    doc.update(products_health(out_dir, now_m, reasons))
    if meta is None:
        meta = _read_json(out_dir / "meta.json") or {}
    build = {
        "generated": meta.get("generated"),
        "models": meta.get("models"),
        "pairs": meta.get("pairs"),
        "station_count": meta.get("station_count"),
        **(build_info or {}),
    }
    for step, err in (build.get("errors") or {}).items():
        _reason(reasons, "degraded", "build", f"Build step '{step}' failed: {err}")
    doc["build"] = build
    worst = max((LEVELS[r["level"]] for r in reasons), default=0)
    doc["status"] = {0: "ok", 1: "degraded", 2: "down"}[worst]
    order = sorted(reasons, key=lambda r: -LEVELS[r["level"]])
    doc["reasons"] = order
    return doc


def build(
    data_dir: Path,
    out_dir: Path,
    now: datetime | None = None,
    meta: dict | None = None,
    build_info: dict | None = None,
    listing: dict[str, int] | None = None,
) -> dict:
    """Compute and write ``out_dir/health.json``; returns the document."""
    doc = compute(data_dir, out_dir, now, meta, build_info, listing)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "health.json"
    old = _read_json(path) if build_info is None else None
    if (
        old
        and isinstance(old.get("build"), dict)
        and doc["build"].get("generated") == old["build"].get("generated")
    ):
        # standalone re-run after the other steps: keep the site build's details
        for k, v in old["build"].items():
            doc["build"].setdefault(k, v)
        for step, err in (old["build"].get("errors") or {}).items():
            _reason(doc["reasons"], "degraded", "build", f"Build step '{step}' failed: {err}")
        worst = max((LEVELS[r["level"]] for r in doc["reasons"]), default=0)
        doc["status"] = {0: "ok", 1: "degraded", 2: "down"}[worst]
        doc["reasons"].sort(key=lambda r: -LEVELS[r["level"]])
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m stationtruth.health",
        description="Write site/data/health.json (pipeline health).",
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="site/data")
    args = p.parse_args(argv)
    doc = build(Path(args.data_dir), Path(args.out))
    print(f"health: {doc['status']} ({len(doc['reasons'])} reasons)")
    for r in doc["reasons"]:
        print(f"  [{r['level']}] {r['msg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
