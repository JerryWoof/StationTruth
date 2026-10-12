"""Synthetic StationTruth data with known model biases.

Writes ``stations.parquet``, ``obs-YYYY-MM-DD.parquet`` and
``fcst-<model>-YYYYMMDDHH.parquet`` files following docs/DATA.md.

The "truth" at each station is an analytic smooth function of time (diurnal
cycle + a few synoptic-scale sinusoids with station-specific phases), so obs
and forecasts can be evaluated at any minute. Observations are truth + small
noise (rounded like METARs); forecasts are truth + a known bias + noise that
grows with lead time. With hourly METARs at HH:50 every forecast valid time
HH:00 has an observation 10 minutes earlier.

Usable from tests (``make_dataset``) and from the command line for
development/benchmarks::

    python tests/synth.py OUT_DIR --stations 5000 --days 35
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

UTC = timezone.utc
START = datetime(2026, 9, 20, tzinfo=UTC)


@dataclass
class ModelSpec:
    bias: dict[str, float]
    diurnal_t: float = 0.0  # amplitude of an extra UTC-hour dependent t bias
    noise: float = 0.6  # error sd at lead 0 (deg C / kt / hPa), grows with lead
    vis: bool = False


DEFAULT_MODELS: dict[str, ModelSpec] = {
    "ecmwf": ModelSpec(
        {"t": 0.5, "td": -0.3, "wspd": 1.0, "gust": 2.0, "mslp": 0.4, "wdir": 10.0},
        diurnal_t=0.5,
    ),
    "gfs": ModelSpec(
        {"t": -1.0, "td": 0.6, "wspd": -1.5, "gust": -2.0, "mslp": -0.5, "wdir": -20.0},
        diurnal_t=1.0,
        noise=0.8,
        vis=True,
    ),
}

BASE_STATIONS = [
    ("LKPR", "Prague/Havel Arpt", "CZ", 50.101, 14.26, 380.0),
    ("KJFK", "New York/JFK", "US", 40.639, -73.779, 4.0),
    ("EGLL", "London/Heathrow", "GB", 51.477, -0.461, 25.0),
    ("YSSY", "Sydney", "AU", -33.946, 151.177, 6.0),
    ("SBGR", "Sao Paulo/Guarulhos", "BR", -23.432, -46.469, 750.0),
    ("ENSB", "Svalbard/Longyear", "NO", 78.246, 15.466, 28.0),
]


@dataclass
class SynthInfo:
    data_dir: Path
    start: datetime
    now: datetime
    stations: list[str]
    models: dict[str, ModelSpec] = field(default_factory=dict)
    slp_stations: list[str] = field(default_factory=list)


def _stations(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = list(BASE_STATIONS[:n])
    countries = ["DE", "FR", "US", "CA", "JP", "CN", "ZA", "IN", "AR", "AU", "MX", "SA", "RU"]
    i = 0
    while len(rows) < n:
        sid = f"X{i:03d}" if i < 1000 else f"Y{i:04d}"
        i += 1
        lat = float(np.round(rng.uniform(-60, 75), 3))
        lon = float(np.round(rng.uniform(-180, 179.9), 3))
        rows.append((sid, f"Synthetic {sid}", str(rng.choice(countries)), lat, lon, 100.0))
    df = pd.DataFrame(rows, columns=["station", "name", "country", "lat", "lon", "elev_m"])
    df["elev_m"] = df["elev_m"].astype("float32")
    return df


class Truth:
    """Analytic truth fields per station: f(sid_index, epoch_minutes)."""

    def __init__(self, st: pd.DataFrame, seed: int):
        rng = np.random.default_rng(seed + 1)
        n = len(st)
        self.lon = st["lon"].to_numpy()
        self.t0 = 25 - 0.35 * np.abs(st["lat"].to_numpy()) + rng.normal(0, 2, n)
        self.phase = rng.uniform(0, 2 * np.pi, (n, 4, 3))
        self.periods = np.array([2.3, 3.7, 5.9]) * 1440.0

    def _syn(self, i, m, k):
        ph = self.phase[i, k]  # (N, 3)
        x = 2 * np.pi * m[:, None] / self.periods[None, :] + ph
        return np.sin(x).sum(axis=1) / 1.7  # ~unit variance

    def fields(self, i: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
        m = m.astype(np.float64)
        local_h = (m / 60.0 + self.lon[i] / 15.0) % 24.0
        t = self.t0[i] + 5 * np.cos(2 * np.pi * (local_h - 15) / 24) + 3 * self._syn(i, m, 0)
        dep = 4 + 1.5 * self._syn(i, m, 1)
        wspd = np.maximum(12 + 3 * self._syn(i, m, 2), 0)
        wdir = (350 + 40 * self._syn(i, m, 3)) % 360
        slp = 1013 + 8 * self._syn(i, m, 1)
        return {"t": t, "td": t - np.abs(dep), "wspd": wspd, "wdir": wdir, "slp": slp}


def make_dataset(
    data_dir: Path,
    n_stations: int = 6,
    days: int = 20,
    start: datetime = START,
    models: dict[str, ModelSpec] | None = None,
    run_hours: tuple[int, ...] = (0, 12),
    run_delay_h: int = 6,
    speci_rate: float = 0.05,
    seed: int = 0,
) -> SynthInfo:
    """Write a synthetic dataset; ``now`` = start + days (obs up to then)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    models = DEFAULT_MODELS if models is None else models
    rng = np.random.default_rng(seed)
    st = _stations(n_stations, rng)
    st.to_parquet(data_dir / "stations.parquet", index=False)
    truth = Truth(st, seed)
    n = len(st)
    ids = st["station"].to_numpy()
    slp_mask = np.arange(n) % 2 == 1  # every other station reports SLP
    now = start + timedelta(days=days)
    m0 = int(start.timestamp() // 60)

    for d in range(days):
        day0 = m0 + d * 1440
        # METARs at HH:50 (half-hourly HH:20 too for station 2), SPECIs at random minutes
        mins = np.arange(day0 + 50, day0 + 1440, 60)
        si = np.repeat(np.arange(n), len(mins))
        sm = np.tile(mins, n)
        types = np.full(len(si), "METAR", dtype=object)
        if n > 2:
            extra = np.arange(day0 + 20, day0 + 1440, 60)
            si = np.r_[si, np.full(len(extra), 2)]
            sm = np.r_[sm, extra]
            types = np.r_[types, np.full(len(extra), "METAR", dtype=object)]
        n_sp = int(speci_rate * len(si))
        if n_sp:
            sp_i = rng.integers(0, n, n_sp)
            sp_m = day0 + rng.integers(0, 144, n_sp) * 10 + 5
            si = np.r_[si, sp_i]
            sm = np.r_[sm, sp_m]
            types = np.r_[types, np.full(n_sp, "SPECI", dtype=object)]
        f = truth.fields(si, sm)
        k = len(si)
        t = np.round(f["t"] + rng.normal(0, 0.2, k), 1)
        td = np.round(f["td"] + rng.normal(0, 0.3, k), 1)
        wspd = np.round(f["wspd"] + rng.normal(0, 0.8, k))
        wdir = np.round((f["wdir"] + rng.normal(0, 5, k)) % 360, -1) % 360
        gust = np.where(wspd >= 15, wspd + 8, np.nan)
        slp = np.where(slp_mask[si], np.round(f["slp"] + rng.normal(0, 0.2, k), 1), np.nan)
        obs = pd.DataFrame(
            {
                "station": ids[si],
                "time": pd.to_datetime(sm * 60, unit="s", utc=True),
                "type": types.astype(str),
                "t": t.astype("float32"),
                "td": td.astype("float32"),
                "wdir": wdir.astype("float32"),
                "wspd": wspd.astype("float32"),
                "gust": gust.astype("float32"),
                "vis_m": np.full(k, 10000, dtype="float32"),
                "ceil_ft": np.full(k, np.nan, dtype="float32"),
                "qnh": np.round(f["slp"], 0).astype("float32"),
                "slp": slp.astype("float32"),
                "cat": np.full(k, "VFR"),
            }
        )
        obs = obs[sm <= int(now.timestamp() // 60)]
        obs = obs.sort_values(["station", "time"]).drop_duplicates(["station", "time"])
        date = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        obs.to_parquet(data_dir / f"obs-{date}.parquet", index=False)

    leads = np.arange(0, 73, 3)
    run = start
    while run + timedelta(hours=run_delay_h) <= now:
        if run.hour in run_hours:
            rm = int(run.timestamp() // 60)
            for name, spec in models.items():
                write_forecast(data_dir, name, spec, truth, ids, rm, leads, rng)
        run += timedelta(hours=1)
    return SynthInfo(data_dir, start, now, list(ids), dict(models), list(ids[slp_mask].tolist()))


def write_forecast(data_dir, name, spec, truth, ids, run_m, leads, rng) -> None:
    n = len(ids)
    si = np.repeat(np.arange(n), len(leads))
    lead = np.tile(leads, n)
    valid = run_m + lead * 60
    f = truth.fields(si, valid)
    k = len(si)
    sd = spec.noise * (1 + lead / 48.0)
    b = spec.bias
    hour = (valid // 60) % 24
    t = f["t"] + b["t"] + spec.diurnal_t * np.cos(2 * np.pi * hour / 24) + rng.normal(0, sd)
    wspd = np.maximum(f["wspd"] + b["wspd"] + rng.normal(0, sd), 0)
    df = pd.DataFrame(
        {
            "station": ids[si],
            "run": pd.to_datetime(np.full(k, run_m * 60), unit="s", utc=True),
            "valid": pd.to_datetime(valid * 60, unit="s", utc=True),
            "lead_h": lead.astype("int16"),
            "t": t.astype("float32"),
            "td": (f["td"] + b["td"] + rng.normal(0, sd)).astype("float32"),
            "wspd": wspd.astype("float32"),
            "wdir": ((f["wdir"] + b["wdir"] + rng.normal(0, 5 * sd)) % 360).astype("float32"),
            "gust": (wspd + 8 + b["gust"] - b["wspd"]).astype("float32"),
            "mslp": (f["slp"] + b["mslp"] + rng.normal(0, sd)).astype("float32"),
            "tcc": rng.uniform(0, 100, k).astype("float32"),
            "vis_m": (np.full(k, 20000.0) if spec.vis else np.full(k, np.nan)).astype("float32"),
        }
    )
    stamp = datetime.fromtimestamp(run_m * 60, UTC).strftime("%Y%m%d%H")
    df.to_parquet(data_dir / f"fcst-{name}-{stamp}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--stations", type=int, default=50)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--start", default=START.strftime("%Y-%m-%d"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-hours", default="0,12", help="comma-separated run hours")
    a = ap.parse_args()
    hours = tuple(int(h) for h in a.run_hours.split(","))
    start = datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=UTC)
    info = make_dataset(
        a.out, n_stations=a.stations, days=a.days, start=start, seed=a.seed, run_hours=hours
    )
    print(f"wrote {a.out} now={info.now:%Y-%m-%dT%H:%MZ}")


if __name__ == "__main__":
    main()
