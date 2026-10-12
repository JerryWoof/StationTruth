"""Forecast model sources: where the GRIB2 files live and which messages we need.

Each source knows how to build the data and index URLs for a run/step, how
to read its index format, and how to pick the messages for our variables.
Internal variable names: ``t``, ``td`` (K), ``u``, ``v``, ``gust`` (m/s),
``mslp`` (Pa), ``tcc`` (fraction or %), ``vis`` (m). :mod:`stationtruth.grib`
converts them to the contract units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

STEPS: tuple[int, ...] = tuple(range(0, 73, 3))
RUN_HOURS: tuple[int, ...] = (0, 12)


@dataclass(frozen=True)
class IndexEntry:
    """One GRIB message listed in an index: byte offset, length (None = to EOF)."""

    offset: int
    length: int | None
    attrs: dict = field(default_factory=dict, hash=False, compare=False)


def parse_jsonl_index(text: str) -> list[IndexEntry]:
    """ECMWF open-data ``.index``: one JSON object per line with ``_offset``/``_length``."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_offset" not in rec:
            continue
        out.append(IndexEntry(int(rec["_offset"]), int(rec.get("_length") or 0) or None, rec))
    return out


def parse_wgrib_idx(text: str) -> list[IndexEntry]:
    """NCEP wgrib2 ``.idx``: ``n:offset:d=YYYYMMDDHH:VAR:LEVEL:FCST:``.

    The length of a message is the next message's offset minus its own; the
    last message runs to the end of the file (length ``None``).
    """
    recs = []
    for line in text.splitlines():
        parts = line.strip().split(":")
        if len(parts) < 6:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        recs.append(
            {
                "n": parts[0],
                "offset": offset,
                "var": parts[3],
                "level": parts[4],
                "fcst": parts[5],
            }
        )
    offsets = sorted({r["offset"] for r in recs})
    nxt = {a: b for a, b in zip(offsets, offsets[1:], strict=False)}
    out = []
    for r in recs:
        end = nxt.get(r["offset"])
        out.append(IndexEntry(r["offset"], None if end is None else end - r["offset"], r))
    return out


@dataclass(frozen=True)
class ModelSource:
    name: str
    steps: tuple[int, ...] = STEPS
    run_hours: tuple[int, ...] = RUN_HOURS

    # -- to override
    def grib_url(self, run: datetime, step: int) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def index_url(self, run: datetime, step: int) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def parse_index(self, text: str) -> list[IndexEntry]:  # pragma: no cover - abstract
        raise NotImplementedError

    def select(self, entries: list[IndexEntry], step: int) -> dict[str, IndexEntry]:
        raise NotImplementedError  # pragma: no cover - abstract

    # -- shared
    def file_name(self, run: datetime) -> str:
        return f"fcst-{self.name}-{run:%Y%m%d%H}.parquet"


@dataclass(frozen=True)
class EcmwfOpenData(ModelSource):
    name: str = "ecmwf"
    base: str = "https://data.ecmwf.int/forecasts"
    # internal variable -> ECMWF short name (levtype sfc)
    params: tuple[tuple[str, str], ...] = (
        ("t", "2t"),
        ("td", "2d"),
        ("u", "10u"),
        ("v", "10v"),
        ("gust", "10fg"),
        ("mslp", "msl"),
        ("tcc", "tcc"),
    )

    def _stem(self, run: datetime, step: int) -> str:
        return (
            f"{self.base}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/oper/{run:%Y%m%d%H}0000-{step}h-oper-fc"
        )

    def grib_url(self, run: datetime, step: int) -> str:
        return self._stem(run, step) + ".grib2"

    def index_url(self, run: datetime, step: int) -> str:
        return self._stem(run, step) + ".index"

    def parse_index(self, text: str) -> list[IndexEntry]:
        return parse_jsonl_index(text)

    def select(self, entries: list[IndexEntry], step: int) -> dict[str, IndexEntry]:
        by_param: dict[str, IndexEntry] = {}
        for e in entries:
            a = e.attrs
            if a.get("levtype") != "sfc" or "levelist" in a:
                continue
            if a.get("type", "fc") != "fc":
                continue
            by_param.setdefault(a.get("param"), e)
        out = {}
        for var, param in self.params:
            # 10fg is the maximum gust since the previous post-processing step;
            # at step 0 there is no such period, so it is left missing.
            if var == "gust" and step == 0:
                continue
            if param in by_param:
                out[var] = by_param[param]
        return out


@dataclass(frozen=True)
class GfsAws(ModelSource):
    name: str = "gfs"
    base: str = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    # internal variable -> (wgrib VAR, LEVEL)
    params: tuple[tuple[str, str, str], ...] = (
        ("t", "TMP", "2 m above ground"),
        ("td", "DPT", "2 m above ground"),
        ("u", "UGRD", "10 m above ground"),
        ("v", "VGRD", "10 m above ground"),
        ("gust", "GUST", "surface"),
        ("mslp", "PRMSL", "mean sea level"),
        ("tcc", "TCDC", "entire atmosphere"),
        ("vis", "VIS", "surface"),
    )

    def grib_url(self, run: datetime, step: int) -> str:
        return f"{self.base}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}"

    def index_url(self, run: datetime, step: int) -> str:
        return self.grib_url(run, step) + ".idx"

    def parse_index(self, text: str) -> list[IndexEntry]:
        return parse_wgrib_idx(text)

    @staticmethod
    def _instantaneous(fcst: str) -> bool:
        f = fcst.lower()
        return not any(w in f for w in ("ave", "acc", "max", "min", "-"))

    def select(self, entries: list[IndexEntry], step: int) -> dict[str, IndexEntry]:
        out = {}
        for var, name, level in self.params:
            for e in entries:
                a = e.attrs
                if a.get("var") == name and a.get("level") == level:
                    if self._instantaneous(a.get("fcst", "")):
                        out[var] = e
                        break
        return out


ECMWF = EcmwfOpenData()
GFS = GfsAws()
SOURCES: dict[str, ModelSource] = {"ecmwf": ECMWF, "gfs": GFS}


def get_source(name: str) -> ModelSource:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(SOURCES)}") from None
