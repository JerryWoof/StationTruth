"""Forecast model sources: where the GRIB2 files live and which messages we need.

Each source knows how to build the data and index URLs for a run/step, how
to read its index format, and how to pick the messages for our variables.
Internal variable names: ``t``, ``td`` (K), ``u``, ``v``, ``gust`` (m/s),
``mslp`` (Pa), ``tcc`` (fraction or %), ``vis`` (m). :mod:`stationtruth.grib`
converts them to the contract units.

Two download layouts (``ModelSource.layout``):

* ``"index"``: one GRIB2 file per step with a byte-offset index; only the
  needed messages are fetched with range requests (ECMWF IFS/AIFS, GFS, AIGFS);
* ``"files"``: one small compressed file per parameter and step (DWD ICON);
  the whole file is downloaded and decompressed.

Which models exist, their steps and metadata come from
:mod:`stationtruth.registry`; :data:`SOURCES` maps every registry id to its source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from . import registry

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
    layout: str = "index"

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
    def availability_url(self, run: datetime) -> str:
        """A URL that exists once the whole run is published (index of the last step)."""
        return self.index_url(run, self.steps[-1])

    def file_name(self, run: datetime) -> str:
        return f"fcst-{self.name}-{run:%Y%m%d%H}.parquet"


@dataclass(frozen=True)
class EcmwfOpenData(ModelSource):
    name: str = "ecmwf"
    base: str = "https://data.ecmwf.int/forecasts"
    model: str = "ifs"  # path component: "ifs" or "aifs-single"
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
            f"{self.base}/{run:%Y%m%d}/{run:%H}z/{self.model}/0p25/oper/"
            f"{run:%Y%m%d%H}0000-{step}h-oper-fc"
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


@dataclass(frozen=True)
class AigfsAws(GfsAws):
    """NOAA AIGFS (the operational GraphCast-based AI GFS), surface file on AWS.

    ``aigfs.YYYYMMDD/HH/model/atmos/grib2/aigfs.tHHz.sfc.fFFF.grib2`` (+ ``.idx``),
    0.25 deg, 6-hourly steps; no gust, cloud or visibility fields.
    """

    name: str = "aigfs"
    base: str = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
    params: tuple[tuple[str, str, str], ...] = (
        ("t", "TMP", "2 m above ground"),
        ("td", "DPT", "2 m above ground"),
        ("u", "UGRD", "10 m above ground"),
        ("v", "VGRD", "10 m above ground"),
        ("mslp", "PRMSL", "mean sea level"),
    )

    def grib_url(self, run: datetime, step: int) -> str:
        return (
            f"{self.base}/aigfs.{run:%Y%m%d}/{run:%H}/model/atmos/grib2/"
            f"aigfs.t{run:%H}z.sfc.f{step:03d}.grib2"
        )


@dataclass(frozen=True)
class IconOpenData(ModelSource):
    """DWD ICON global on its native icosahedral (R3B7, ~13 km) grid.

    One bz2-compressed GRIB2 file per parameter and step:
    ``{base}/{HH}/{param}/icon_global_icosahedral_single-level_{YYYYMMDDHH}_{SSS}_{PARAM}.grib2.bz2``.
    The server keeps only the latest run of each hour (about 24 h), so there is
    no backfill. Cell-centre coordinates are the time-invariant ``CLAT``/``CLON``
    files of the same run (see :func:`stationtruth.grib.unstructured_weights`).
    """

    name: str = "icon"
    base: str = "https://opendata.dwd.de/weather/nwp/icon/grib"
    layout: str = "files"
    # internal variable -> ICON parameter name
    params: tuple[tuple[str, str], ...] = (
        ("t", "T_2M"),
        ("td", "TD_2M"),
        ("u", "U_10M"),
        ("v", "V_10M"),
        ("gust", "VMAX_10M"),  # maximum over the previous hour
        ("mslp", "PMSL"),
        ("tcc", "CLCT"),
    )

    def file_url(self, run: datetime, step: int, param: str) -> str:
        return (
            f"{self.base}/{run:%H}/{param.lower()}/icon_global_icosahedral_single-level_"
            f"{run:%Y%m%d%H}_{step:03d}_{param}.grib2.bz2"
        )

    def coord_url(self, run: datetime, which: str) -> str:
        """Time-invariant cell-centre latitude (``CLAT``) or longitude (``CLON``)."""
        return (
            f"{self.base}/{run:%H}/{which.lower()}/icon_global_icosahedral_time-invariant_"
            f"{run:%Y%m%d%H}_{which}.grib2.bz2"
        )

    def files(self, step: int) -> list[tuple[str, str]]:
        """(internal variable, ICON parameter) to download for ``step``."""
        # VMAX_10M is a maximum over the hour before the step: none at step 0
        return [(v, p) for v, p in self.params if not (v == "gust" and step == 0)]

    def availability_url(self, run: datetime) -> str:
        return self.file_url(run, self.steps[-1], self.params[-1][1])

    def grib_url(self, run: datetime, step: int) -> str:
        return self.file_url(run, step, self.params[0][1])

    def index_url(self, run: datetime, step: int) -> str:
        return self.file_url(run, step, self.params[0][1])


def _source(info: registry.ModelInfo) -> ModelSource:
    steps = info.steps
    if info.id == "ecmwf":
        return EcmwfOpenData(steps=steps)
    if info.id == "aifs":
        # same server and index format as IFS; AIFS has no 10 m gust field
        return EcmwfOpenData(
            name="aifs",
            model="aifs-single",
            steps=steps,
            params=tuple(p for p in EcmwfOpenData.params if p[0] != "gust"),
        )
    if info.id == "gfs":
        return GfsAws(steps=steps)
    if info.id == "aigfs":
        return AigfsAws(steps=steps)
    if info.id == "icon":
        return IconOpenData(steps=steps)
    raise ValueError(f"no download source for registry model {info.id!r}")  # pragma: no cover


SOURCES: dict[str, ModelSource] = {m.id: _source(m) for m in registry.REGISTRY}
ECMWF = SOURCES["ecmwf"]
GFS = SOURCES["gfs"]
AIFS = SOURCES["aifs"]
ICON = SOURCES["icon"]
AIGFS = SOURCES["aigfs"]


def get_source(name: str) -> ModelSource:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {list(SOURCES)}") from None
