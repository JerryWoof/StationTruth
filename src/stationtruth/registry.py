"""The forecast models StationTruth verifies: single source of truth.

Everything that lists models (download sources in :mod:`stationtruth.models`,
the verified/corrected model names in :mod:`stationtruth.verify`, the
corrections and the blend in :mod:`stationtruth.correct`, the command line)
derives from :data:`REGISTRY`. The site keeps a matching list in
``site/common.js`` (checked by ``tests/test_registry.py``).

Model names on disk and in the site JSON:

* raw model ``<id>`` (``fcst-<id>-YYYYMMDDHH.parquet``),
* its bias-corrected version ``<id>_bc``,
* ``blend``: the mean of all corrected models (see :mod:`stationtruth.correct`).
"""

from __future__ import annotations

from dataclasses import dataclass

#: contract forecast columns a model can provide (docs/DATA.md, ``fcst-<model>``)
FCST_PARAMS: tuple[str, ...] = ("t", "td", "wspd", "wdir", "gust", "mslp", "tcc", "vis_m")

BLEND = "blend"
BC_SUFFIX = "_bc"


@dataclass(frozen=True)
class ModelInfo:
    id: str  # file/JSON name, [a-z0-9]+ (no underscore: "_bc" marks corrected models)
    label: str  # short name for charts
    long: str  # full name
    kind: str  # "physics" (numerical weather prediction) or "ai" (machine-learned)
    provider: str
    resolution: str
    licence: str
    attribution: str  # text the licence requires us to show
    url: str  # dataset home page
    step_h: int  # output interval used (h), steps 0..72
    params: tuple[str, ...]  # contract columns the model provides (others are NaN)
    retention: str  # how long the open-data server keeps a run (limits backfill)

    @property
    def corrected(self) -> str:
        return self.id + BC_SUFFIX

    @property
    def is_ai(self) -> bool:
        return self.kind == "ai"

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(range(0, 73, self.step_h))


_ALL = ("t", "td", "wspd", "wdir", "gust", "mslp", "tcc")

REGISTRY: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="ecmwf",
        label="ECMWF",
        long="ECMWF IFS",
        kind="physics",
        provider="ECMWF",
        resolution="0.25°",
        licence="CC BY 4.0",
        attribution="Copyright © ECMWF. Source: www.ecmwf.int. CC BY 4.0, modified.",
        url="https://www.ecmwf.int/en/forecasts/datasets/open-data",
        step_h=3,
        params=_ALL,
        retention="about 4 days",
    ),
    ModelInfo(
        id="gfs",
        label="GFS",
        long="NOAA GFS",
        kind="physics",
        provider="NOAA/NCEP",
        resolution="0.25°",
        licence="public domain (US government)",
        attribution="NOAA GFS via the NOAA Open Data Dissemination program.",
        url="https://registry.opendata.aws/noaa-gfs-bdp-pds/",
        step_h=3,
        params=(*_ALL, "vis_m"),
        retention="years (AWS archive)",
    ),
    ModelInfo(
        id="icon",
        label="ICON",
        long="DWD ICON",
        kind="physics",
        provider="Deutscher Wetterdienst",
        resolution="13 km icosahedral",
        licence="DWD open-data terms of use (attribution required)",
        attribution="Source: Deutscher Wetterdienst",
        url="https://opendata.dwd.de/weather/nwp/icon/grib/",
        step_h=3,
        params=_ALL,
        retention="about 24 hours (latest run per hour only)",
    ),
    ModelInfo(
        id="aifs",
        label="AIFS",
        long="ECMWF AIFS",
        kind="ai",
        provider="ECMWF",
        resolution="0.25°",
        licence="CC BY 4.0",
        attribution="Copyright © ECMWF. Source: www.ecmwf.int. CC BY 4.0, modified.",
        url="https://www.ecmwf.int/en/forecasts/datasets/open-data",
        step_h=6,
        params=("t", "td", "wspd", "wdir", "mslp", "tcc"),
        retention="about 3 days",
    ),
    ModelInfo(
        id="aigfs",
        label="AIGFS",
        long="NOAA AIGFS",
        kind="ai",
        provider="NOAA/NCEP",
        resolution="0.25°",
        licence="public domain (US government)",
        attribution="NOAA AIGFS via the NOAA Open Data Dissemination program.",
        url="https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/index.html",
        step_h=6,
        params=("t", "td", "wspd", "wdir", "mslp"),
        retention="months (AWS archive)",
    ),
)

RAW_MODELS: tuple[str, ...] = tuple(m.id for m in REGISTRY)
CORRECTED_MODELS: tuple[str, ...] = tuple(m.corrected for m in REGISTRY)
#: every verified model, in display order: raw models, corrected models, blend
MODELS: tuple[str, ...] = (*RAW_MODELS, *CORRECTED_MODELS, BLEND)

_BY_ID = {m.id: m for m in REGISTRY}


def get(model_id: str) -> ModelInfo:
    """Registry entry of a raw model id (``ValueError`` for unknown ids)."""
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise ValueError(f"unknown model {model_id!r}; choose from {list(RAW_MODELS)}") from None


def raw_of(name: str) -> str | None:
    """Raw model behind a model name: ``ecmwf_bc`` -> ``ecmwf``; ``blend`` -> None."""
    base = name[: -len(BC_SUFFIX)] if name.endswith(BC_SUFFIX) else name
    return base if base in _BY_ID else None


def kind_of(name: str) -> str | None:
    """``physics`` / ``ai`` for raw and corrected models, None for the blend."""
    raw = raw_of(name)
    return None if raw is None else _BY_ID[raw].kind


def parse_list(arg: str | None) -> list[str]:
    """``"all"`` (or None) -> every raw model; empty -> none; else a validated comma list."""
    if arg is None or arg.strip() == "all":
        return list(RAW_MODELS)
    names = [s.strip() for s in arg.split(",") if s.strip()]
    for n in names:
        get(n)
    return names
