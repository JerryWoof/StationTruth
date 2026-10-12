"""Country -> verification region mapping.

Regions (docs/DATA.md): Europe, North America, Central America & Caribbean,
South America, Africa, Middle East, Asia, Oceania, Polar.

Rules, applied in order:

1. ``|lat| > 66.5`` -> Polar (polar circles), whatever the country.
2. Countries spanning two regions are split by coordinates:
   Russia (lon < 60 E -> Europe, else Asia), US (Hawaii and other Pacific
   points west of 140 W and south of 30 N -> Oceania).
3. ISO 3166-1 alpha-2 lookup (all assigned codes, plus XK for Kosovo).
4. Unknown/empty country -> coarse lat/lon boxes.

Debatable choices, kept deliberately simple: Turkey is
Middle East (most of its airports are in Anatolia), Cyprus is Europe; Egypt
and Sudan are Africa; the Caucasus states are Asia; Mexico and Bermuda are
North America; overseas territories follow geography (e.g. Reunion -> Africa,
New Caledonia -> Oceania, French Guiana -> South America).
"""

from __future__ import annotations

import numpy as np

REGIONS: tuple[str, ...] = (
    "Europe",
    "North America",
    "Central America & Caribbean",
    "South America",
    "Africa",
    "Middle East",
    "Asia",
    "Oceania",
    "Polar",
)

POLAR_LAT = 66.5

_GROUPS: dict[str, str] = {
    "Europe": (
        "AD AL AT AX BA BE BG BY CH CY CZ DE DK EE ES FI FO FR GB GG GI GR HR HU IE IM IS IT "
        "JE LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SJ SK SM UA VA XK"
    ),
    "North America": "BM CA GL MX PM US",
    "Central America & Caribbean": (
        "AG AI AW BB BL BQ BS BZ CR CU CW DM DO GD GP GT HN HT JM KN KY LC MF MQ MS NI PA PR "
        "SV SX TC TT VC VG VI"
    ),
    "South America": "AR BO BR CL CO EC FK GF GS GY PE PY SR UY VE",
    "Africa": (
        "AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ GW KE KM LR LS LY "
        "MA MG ML MR MU MW MZ NA NE NG RE RW SC SD SH SL SN SO SS ST SZ TD TF TG TN TZ UG YT "
        "ZA ZM ZW"
    ),
    "Middle East": "AE BH IL IQ IR JO KW LB OM PS QA SA SY TR YE",
    "Asia": (
        "AF AM AZ BD BN BT CC CN CX GE HK ID IN IO JP KG KH KP KR KZ LA LK MM MN MO MV MY NP "
        "PH PK SG TH TJ TL TM TW UZ VN"
    ),
    "Oceania": "AS AU CK FJ FM GU HM KI MH MP NC NF NR NU NZ PF PG PN PW SB TK TO TV UM VU WF WS",
    "Polar": "AQ BV",
}

COUNTRY_REGION: dict[str, str] = {
    code: region for region, codes in _GROUPS.items() for code in codes.split()
}


def _by_coords(lat: float, lon: float) -> str | None:
    """Very coarse fallback for stations without a usable country code."""
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    if lat >= 35 and -25 <= lon < 40:
        return "Europe"
    if lat >= 35 and 40 <= lon < 60:
        return "Europe" if lat >= 45 else "Middle East"
    if 12 <= lat < 42 and 34 <= lon < 63:
        return "Middle East"
    if -40 <= lat < 37 and -20 <= lon < 52:
        return "Africa"
    if lat >= 0 and 60 <= lon < 150 or (lat >= 45 and lon >= 150):
        return "Asia"
    if -11 <= lat < 0 and 95 <= lon < 141:
        return "Asia"
    if lat < 0 and lon >= 110 or (lat < 30 and (lon >= 150 or lon < -140)):
        return "Oceania"
    if lat >= 24 and -170 <= lon < -50:
        return "North America"
    if 7 <= lat < 24 and -120 <= lon < -58:
        return "Central America & Caribbean"
    if lat < 13 and -92 <= lon < -30:
        return "South America"
    return None


def region_for(country: str | None, lat: float, lon: float) -> str | None:
    """Region of one station, or None if it cannot be determined."""
    lat = float(lat) if lat is not None else float("nan")
    lon = float(lon) if lon is not None else float("nan")
    if np.isfinite(lat) and abs(lat) > POLAR_LAT:
        return "Polar"
    cc = (country or "").strip().upper()
    if cc == "RU" and np.isfinite(lon):
        return "Europe" if -30 <= lon < 60 else "Asia"
    if cc == "US" and np.isfinite(lat) and np.isfinite(lon) and lat < 30 and lon < -140:
        return "Oceania"
    if cc in COUNTRY_REGION:
        return COUNTRY_REGION[cc]
    return _by_coords(lat, lon)


def region_index(countries, lats, lons) -> np.ndarray:
    """Vectorised helper: int8 index into REGIONS per station, -1 if unknown."""
    lookup = {r: i for i, r in enumerate(REGIONS)}
    out = np.full(len(lats), -1, dtype=np.int8)
    for i, (c, la, lo) in enumerate(zip(countries, lats, lons, strict=True)):
        r = region_for(c if isinstance(c, str) else None, la, lo)
        if r is not None:
            out[i] = lookup[r]
    return out
