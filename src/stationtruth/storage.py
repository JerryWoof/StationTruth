"""Data store: GitHub Release assets (production) or a plain directory (local/tests).

Both backends hold files with the exact contract names (docs/DATA.md). On
GitHub, each file lives in the monthly prerelease ``data-YYYY-MM`` given by
the date in its name; ``stations.parquet`` is written to the current month
and read from the newest month that has it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import http

DEFAULT_REPO = "JerryWoof/StationTruth"
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
STATIONS_FILE = "stations.parquet"

RELEASE_BODY = (
    "Data store for StationTruth (not a software release).\n\n"
    "Assets are written automatically by GitHub Actions: hourly METAR observations "
    "(`obs-YYYY-MM-DD.parquet`), ECMWF and GFS runs interpolated to stations "
    "(`fcst-<model>-YYYYMMDDHH.parquet`) and the station list (`stations.parquet`). "
    "See docs/DATA.md for the file formats."
)

_OBS_RE = re.compile(r"^obs-(\d{4})-(\d{2})-(\d{2})\.parquet$")
_FCST_RE = re.compile(r"^fcst-([a-z0-9_]+)-(\d{4})(\d{2})(\d{2})(\d{2})\.parquet$")


def month_of(name: str, today: date | None = None) -> str:
    """Release month ``YYYY-MM`` for an asset name."""
    m = _OBS_RE.match(name)
    if m:
        return f"{m[1]}-{m[2]}"
    m = _FCST_RE.match(name)
    if m:
        return f"{m[2]}-{m[3]}"
    if name == STATIONS_FILE:
        today = today or datetime.now(timezone.utc).date()
        return f"{today:%Y-%m}"
    raise ValueError(f"not a StationTruth data file name: {name!r}")


def release_tag(month: str) -> str:
    return f"data-{month}"


def months_between(start: date, end: date) -> list[str]:
    """``YYYY-MM`` strings from ``start``'s month to ``end``'s month, inclusive."""
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def parse_name(name: str) -> dict | None:
    """``{"kind": "obs", "date": date}`` / ``{"kind": "fcst", "model", "run"}`` / stations."""
    m = _OBS_RE.match(name)
    if m:
        return {"kind": "obs", "date": date(int(m[1]), int(m[2]), int(m[3]))}
    m = _FCST_RE.match(name)
    if m:
        run = datetime(int(m[2]), int(m[3]), int(m[4]), int(m[5]), tzinfo=timezone.utc)
        return {"kind": "fcst", "model": m[1], "run": run}
    if name == STATIONS_FILE:
        return {"kind": "stations"}
    return None


def names_for_window(
    available: Iterable[str], start: datetime, end: datetime, fcst_margin_h: int = 72
) -> list[str]:
    """Asset names needed to work on ``[start, end]``.

    Observations for each day in the window, forecast runs whose valid times
    can fall in the window (initialised up to ``fcst_margin_h`` before the
    start), and the station list.
    """
    out = []
    first_day, last_day = start.date(), end.date()
    run_start = start - timedelta(hours=fcst_margin_h)
    for name in sorted(set(available)):
        info = parse_name(name)
        if info is None:
            continue
        if info["kind"] == "obs" and first_day <= info["date"] <= last_day:
            out.append(name)
        elif info["kind"] == "fcst" and run_start <= info["run"] <= end:
            out.append(name)
        elif info["kind"] == "stations":
            out.append(name)
    return out


class Storage:
    """Common interface of the backends."""

    def _is_source_dir(self, path: Path) -> bool:
        return False

    def list(self, months: Iterable[str] | None = None) -> dict[str, dict]:
        """``{name: {"size": int, "updated": str}}`` for the given months (None = all)."""
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        return name in self.list([month_of(name)])

    def get(self, name: str, dest_dir: Path) -> Path | None:
        """Copy/download ``name`` into ``dest_dir``; ``None`` if absent."""
        raise NotImplementedError

    def put(self, path: Path, name: str | None = None) -> None:
        """Store ``path`` as ``name`` (default: its file name), replacing any existing copy."""
        raise NotImplementedError

    def sync(
        self, dest_dir: Path, start: datetime, end: datetime, *, prune: bool = False, log=print
    ) -> list[Path]:
        """Download everything needed for ``[start, end]`` into ``dest_dir``.

        Files already present with the same size and update stamp (tracked in
        ``dest_dir/.sync.json``) are not downloaded again. With ``prune``, data
        files outside the window are deleted from ``dest_dir``.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        months = months_between((start - timedelta(days=4)).date(), end.date())
        listing = self.list(months)
        wanted = names_for_window(listing, start, end)
        manifest_path = dest_dir / ".sync.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            manifest = {}
        out, fetched, fetched_bytes = [], 0, 0
        for name in wanted:
            info = listing[name]
            local = dest_dir / name
            stamp = f"{info.get('size')}|{info.get('updated')}"
            if local.exists() and manifest.get(name) == stamp:
                out.append(local)
                continue
            path = self.get(name, dest_dir)
            if path is not None:
                manifest[name] = stamp
                out.append(path)
                fetched += 1
                fetched_bytes += path.stat().st_size
        if prune and not self._is_source_dir(dest_dir):
            # Drop files that fell out of the window so a cached data dir stays small.
            keep = set(wanted)
            for p in dest_dir.glob("*.parquet"):
                if p.name not in keep and parse_name(p.name) is not None:
                    p.unlink()
                    manifest.pop(p.name, None)
        manifest_path.write_text(json.dumps(manifest, indent=0, sort_keys=True))
        if log:
            log(
                f"sync: {len(wanted)} files in window, downloaded {fetched} "
                f"({fetched_bytes / 1e6:.1f} MB), {len(wanted) - fetched} up to date"
            )
        return out


@dataclass
class LocalStorage(Storage):
    """A directory with contract-named files."""

    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self, months: Iterable[str] | None = None) -> dict[str, dict]:
        wanted = None if months is None else set(months)
        out = {}
        for p in self.root.iterdir():
            info = parse_name(p.name)
            if info is None or not p.is_file():
                continue
            if wanted is not None and info["kind"] != "stations":
                if month_of(p.name) not in wanted:
                    continue
            st = p.stat()
            out[p.name] = {"size": st.st_size, "updated": str(st.st_mtime_ns)}
        return out

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def _is_source_dir(self, path: Path) -> bool:
        return Path(path).resolve() == self.root.resolve()

    def get(self, name: str, dest_dir: Path) -> Path | None:
        src = self.root / name
        if not src.is_file():
            return None
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        return dest

    def put(self, path: Path, name: str | None = None) -> None:
        name = name or Path(path).name
        month_of(name)  # validate
        dest = self.root / name
        if Path(path).resolve() != dest.resolve():
            tmp = dest.with_name(dest.name + ".tmp")
            shutil.copy2(path, tmp)
            os.replace(tmp, dest)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class GitHubStorage(Storage):
    """Monthly prereleases ``data-YYYY-MM`` in a GitHub repository."""

    def __init__(self, repo: str | None = None, token: str | None = None, log=print):
        self.repo = repo or os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.log = log
        self._releases: dict[str, dict | None] = {}
        self._assets: dict[str, dict[str, dict]] = {}
        self._no_redirect = urllib.request.build_opener(_NoRedirect)

    # -- low level
    def _headers(self, extra: dict | None = None) -> dict:
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if extra:
            h.update(extra)
        return h

    def _api(self, method: str, url: str, payload=None, headers=None, data=None):
        if not url.startswith("http"):
            url = API + url
        body = data
        hdrs = self._headers(headers)
        if payload is not None:
            body = json.dumps(payload).encode()
            hdrs["Content-Type"] = "application/json"
        status, _, raw = http.request(url, method=method, headers=hdrs, data=body, timeout=300)
        if not raw:
            return None
        return json.loads(raw)

    def _require_token(self):
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required to write release assets")

    # -- releases
    def release(self, month: str, create: bool = False) -> dict | None:
        tag = release_tag(month)
        rel = self._releases.get(tag)
        if rel is None:
            try:
                rel = self._api("GET", f"/repos/{self.repo}/releases/tags/{tag}")
            except http.HTTPStatusError as exc:
                if exc.status != 404:
                    raise
                rel = None
            if rel is None and create:
                self._require_token()
                if self.log:
                    self.log(f"creating release {tag}")
                rel = self._api(
                    "POST",
                    f"/repos/{self.repo}/releases",
                    {
                        "tag_name": tag,
                        "name": f"Data {month}",
                        "body": RELEASE_BODY,
                        "prerelease": True,
                        "draft": False,
                    },
                )
            self._releases[tag] = rel
        return rel

    def _release_assets(self, month: str) -> dict[str, dict]:
        if month in self._assets:
            return self._assets[month]
        rel = self.release(month)
        assets: dict[str, dict] = {}
        if rel is not None:
            page = 1
            while True:
                batch = self._api(
                    "GET",
                    f"/repos/{self.repo}/releases/{rel['id']}/assets?per_page=100&page={page}",
                )
                for a in batch or []:
                    assets[a["name"]] = a
                if not batch or len(batch) < 100:
                    break
                page += 1
        self._assets[month] = assets
        return assets

    # -- interface
    def list(self, months: Iterable[str] | None = None) -> dict[str, dict]:
        if months is None:
            rels = self._api("GET", f"/repos/{self.repo}/releases?per_page=100") or []
            months = [r["tag_name"][5:] for r in rels if r["tag_name"].startswith("data-")]
        out: dict[str, dict] = {}
        # Oldest first, so a newer month's stations.parquet wins.
        for month in sorted(set(months)):
            for name, a in self._release_assets(month).items():
                if parse_name(name) is None:
                    continue
                out[name] = {
                    "size": a.get("size"),
                    "updated": a.get("updated_at"),
                    "month": month,
                    "id": a["id"],
                }
        return out

    def exists(self, name: str) -> bool:
        return name in self._release_assets(month_of(name))

    def _find_asset(self, name: str) -> dict | None:
        if name != STATIONS_FILE:
            return self._release_assets(month_of(name)).get(name)
        today = datetime.now(timezone.utc).date().replace(day=1)
        for back in range(4):
            first = today
            for _ in range(back):
                first = (first - timedelta(days=1)).replace(day=1)
            a = self._release_assets(f"{first:%Y-%m}").get(name)
            if a is not None:
                return a
        return None

    def download_asset(self, asset: dict, dest: Path) -> Path:
        url = f"{API}/repos/{self.repo}/releases/assets/{asset['id']}"
        status, headers, body = http.request(
            url,
            headers=self._headers({"Accept": "application/octet-stream"}),
            opener=self._no_redirect,
            timeout=300,
        )
        if status in (301, 302, 303, 307, 308):
            location = headers.get("Location") or headers.get("location")
            # The signed storage URL must be fetched without our Authorization header.
            body = http.get(location, timeout=300)
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, dest)
        return dest

    def get(self, name: str, dest_dir: Path) -> Path | None:
        asset = self._find_asset(name)
        if asset is None:
            return None
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return self.download_asset(asset, dest_dir / name)

    def put(self, path: Path, name: str | None = None) -> None:
        self._require_token()
        path = Path(path)
        name = name or path.name
        month = month_of(name)
        rel = self.release(month, create=True)
        assets = self._release_assets(month)
        old = assets.get(name)
        if old is not None:
            self._api("DELETE", f"/repos/{self.repo}/releases/assets/{old['id']}")
            assets.pop(name, None)
        url = f"{UPLOADS}/repos/{self.repo}/releases/{rel['id']}/assets?" + urllib.parse.urlencode(
            {"name": name}
        )
        data = path.read_bytes()
        headers = {"Content-Type": "application/octet-stream"}
        try:
            new = self._api("POST", url, headers=headers, data=data)
        except http.HTTPStatusError as exc:
            if exc.status != 422:
                raise
            # "already_exists": an earlier attempt landed (e.g. retried after a 502)
            # or a concurrent writer won. Replace it once more.
            self._assets.pop(month, None)
            stale = self._release_assets(month).get(name)
            if stale is not None:
                self._api("DELETE", f"/repos/{self.repo}/releases/assets/{stale['id']}")
            new = self._api("POST", url, headers=headers, data=data)
            assets = self._release_assets(month)
        if new:
            assets[name] = new
        if self.log:
            self.log(
                f"uploaded {name} ({path.stat().st_size / 1e6:.2f} MB) to {release_tag(month)}"
            )


def open_storage(data_dir: str | Path | None, log=print) -> Storage:
    """``LocalStorage(data_dir)`` when a directory is given, else GitHub Releases."""
    if data_dir:
        return LocalStorage(Path(data_dir))
    return GitHubStorage(log=log)
