from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from stationtruth import http, storage

UTC = timezone.utc


def test_month_of_and_parse_name():
    assert storage.month_of("obs-2026-10-12.parquet") == "2026-10"
    assert storage.month_of("fcst-ecmwf-2026093012.parquet") == "2026-09"
    assert storage.month_of("stations.parquet", today=date(2026, 11, 2)) == "2026-11"
    with pytest.raises(ValueError):
        storage.month_of("whatever.txt")
    assert storage.parse_name("fcst-gfs-2026101200.parquet") == {
        "kind": "fcst",
        "model": "gfs",
        "run": datetime(2026, 10, 12, tzinfo=UTC),
    }
    assert storage.months_between(date(2026, 11, 20), date(2027, 2, 1)) == [
        "2026-11",
        "2026-12",
        "2027-01",
        "2027-02",
    ]


def test_names_for_window():
    names = [
        "obs-2026-10-01.parquet",
        "obs-2026-10-05.parquet",
        "obs-2026-10-12.parquet",
        "fcst-ecmwf-2026100112.parquet",
        "fcst-ecmwf-2026100300.parquet",
        "fcst-gfs-2026101200.parquet",
        "stations.parquet",
        "README.txt",
    ]
    got = storage.names_for_window(
        names, datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 12, 16, tzinfo=UTC)
    )
    assert got == [
        "fcst-ecmwf-2026100300.parquet",
        "fcst-gfs-2026101200.parquet",
        "obs-2026-10-05.parquet",
        "obs-2026-10-12.parquet",
        "stations.parquet",
    ]


def test_local_storage_roundtrip_and_sync(tmp_path):
    root = tmp_path / "store"
    st = storage.LocalStorage(root)
    src = tmp_path / "x.parquet"
    src.write_bytes(b"abc")
    st.put(src, "obs-2026-10-12.parquet")
    st.put(src, "stations.parquet")
    st.put(src, "fcst-gfs-2026090100.parquet")
    with pytest.raises(ValueError):
        st.put(src, "bad.parquet")
    assert st.exists("obs-2026-10-12.parquet")
    assert set(st.list(["2026-10"])) == {"obs-2026-10-12.parquet", "stations.parquet"}
    out = tmp_path / "out"
    assert st.get("missing-2026.parquet", out) is None
    assert st.get("obs-2026-10-12.parquet", out).read_bytes() == b"abc"

    dest = tmp_path / "synced"
    dest.mkdir()
    (dest / "obs-2026-01-01.parquet").write_bytes(b"old")
    files = st.sync(
        dest,
        datetime(2026, 10, 1, tzinfo=UTC),
        datetime(2026, 10, 12, 12, tzinfo=UTC),
        prune=True,
        log=None,
    )
    assert sorted(p.name for p in files) == ["obs-2026-10-12.parquet", "stations.parquet"]
    assert not (dest / "obs-2026-01-01.parquet").exists()  # pruned
    manifest = json.loads((dest / ".sync.json").read_text())
    assert set(manifest) == {"obs-2026-10-12.parquet", "stations.parquet"}
    # second sync downloads nothing
    calls = []
    orig = st.get
    st.get = lambda name, d: calls.append(name) or orig(name, d)
    st.sync(dest, datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC), log=None)
    assert calls == []


def test_sync_never_prunes_its_own_source(tmp_path):
    st = storage.LocalStorage(tmp_path)
    (tmp_path / "obs-2020-01-01.parquet").write_bytes(b"x")
    st.sync(
        tmp_path,
        datetime(2026, 10, 1, tzinfo=UTC),
        datetime(2026, 10, 2, tzinfo=UTC),
        prune=True,
        log=None,
    )
    assert (tmp_path / "obs-2020-01-01.parquet").exists()


class FakeGitHub:
    """Minimal in-memory GitHub releases API."""

    def __init__(self):
        self.releases = {}  # tag -> {"id", "tag_name", "assets": {name: asset}}
        self.blobs = {}  # asset id -> bytes
        self.next_id = 100
        self.calls = []

    def _id(self):
        self.next_id += 1
        return self.next_id

    def __call__(self, url, method="GET", headers=None, data=None, opener=None, **kw):
        self.calls.append((method, url))
        headers = headers or {}
        path = url.split("github.com", 1)[1]
        if method == "GET" and "/releases/tags/" in path:
            tag = path.rsplit("/", 1)[1]
            rel = self.releases.get(tag)
            if rel is None:
                raise http.HTTPStatusError(url, 404)
            return 200, {}, json.dumps({"id": rel["id"], "tag_name": tag}).encode()
        if method == "POST" and path.endswith("/releases"):
            body = json.loads(data)
            assert body["prerelease"] is True
            rel = {"id": self._id(), "tag_name": body["tag_name"], "assets": {}}
            self.releases[body["tag_name"]] = rel
            return 201, {}, json.dumps({"id": rel["id"], "tag_name": rel["tag_name"]}).encode()
        if method == "GET" and "/assets?per_page" in path:
            rid = int(path.split("/releases/")[1].split("/")[0])
            rel = next(r for r in self.releases.values() if r["id"] == rid)
            return 200, {}, json.dumps(list(rel["assets"].values())).encode()
        if method == "POST" and "uploads.github.com" in url:
            assert headers.get("Authorization") == "Bearer tok"
            rid = int(path.split("/releases/")[1].split("/")[0])
            name = url.split("name=", 1)[1]
            rel = next(r for r in self.releases.values() if r["id"] == rid)
            if name in rel["assets"]:
                raise http.HTTPStatusError(url, 422)
            a = {"id": self._id(), "name": name, "size": len(data), "updated_at": str(self.next_id)}
            rel["assets"][name] = a
            self.blobs[a["id"]] = data
            return 201, {}, json.dumps(a).encode()
        if method == "DELETE" and "/releases/assets/" in path:
            aid = int(path.rsplit("/", 1)[1])
            for rel in self.releases.values():
                for n, a in list(rel["assets"].items()):
                    if a["id"] == aid:
                        del rel["assets"][n]
            return 204, {}, b""
        if method == "GET" and "/releases/assets/" in path:
            aid = int(path.rsplit("/", 1)[1])
            assert opener is not None  # redirects must not carry the token
            return 302, {"Location": f"https://objects.example/{aid}"}, b""
        raise AssertionError(f"unexpected {method} {url}")


@pytest.fixture
def fake_github(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(storage.http, "request", fake)

    def fake_get(url, **kw):
        assert url.startswith("https://objects.example/")
        return fake.blobs[int(url.rsplit("/", 1)[1])]

    monkeypatch.setattr(storage.http, "get", fake_get)
    return fake


def test_github_storage_put_replace_get(fake_github, tmp_path):
    gh = storage.GitHubStorage(repo="o/r", token="tok", log=None)
    f = tmp_path / "obs-2026-10-12.parquet"
    f.write_bytes(b"v1")
    gh.put(f)
    assert "data-2026-10" in fake_github.releases
    f.write_bytes(b"v2-longer")
    gh.put(f)
    assets = fake_github.releases["data-2026-10"]["assets"]
    assert (
        list(assets) == ["obs-2026-10-12.parquet"] and assets["obs-2026-10-12.parquet"]["size"] == 9
    )

    gh2 = storage.GitHubStorage(repo="o/r", token=None, log=None)  # fresh caches, read-only
    assert gh2.exists("obs-2026-10-12.parquet")
    assert not gh2.exists("obs-2026-09-12.parquet")  # release missing -> nothing
    got = gh2.get("obs-2026-10-12.parquet", tmp_path / "dl")
    assert got.read_bytes() == b"v2-longer"
    with pytest.raises(RuntimeError):
        gh2.put(f)


def test_github_storage_put_recovers_from_422(fake_github, tmp_path):
    gh = storage.GitHubStorage(repo="o/r", token="tok", log=None)
    f = tmp_path / "fcst-gfs-2026101200.parquet"
    f.write_bytes(b"one")
    gh.put(f)
    # Another writer replaced the asset behind our back: our cache is stale.
    rel = fake_github.releases["data-2026-10"]
    rel["assets"]["fcst-gfs-2026101200.parquet"]["id"] = 999
    gh._assets["2026-10"].pop("fcst-gfs-2026101200.parquet")
    f.write_bytes(b"two")
    gh.put(f)
    assert (
        fake_github.releases["data-2026-10"]["assets"]["fcst-gfs-2026101200.parquet"]["size"] == 3
    )


def test_github_stations_from_previous_month(fake_github, tmp_path, monkeypatch):
    gh = storage.GitHubStorage(repo="o/r", token="tok", log=None)
    f = tmp_path / "stations.parquet"
    f.write_bytes(b"st")
    gh.put(f)  # current month
    this_month = datetime.now(UTC).strftime("%Y-%m")
    assert this_month in "".join(fake_github.releases)
    got = storage.GitHubStorage(repo="o/r", token="tok", log=None).get(
        "stations.parquet", tmp_path / "d"
    )
    assert got.read_bytes() == b"st"


def test_open_storage(tmp_path):
    assert isinstance(storage.open_storage(tmp_path), storage.LocalStorage)
    assert isinstance(storage.open_storage(None), storage.GitHubStorage)
