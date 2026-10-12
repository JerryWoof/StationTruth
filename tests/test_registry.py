from __future__ import annotations

import re
from pathlib import Path

import pytest

from stationtruth import models, registry, storage, verify

SITE = Path(__file__).resolve().parents[1] / "site"


def test_ids_and_model_lists():
    ids = [m.id for m in registry.REGISTRY]
    assert len(set(ids)) == len(ids)
    for m in registry.REGISTRY:
        assert re.fullmatch(r"[a-z0-9]+", m.id), m.id  # "_bc" is reserved for corrections
        assert m.kind in ("physics", "ai")
        assert m.step_h in (1, 3, 6) and m.steps[0] == 0 and m.steps[-1] == 72
        assert set(m.params) <= set(registry.FCST_PARAMS)
        assert {"t", "td", "wspd", "wdir", "mslp"} <= set(m.params)
        assert m.licence and m.attribution and m.url.startswith("https://")
    assert registry.MODELS == (
        *registry.RAW_MODELS,
        *(f"{i}_bc" for i in registry.RAW_MODELS),
        "blend",
    )
    assert verify.RAW_MODELS == registry.RAW_MODELS and verify.MODELS == registry.MODELS
    assert {"ecmwf", "gfs", "aifs", "icon"} <= set(registry.RAW_MODELS)
    assert {m.kind for m in registry.REGISTRY} == {"physics", "ai"}


def test_lookup_helpers():
    assert registry.get("aifs").is_ai and not registry.get("ecmwf").is_ai
    assert registry.get("icon").corrected == "icon_bc"
    assert registry.raw_of("aifs_bc") == "aifs" and registry.raw_of("gfs") == "gfs"
    assert registry.raw_of("blend") is None and registry.raw_of("nope_bc") is None
    assert registry.kind_of("aifs_bc") == "ai" and registry.kind_of("blend") is None
    assert registry.parse_list("all") == list(registry.RAW_MODELS)
    assert registry.parse_list(None) == list(registry.RAW_MODELS)
    assert registry.parse_list("") == []
    assert registry.parse_list(" gfs, aifs ") == ["gfs", "aifs"]
    with pytest.raises(ValueError, match="unknown model"):
        registry.parse_list("ecmwf,nam")


def test_file_names_accepted_everywhere(tmp_path):
    from datetime import datetime, timezone

    run = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    for mid, src in models.SOURCES.items():
        name = src.file_name(run)
        assert name == f"fcst-{mid}-2026101212.parquet"
        info = storage.parse_name(name)
        assert info is not None and info["model"] == mid
        (tmp_path / name).write_bytes(b"x")
    _, fcst = verify.list_files(tmp_path)
    assert sorted(m for m, _, _ in fcst) == sorted(registry.RAW_MODELS)


def test_site_model_list_matches_registry():
    """site/common.js keeps its own copy of the registry (labels, colours)."""
    js = (SITE / "common.js").read_text()
    block = js[js.index("const REGISTRY = [") : js.index("];", js.index("const REGISTRY = ["))]
    ids = re.findall(r"\bid: '([a-z0-9]+)'", block)
    assert ids == list(registry.RAW_MODELS)
    for m in registry.REGISTRY:
        entry = block[block.index(f"id: '{m.id}'") :].split("\n")[0]
        assert f"kind: '{m.kind}'" in entry, m.id
