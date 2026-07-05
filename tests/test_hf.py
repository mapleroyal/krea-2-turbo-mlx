from __future__ import annotations

from pathlib import Path

import pytest

from krea_2_turbo_mlx import hf
from krea_2_turbo_mlx.constants import OFFICIAL_HF_REPO_ID, OFFICIAL_HF_REVISION
from krea_2_turbo_mlx.errors import Krea2TurboMlxError


def test_download_source_uses_pinned_revision_allow_patterns_and_local_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    destination = tmp_path / "source"

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.update(kwargs)
        destination.mkdir()
        return str(destination)

    monkeypatch.setattr(hf, "_load_snapshot_download", lambda: fake_snapshot_download)

    path = hf.download_source(
        OFFICIAL_HF_REPO_ID,
        local_dir=destination,
    )

    assert path == destination
    assert calls["repo_id"] == OFFICIAL_HF_REPO_ID
    assert calls["revision"] == OFFICIAL_HF_REVISION
    assert calls["local_dir"] == destination
    assert "transformer/**" in calls["allow_patterns"]
    assert calls["ignore_patterns"] == ["turbo.safetensors"]


def test_download_source_returns_existing_local_path_without_hf_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    monkeypatch.setattr(
        hf,
        "_load_snapshot_download",
        lambda: pytest.fail("snapshot_download should not be loaded"),
    )

    assert hf.download_source(source) == source


def test_download_source_requires_revision_for_custom_remote() -> None:
    with pytest.raises(Krea2TurboMlxError, match="requires --revision"):
        hf.download_source("someone/model")
