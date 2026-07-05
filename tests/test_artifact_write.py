from __future__ import annotations

from pathlib import Path

import pytest

from krea_2_turbo_mlx.artifact_write import (
    atomic_output_dir,
    ensure_empty_or_missing_output_dir,
    preflight_free_space,
)
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.json_io import read_json_object, write_json


def test_atomic_output_dir_commits_complete_directory(tmp_path: Path) -> None:
    output = tmp_path / "artifact"

    with atomic_output_dir(output, label="test") as temp:
        (temp / "artifact.json").write_text("{}", encoding="utf-8")

    assert output.is_dir()
    assert (output / "artifact.json").exists()
    assert not any(path.name.startswith(".artifact.tmp-") for path in tmp_path.iterdir())


def test_atomic_output_dir_cleans_failed_attempt(tmp_path: Path) -> None:
    output = tmp_path / "artifact"

    with pytest.raises(RuntimeError):
        with atomic_output_dir(output, label="test") as temp:
            (temp / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")

    assert not output.exists()
    assert not any(path.name.startswith(".artifact.tmp-") for path in tmp_path.iterdir())


def test_existing_nonempty_output_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "existing.txt").write_text("x", encoding="utf-8")

    with pytest.raises(Krea2TurboMlxError, match="must be empty"):
        ensure_empty_or_missing_output_dir(output, label="test")


def test_preflight_free_space_reports_impossible_requirement(tmp_path: Path) -> None:
    output = tmp_path / "artifact"

    with pytest.raises(Krea2TurboMlxError, match="Not enough free disk space"):
        preflight_free_space(output, required_bytes=10**30, label="test")


def test_json_writer_round_trips_object_atomically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json(path, {"b": 2, "a": 1})

    assert read_json_object(path) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").startswith('{\n  "a": 1')

