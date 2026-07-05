from __future__ import annotations

import json
import os
import subprocess
from io import StringIO
from pathlib import Path

from krea_2_turbo_mlx import cli
from krea_2_turbo_mlx.setup_flow import SetupConfig, _write_gui_launcher, run_setup_cli


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_setup_dry_run_prints_plan_without_writing_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.json"
    args = cli.build_parser().parse_args(
        ["setup", "--config", str(config_path), "--accept-defaults", "--dry-run"]
    )
    stream = StringIO()

    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow._environment_errors", lambda: [])

    assert run_setup_cli(args, stream=stream) == 0
    assert not config_path.exists()
    assert "Artifact: full-precision" in stream.getvalue()
    assert "[dry-run]" in stream.getvalue()


def test_setup_reuses_existing_artifact_and_still_runs_doctor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "artifact.json").write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "setup",
            "--config",
            str(tmp_path / "config.json"),
            "--accept-defaults",
            "--source-dir",
            str(tmp_path / "missing-source"),
            "--artifact-dir",
            str(artifact),
        ]
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow._environment_errors", lambda: [])
    monkeypatch.setattr(
        "krea_2_turbo_mlx.setup_flow.download_source",
        lambda *args, **kwargs: calls.setdefault("download", True),
    )
    monkeypatch.setattr(
        "krea_2_turbo_mlx.setup_flow.run_conversion",
        lambda *args, **kwargs: calls.setdefault("convert", True),
    )
    def fake_doctor(*, model: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"runtime_required": True}
        calls["doctor"] = model
        return {"status": "ok", "error_count": 0, "warning_count": 0, "checks": []}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow.run_doctor", fake_doctor)

    assert run_setup_cli(args, stream=StringIO()) == 0
    assert "download" not in calls
    assert "convert" not in calls
    assert calls["doctor"] == artifact


def test_setup_downloads_converts_validates_and_writes_launcher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "models" / "Krea-2-Turbo"
    artifact = tmp_path / "artifact"
    args = cli.build_parser().parse_args(
        [
            "setup",
            "--config",
            str(tmp_path / "config.json"),
            "--accept-defaults",
            "--source-dir",
            str(source),
            "--artifact-dir",
            str(artifact),
        ]
    )
    calls: list[str] = []

    def fake_download(*args: object, **kwargs: object) -> Path:
        calls.append("download")
        source.mkdir(parents=True)
        (source / "model_index.json").write_text("{}", encoding="utf-8")
        return source

    def fake_convert(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append("convert")
        artifact.mkdir()
        (artifact / "artifact.json").write_text("{}", encoding="utf-8")
        return {"status": "artifact_written"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow._environment_errors", lambda: [])
    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow.download_source", fake_download)
    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow.run_conversion", fake_convert)
    monkeypatch.setattr(
        "krea_2_turbo_mlx.setup_flow.run_doctor",
        lambda *, model, **kwargs: calls.append("doctor")
        or {"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
    )

    assert run_setup_cli(args, stream=StringIO()) == 0
    assert calls == ["download", "convert", "doctor"]
    assert (tmp_path / "Launch Krea 2 Turbo.command").is_file()


def test_gui_launcher_delegates_to_launch_script(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / ".krea-2-turbo-mlx" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text("{}", encoding="utf-8")

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    launch_log = tmp_path / "launch.log"
    _write_executable(
        scripts_dir / "launch.sh",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "$LAUNCH_LOG"
""",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAUNCH_LOG", str(launch_log))

    launcher = _write_gui_launcher(config_path)

    subprocess.run([str(launcher), "--open"], cwd=tmp_path, check=True)
    assert launch_log.read_text(encoding="utf-8").splitlines() == [
        f"--config {config_path} --open"
    ]


def test_setup_cleanup_source_removes_only_project_local_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "models" / "Krea-2-Turbo"
    source.mkdir(parents=True)
    (source / "model_index.json").write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact"
    args = cli.build_parser().parse_args(
        [
            "setup",
            "--config",
            "config.json",
            "--accept-defaults",
            "--cleanup-source",
            "--source-dir",
            "models/Krea-2-Turbo",
            "--artifact-dir",
            "artifact",
        ]
    )

    def fake_convert(*args: object, **kwargs: object) -> dict[str, object]:
        artifact.mkdir()
        (artifact / "artifact.json").write_text("{}", encoding="utf-8")
        return {"status": "artifact_written"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow._environment_errors", lambda: [])
    monkeypatch.setattr("krea_2_turbo_mlx.setup_flow.run_conversion", fake_convert)
    monkeypatch.setattr(
        "krea_2_turbo_mlx.setup_flow.run_doctor",
        lambda *, model, **kwargs: {"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
    )

    assert run_setup_cli(args, stream=StringIO()) == 0
    assert not source.exists()


def test_setup_config_form_round_trips_defaults_and_overrides() -> None:
    config = SetupConfig.from_form(
        {
            "source": ["someone/model"],
            "revision": ["abc123"],
            "source_dir": ["models/custom"],
            "artifact_dir": ["artifacts/custom"],
            "output_dir": ["renders"],
            "cleanup_source": ["on"],
        }
    )

    assert config.source == "someone/model"
    assert config.revision == "abc123"
    assert config.source_dir == Path("models/custom")
    assert config.artifact_dir == Path("artifacts/custom")
    assert config.output_dir == Path("renders")
    assert config.cleanup_source is True
