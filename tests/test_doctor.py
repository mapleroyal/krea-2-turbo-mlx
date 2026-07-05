from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from krea_2_turbo_mlx import doctor as doctor_module
from krea_2_turbo_mlx.doctor import run_doctor
from safetensors_fixtures import write_safetensors_fixture


def test_doctor_accepts_minimal_local_source_without_download(tmp_path: Path) -> None:
    _write_source(tmp_path)

    report = run_doctor(source=tmp_path)

    assert report["status"] == "ok"
    assert report["error_count"] == 0
    check_names = {check["name"] for check in report["checks"]}
    assert "source.manifest" in check_names
    assert "source.required_metadata" in check_names
    assert "source.safetensors" in check_names


def test_doctor_reports_missing_source_metadata_as_error(tmp_path: Path) -> None:
    _write_source(tmp_path)
    (tmp_path / "vae" / "config.json").unlink()

    report = run_doctor(source=tmp_path)

    assert report["status"] == "error"
    errors = [check for check in report["checks"] if check["status"] == "error"]
    assert "source.required_metadata" in {check["name"] for check in errors}


def test_doctor_rejects_quantized_artifact_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "artifact.json").write_text(
        json.dumps(
            {
                "format": "krea-2-turbo-mlx-artifact",
                "quantization": {"bits": 8},
            }
        ),
        encoding="utf-8",
    )

    report = run_doctor(model=artifact)

    assert report["status"] == "error"
    assert "model.precision" in {
        check["name"] for check in report["checks"] if check["status"] == "error"
    }


def test_doctor_rejects_raw_source_passed_as_model(tmp_path: Path) -> None:
    _write_source(tmp_path)

    report = run_doctor(model=tmp_path)

    assert report["status"] == "error"
    error = [check for check in report["checks"] if check["name"] == "model.raw_source"][0]
    assert "raw Diffusers source" in error["message"]
    assert "krea-2-turbo-mlx convert" in error["message"]


def test_missing_mlx_is_warning_not_an_error() -> None:
    real_import = __import__("importlib").import_module

    def fake_import(name: str) -> object:
        if name == "mlx.core":
            raise ImportError("missing mlx")
        return real_import(name)

    with mock.patch(
        "krea_2_turbo_mlx.doctor.importlib.import_module",
        side_effect=fake_import,
    ):
        report = run_doctor()

    mlx_check = [check for check in report["checks"] if check["name"] == "mlx"][0]
    assert report["status"] == "ok"
    assert mlx_check["status"] == "warning"


def test_runtime_required_missing_runtime_modules_are_errors() -> None:
    real_import = __import__("importlib").import_module
    missing = {"mlx.core", "numpy", "PIL", "safetensors", "transformers"}

    def fake_import(name: str) -> object:
        if name in missing:
            raise ImportError(f"missing {name}")
        return real_import(name)

    with mock.patch(
        "krea_2_turbo_mlx.doctor.importlib.import_module",
        side_effect=fake_import,
    ):
        report = run_doctor(runtime_required=True)

    errors = {check["name"] for check in report["checks"] if check["status"] == "error"}
    assert report["status"] == "error"
    assert {"mlx", "numpy", "Pillow", "safetensors", "transformers"} <= errors


def test_doctor_disk_and_memory_checks_report_concrete_values(
    monkeypatch,
) -> None:
    found_disk = doctor_module.RECOMMENDED_SETUP_DISK_BYTES - 1
    found_memory = doctor_module.RECOMMENDED_GENERATION_MEMORY_BYTES - 1

    monkeypatch.setattr(
        doctor_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=found_disk),
    )
    monkeypatch.setattr(doctor_module, "_available_memory_bytes", lambda: found_memory)

    report = run_doctor()

    checks = {check["name"]: check for check in report["checks"]}
    disk = checks["disk.free"]
    memory = checks["memory.available"]
    assert disk["status"] == "warning"
    assert disk["details"]["found_bytes"] == found_disk
    assert (
        disk["details"]["recommended_bytes"]
        == doctor_module.RECOMMENDED_SETUP_DISK_BYTES
    )
    assert "recommended for setup" in disk["message"]
    assert memory["status"] == "warning"
    assert memory["details"]["found_bytes"] == found_memory
    assert (
        memory["details"]["recommended_bytes"]
        == doctor_module.RECOMMENDED_GENERATION_MEMORY_BYTES
    )
    assert "generation needs" in memory["message"]


def _write_source(root: Path) -> None:
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline"}),
        encoding="utf-8",
    )
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "config.json").write_text("{}", encoding="utf-8")
    write_safetensors_fixture(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        {"transformer.weight": ("F32", [1])},
    )
