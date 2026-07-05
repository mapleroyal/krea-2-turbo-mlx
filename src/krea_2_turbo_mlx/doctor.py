from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._version import __version__
from .constants import ARTIFACT_FORMAT, EXPECTED_SOURCE_METADATA_PATHS, PROJECT_NAME
from .errors import Krea2TurboMlxError
from .json_io import read_json_object
from .manifest import generate_manifest

# Disk is reported in decimal GB by macOS (Finder), so recommendations use decimal
# units to match what users see. Peak setup usage is the converted artifact plus the
# source download coexisting during conversion (~73 GB); 75 GB leaves a little headroom.
RECOMMENDED_SETUP_DISK_BYTES = 75 * 1000**3
# Generation holds the whole model resident and peaks near 49 GB; require roughly 50 GB
# of memory available beyond the OS and any other running workloads.
RECOMMENDED_GENERATION_MEMORY_BYTES = 50 * 1000**3

MIN_ARTIFACT_REQUIRED_FILES = {
    "artifact.json",
    "conversion_report.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
}


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def run_doctor(
    *,
    model: str | Path | None = None,
    source: str | Path | None = None,
    runtime_required: bool = False,
) -> dict[str, Any]:
    checks: list[DoctorCheck] = []
    checks.extend(_environment_checks(runtime_required=runtime_required))
    if source is not None:
        checks.extend(_source_checks(Path(source).expanduser()))
    if model is not None:
        checks.extend(_model_checks(Path(model).expanduser()))

    error_count = sum(1 for check in checks if check.status == "error")
    warning_count = sum(1 for check in checks if check.status == "warning")
    return {
        "schema_version": 1,
        "generator": PROJECT_NAME,
        "generator_version": __version__,
        "status": "error" if error_count else "ok",
        "error_count": error_count,
        "warning_count": warning_count,
        "checks": [check.to_dict() for check in checks],
    }


def format_doctor_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"{PROJECT_NAME} doctor: {report.get('status', 'unknown')}",
        f"errors: {report.get('error_count', 0)}, warnings: {report.get('warning_count', 0)}",
    ]
    for check in report.get("checks", []):
        status = str(check.get("status", "unknown")).upper()
        lines.append(f"{status}: {check.get('name')}: {check.get('message')}")
    return "\n".join(lines) + "\n"


def _environment_checks(*, runtime_required: bool = False) -> list[DoctorCheck]:
    checks = [
        _ok(
            "python",
            f"{platform.python_implementation()} {platform.python_version()}",
            minimum="3.10",
        )
        if sys.version_info >= (3, 10)
        else _error(
            "python",
            f"Python 3.10 or newer is required; found {platform.python_version()}",
        ),
        _ok("package", f"{PROJECT_NAME} {__version__}"),
    ]

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        checks.append(_ok("platform", "macOS on Apple Silicon"))
    else:
        checks.append(
            _warning(
                "platform",
                f"MLX generation is intended for macOS on Apple Silicon; found {system} {machine}",
            )
        )

    checks.append(_module_check("huggingface_hub", "huggingface-hub", required_for="remote metadata"))
    checks.append(
        _module_check(
            "mlx.core",
            "mlx",
            required_for="MLX runtime execution",
            required=runtime_required,
        )
    )
    checks.append(
        _module_check(
            "numpy",
            "numpy",
            required_for="runtime tensor and image conversion",
            required=runtime_required,
        )
    )
    checks.append(
        _module_check(
            "PIL",
            "Pillow",
            required_for="PNG and source-image handling",
            required=runtime_required,
        )
    )
    checks.append(
        _module_check(
            "safetensors",
            "safetensors",
            required_for="runtime validation",
            required=runtime_required,
        )
    )
    checks.append(
        _module_check(
            "transformers",
            "transformers",
            required_for="text tokenization",
            required=runtime_required,
        )
    )
    checks.append(_disk_check(Path.cwd()))
    checks.append(_unified_memory_check())
    return checks


def _module_check(
    module_name: str,
    label: str,
    *,
    required_for: str,
    required: bool = False,
) -> DoctorCheck:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        factory = _error if required else _warning
        return factory(
            label,
            f"{label} is not importable ({required_for}). Run `./setup.sh` or "
            "install `krea-2-turbo-mlx[runtime]`.",
        )
    except Exception as exc:  # pragma: no cover - depends on third-party import side effects.
        factory = _error if required else _warning
        return factory(label, f"{label} import failed ({required_for}): {exc}")
    version = getattr(module, "__version__", None)
    message = label if version is None else f"{label} {version}"
    return _ok(label, message)


def _source_checks(source: Path) -> list[DoctorCheck]:
    if not source.exists():
        return [_error("source", f"Source path does not exist: {source}; doctor does not download")]
    if not source.is_dir():
        return [_error("source", f"Source path must be a directory: {source}")]

    try:
        manifest = generate_manifest(source)
    except Krea2TurboMlxError as exc:
        return [_error("source.manifest", str(exc))]

    checks: list[DoctorCheck] = [_ok("source.manifest", "Local source manifest generated")]
    missing = [
        str(source / rel_path)
        for rel_path in EXPECTED_SOURCE_METADATA_PATHS
        if not (source / rel_path).exists()
    ]
    if missing:
        checks.append(_error("source.required_metadata", "Source is missing required metadata", missing=missing))
    else:
        checks.append(_ok("source.required_metadata", "Required source metadata is present"))

    header_errors = [
        f"{header.get('path')}: {header.get('error')}"
        for header in manifest.get("safetensors_headers", [])
        if header.get("error")
    ]
    if header_errors:
        checks.append(_error("source.safetensors", "Invalid safetensors headers", errors=header_errors[:8]))
    else:
        checks.append(_ok("source.safetensors", "Safetensors headers are readable"))

    index_errors = [
        str(index.get("path"))
        for index in manifest.get("safetensors_indexes", [])
        if index.get("validation", {}).get("status") == "error"
    ]
    if index_errors:
        checks.append(_error("source.indexes", "Safetensors indexes do not match headers", indexes=index_errors))
    else:
        checks.append(_ok("source.indexes", "Safetensors indexes match readable headers"))
    return checks


def _model_checks(root: Path) -> list[DoctorCheck]:
    if not root.exists():
        return [_error("model", f"Model artifact directory does not exist: {root}")]
    if not root.is_dir():
        return [_error("model", f"Model artifact path must be a directory: {root}")]

    artifact_path = root / "artifact.json"
    if not artifact_path.exists():
        if _looks_like_raw_source(root):
            return [
                _error(
                    "model.raw_source",
                    "This path is a raw Diffusers source, not a converted MLX artifact. "
                    f"Run `krea-2-turbo-mlx convert --source {root} --output artifacts/krea-2-turbo-mlx`.",
                )
            ]
        return [
            _error(
                "model.artifact",
                f"Missing converted artifact metadata at {artifact_path}. "
                "Run setup or convert before using this path as --model.",
            )
        ]

    try:
        artifact = read_json_object(artifact_path)
    except Krea2TurboMlxError as exc:
        return [_error("model.artifact", str(exc))]

    checks: list[DoctorCheck] = [_ok("model.artifact", f"Artifact metadata found at {artifact_path}")]
    if artifact.get("format") == ARTIFACT_FORMAT:
        checks.append(_ok("model.format", "Artifact format is recognized"))
    else:
        checks.append(_error("model.format", f"Unrecognized artifact format: {artifact.get('format')}"))
    if "quantization" in artifact:
        checks.append(_error("model.precision", "This build supports only full-precision artifacts; quantized artifacts are not supported."))
    else:
        checks.append(_ok("model.precision", "No quantization metadata detected"))
    checks.extend(_artifact_precision_checks(artifact))
    checks.extend(_artifact_file_checks(root, artifact))
    checks.extend(_artifact_safetensors_checks(root))
    return checks


def _artifact_precision_checks(artifact: Mapping[str, Any]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    if artifact.get("full_precision_only") is True:
        checks.append(_ok("model.full_precision", "Artifact declares full-precision-only conversion"))
    else:
        checks.append(_error("model.full_precision", "Artifact does not declare full_precision_only=true"))

    precision = artifact.get("precision")
    if not isinstance(precision, Mapping):
        return [_error("model.precision_proof", "Artifact precision proof is missing")]
    if precision.get("dtype_equivalence_verified") is True:
        checks.append(_ok("model.dtype_equivalence", "Written tensor dtypes match source dtypes"))
    else:
        checks.append(_error("model.dtype_equivalence", "Source/write dtype equivalence is not verified"))
    if precision.get("quantized_dtypes_present") is False:
        checks.append(_ok("model.quantized_dtypes", "No quantized dtypes are declared"))
    else:
        checks.append(_error("model.quantized_dtypes", "Artifact declares quantized dtypes"))
    return checks


def _artifact_file_checks(root: Path, artifact: Mapping[str, Any]) -> list[DoctorCheck]:
    required = set(MIN_ARTIFACT_REQUIRED_FILES)
    declared = artifact.get("required_files", [])
    if isinstance(declared, list):
        required.update(str(item) for item in declared)
    missing = [str(root / rel_path) for rel_path in sorted(required) if not (root / rel_path).is_file()]
    if missing:
        return [
            _error(
                "model.required_files",
                "Artifact is missing required files",
                missing=missing[:16],
            )
        ]
    return [_ok("model.required_files", "Required artifact files are present")]


def _artifact_safetensors_checks(root: Path) -> list[DoctorCheck]:
    try:
        manifest = generate_manifest(root)
    except Krea2TurboMlxError as exc:
        return [_error("model.manifest", str(exc))]

    checks: list[DoctorCheck] = [_ok("model.manifest", "Artifact manifest generated")]
    header_errors = [
        f"{header.get('path')}: {header.get('error')}"
        for header in manifest.get("safetensors_headers", [])
        if header.get("error")
    ]
    if header_errors:
        checks.append(_error("model.safetensors", "Invalid artifact safetensors headers", errors=header_errors[:8]))
    else:
        checks.append(_ok("model.safetensors", "Artifact safetensors headers are readable"))

    transformer_indexes = [
        index
        for index in manifest.get("safetensors_indexes", [])
        if index.get("component") == "transformer"
    ]
    if not transformer_indexes:
        checks.append(_error("model.transformer_index", "Transformer safetensors index is missing"))
        return checks

    index_errors = [
        str(index.get("path"))
        for index in transformer_indexes
        if index.get("validation", {}).get("status") == "error"
    ]
    if index_errors:
        checks.append(
            _error(
                "model.transformer_index",
                "Transformer safetensors index does not match written shards",
                indexes=index_errors,
            )
        )
    else:
        checks.append(_ok("model.transformer_index", "Transformer index matches written shards"))
    return checks


def _ok(name: str, message: str, **details: Any) -> DoctorCheck:
    return DoctorCheck(name, "ok", message, details or None)


def _warning(name: str, message: str, **details: Any) -> DoctorCheck:
    return DoctorCheck(name, "warning", message, details or None)


def _error(name: str, message: str, **details: Any) -> DoctorCheck:
    return DoctorCheck(name, "error", message, details or None)


def _disk_check(path: Path) -> DoctorCheck:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return _warning("disk.free", f"Unable to check free disk space at {path}: {exc}")
    details = {
        "path": str(path),
        "found_bytes": usage.free,
        "recommended_bytes": RECOMMENDED_SETUP_DISK_BYTES,
    }
    message = (
        f"Free disk at {path}: {_format_gb(usage.free)}; recommended for setup: "
        f"{_format_gb(RECOMMENDED_SETUP_DISK_BYTES)}."
    )
    if usage.free < RECOMMENDED_SETUP_DISK_BYTES:
        return _warning(
            "disk.free",
            message + " Free space or choose a larger project disk before setup.",
            **details,
        )
    return _ok("disk.free", message, **details)


def _unified_memory_check() -> DoctorCheck:
    found = _available_memory_bytes()
    details = {
        "found_bytes": found,
        "recommended_bytes": RECOMMENDED_GENERATION_MEMORY_BYTES,
    }
    needed = (
        f"generation needs about {_format_gb(RECOMMENDED_GENERATION_MEMORY_BYTES)} of "
        "memory beyond the OS and your other workloads"
    )
    if found is None:
        return _warning(
            "memory.available",
            f"Unable to determine available memory; {needed}.",
            **details,
        )
    message = f"Available memory: {_format_gb(found)}; {needed}."
    if found < RECOMMENDED_GENERATION_MEMORY_BYTES:
        return _warning(
            "memory.available",
            message + " Free up memory or use a machine with more before generating.",
            **details,
        )
    return _ok("memory.available", message, **details)


def _available_memory_bytes() -> int | None:
    """Best-effort estimate of memory that can be allocated without swapping.

    macOS keeps free RAM busy as file cache, so raw free pages understate what is
    actually available. This sums the reclaimable page classes (free, inactive,
    speculative, purgeable) reported by ``vm_stat`` — roughly what the system can
    hand to a new allocation before it needs to swap.
    """

    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    page_size = 4096
    pages: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if "page size of" in line:
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                page_size = int(digits)
            continue
        label, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            pages[label.strip()] = int(value)

    reclaimable_keys = (
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    )
    reclaimable = sum(pages.get(key, 0) for key in reclaimable_keys)
    if reclaimable == 0 and not any(key in pages for key in reclaimable_keys):
        return None
    return reclaimable * page_size


def _looks_like_raw_source(root: Path) -> bool:
    return any((root / rel_path).is_file() for rel_path in EXPECTED_SOURCE_METADATA_PATHS)


def _format_gb(value: int | None) -> str:
    """Format bytes as decimal GB, matching how macOS reports disk and memory."""
    if value is None:
        return "unknown"
    return f"{value / 1000**3:.2f} GB"
