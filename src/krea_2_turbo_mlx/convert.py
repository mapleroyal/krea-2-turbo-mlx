from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from ._version import __version__
from .artifact_write import atomic_output_dir, preflight_free_space
from .constants import (
    ARTIFACT_FORMAT,
    EXPECTED_SOURCE_METADATA_PATHS,
    FULL_PRECISION_ONLY,
    PROJECT_NAME,
)
from .errors import Krea2TurboMlxError
from .hf import download_source, resolve_source_revision
from .json_io import write_json
from .manifest import generate_manifest
from .safetensors_copy import write_selected_safetensors
from .tensor_selection import SELECTION_POLICY_VERSION, select_manifest_tensors

SCHEMA_VERSION = 2
CONVERSION_REPORT_FORMAT = "krea-2-turbo-mlx-conversion-report"


def build_conversion_plan(
    source: str | Path,
    *,
    revision: str | None = None,
    output: str | Path,
) -> dict[str, Any]:
    manifest = generate_manifest(source, revision=revision)
    _assert_readable_safetensors_headers(manifest)
    selection = select_manifest_tensors(manifest)
    _assert_full_precision_selection(selection)

    source_info = _source_info(manifest, source, revision)
    report = _build_conversion_report(
        source_info=source_info,
        manifest=manifest,
        selection=selection,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_conversion_plan_generated",
        "action": "convert",
        "source": source_info,
        "source_kind": source_info["kind"],
        "revision": source_info.get("revision"),
        "output": str(Path(output).expanduser()),
        "writes_artifact": False,
        "full_precision_only": FULL_PRECISION_ONLY,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "required_output_paths": _required_output_paths(manifest, selection),
        "tensor_summary": manifest.get("tensor_summary", {}),
        "selection_summary": selection["summary"],
        "conversion_report": report,
    }


def run_conversion(
    source: str | Path,
    *,
    revision: str | None = None,
    output: str | Path,
    source_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    source_path = Path(source).expanduser()
    if not source_path.exists():
        effective_revision = resolve_source_revision(source, revision)
        source_path = download_source(
            source,
            revision=effective_revision,
            local_dir=source_dir,
            progress_callback=lambda message: _emit_progress(
                progress_callback,
                f"download: {message}",
            ),
        )
        revision = effective_revision
    if not source_path.is_dir():
        raise Krea2TurboMlxError(f"Conversion source must be a directory: {source_path}")

    _emit_progress(progress_callback, "validate source layout")
    _assert_required_metadata(source_path)
    output_path = Path(output).expanduser()
    _emit_progress(progress_callback, "build manifest and conversion plan")
    plan = build_conversion_plan(source_path, revision=revision, output=output_path)
    required_bytes = _required_output_bytes(source_path, plan)
    _emit_progress(progress_callback, "check free disk space")
    preflight_free_space(output_path, required_bytes=required_bytes, label="conversion")

    with atomic_output_dir(output_path, label="conversion") as temp:
        _emit_progress(progress_callback, "copy metadata")
        copied_metadata = _copy_metadata_files(source_path, temp, plan)
        _emit_progress(progress_callback, "copy safetensors")
        copy_summary = write_selected_safetensors(
            source_root=source_path,
            output_root=temp,
            decisions=plan["conversion_report"]["tensor_decisions"],
            progress_callback=lambda message: _emit_progress(
                progress_callback,
                f"safetensors: {message}",
            ),
        )
        report = dict(plan["conversion_report"])
        report["copy_summary"] = copy_summary
        report["copied_metadata"] = copied_metadata
        artifact = _artifact_metadata(
            plan=plan,
            report=report,
            copy_summary=copy_summary,
            source_path=source_path,
            copied_metadata=copied_metadata,
        )
        _emit_progress(progress_callback, "write reports")
        write_json(temp / "conversion_report.json", report)
        write_json(temp / "artifact.json", artifact)

    result = dict(plan)
    result.update(
        {
            "status": "artifact_written",
            "writes_artifact": True,
            "artifact_path": str(output_path),
            "artifact": artifact,
        }
    )
    return result


def _build_conversion_report(
    *,
    source_info: dict[str, Any],
    manifest: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": CONVERSION_REPORT_FORMAT,
        "generator": PROJECT_NAME,
        "generator_version": __version__,
        "source": source_info,
        "full_precision_only": FULL_PRECISION_ONLY,
        "selection_policy_version": selection["policy_version"],
        "manifest_status": manifest.get("status"),
        "manifest_tensor_summary": manifest.get("tensor_summary", {}),
        "selection_summary": selection["summary"],
        "rule_matches": selection["rule_matches"],
        "tensor_decisions": selection["decisions"],
    }


def _artifact_metadata(
    *,
    plan: dict[str, Any],
    report: dict[str, Any],
    copy_summary: dict[str, Any],
    source_path: Path,
    copied_metadata: list[str],
) -> dict[str, Any]:
    summary = report["selection_summary"]
    selected = summary["selected"]
    excluded = summary["excluded"]
    return {
        "schema_version": SCHEMA_VERSION,
        "format": ARTIFACT_FORMAT,
        "generator": PROJECT_NAME,
        "generator_version": __version__,
        "source": plan["source"],
        "full_precision_only": FULL_PRECISION_ONLY,
        "selection_policy_version": plan["selection_policy_version"],
        "selected": {
            "tensor_count": selected["tensor_count"],
            "total_tensor_bytes": selected["total_tensor_bytes"],
            "dtypes": selected["dtypes"],
            "components": selected["components"],
        },
        "excluded": {
            "tensor_count": excluded["tensor_count"],
            "total_tensor_bytes": excluded["total_tensor_bytes"],
            "dtypes": excluded["dtypes"],
            "components": excluded["components"],
        },
        "byte_totals": {
            "selected_tensor_bytes": selected["total_tensor_bytes"],
            "excluded_tensor_bytes": excluded["total_tensor_bytes"],
        },
        "precision": {
            "preserves_source_dtypes": True,
            "dtype_equivalence_verified": bool(
                copy_summary.get("dtype_equivalence_verified")
            ),
            "quantized_dtypes_present": False,
            "selected_dtype_histogram": selected["dtypes"],
        },
        "conversion_report": "conversion_report.json",
        "required_files": plan["required_output_paths"],
        "provenance": {
            "source_revision": _source_revision(plan["source"]),
            "selected_tensor_fingerprint": _selected_tensor_fingerprint(report),
            "selected_header_fingerprint": _selected_header_fingerprint(report),
            "copied_metadata_fingerprint": _copied_metadata_fingerprint(
                source_path,
                copied_metadata,
            ),
            "source_layout": _source_layout_summary(plan),
        },
    }


def _source_info(
    manifest: dict[str, Any],
    requested_source: str | Path,
    requested_revision: str | None,
) -> dict[str, Any]:
    source_kind = str(manifest.get("source_kind", "unknown"))
    if source_kind == "huggingface_model":
        return {
            "kind": source_kind,
            "repo_id": manifest.get("repo_id") or str(requested_source),
            "requested_revision": requested_revision,
            "revision": manifest.get("resolved_revision"),
        }

    path = Path(str(manifest.get("source") or requested_source)).expanduser()
    return {
        "kind": source_kind,
        "path": str(path),
        "revision": requested_revision,
    }


def _assert_readable_safetensors_headers(manifest: dict[str, Any]) -> None:
    errors = [
        f"{header.get('path')}: {header.get('error')}"
        for header in manifest.get("safetensors_headers", [])
        if header.get("error")
    ]
    if errors:
        raise Krea2TurboMlxError(
            "Cannot build an exact conversion plan with unreadable safetensors headers: "
            + "; ".join(errors[:8])
        )


def _assert_full_precision_selection(selection: dict[str, Any]) -> None:
    summary = selection["summary"]
    if summary.get("quantized_dtypes_present"):
        raise Krea2TurboMlxError(
            "Quantized safetensors dtypes are outside the full-precision contract: "
            + ", ".join(summary.get("quantized_dtypes", []))
        )


def _assert_required_metadata(source_path: Path) -> None:
    missing = [
        rel_path
        for rel_path in EXPECTED_SOURCE_METADATA_PATHS
        if not (source_path / rel_path).is_file()
    ]
    if missing:
        raise Krea2TurboMlxError(
            "Source is missing required Diffusers metadata: " + ", ".join(missing)
        )


def _required_output_paths(
    manifest: dict[str, Any],
    selection: dict[str, Any],
) -> list[str]:
    paths = {"artifact.json", "conversion_report.json"}
    for summary in [*manifest.get("configs", []), *manifest.get("text_metadata", [])]:
        path = str(summary.get("path", ""))
        if path:
            paths.add(path)

    for decision in selection["decisions"]:
        if decision["keep"] and decision.get("destination_path"):
            paths.add(str(decision["destination_path"]))
    if any(
        decision["keep"] and decision["component"] == "transformer"
        for decision in selection["decisions"]
    ):
        paths.add("transformer/diffusion_pytorch_model.safetensors.index.json")
    return sorted(paths)


def _required_output_bytes(source_path: Path, plan: dict[str, Any]) -> int:
    selected_bytes = int(
        plan["selection_summary"]["selected"].get("total_tensor_bytes", 0)
    )
    metadata_bytes = sum(
        (source_path / path).stat().st_size
        for path in _metadata_paths_to_copy(source_path, plan)
        if (source_path / path).is_file()
    )
    report_bytes = len(json.dumps(plan.get("conversion_report", {})).encode("utf-8"))
    return selected_bytes + metadata_bytes + report_bytes + 16 * 1024


def _copy_metadata_files(
    source_path: Path,
    output_path: Path,
    plan: dict[str, Any],
) -> list[str]:
    copied: list[str] = []
    for rel_path in _metadata_paths_to_copy(source_path, plan):
        source_file = source_path / rel_path
        destination_file = output_path / rel_path
        if not source_file.is_file():
            continue
        try:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination_file)
        except OSError as exc:
            raise Krea2TurboMlxError(
                f"Unable to copy metadata file {source_file} to {destination_file}: {exc}"
            ) from exc
        copied.append(rel_path)
    return copied


def _metadata_paths_to_copy(source_path: Path, plan: dict[str, Any]) -> list[str]:
    paths = set(EXPECTED_SOURCE_METADATA_PATHS)
    for rel_path in plan.get("required_output_paths", []):
        if rel_path.endswith((".safetensors", ".safetensors.index.json")):
            continue
        if rel_path in {"artifact.json", "conversion_report.json"}:
            continue
        paths.add(str(rel_path))

    tokenizer = source_path / "tokenizer"
    if tokenizer.is_dir():
        for path in tokenizer.rglob("*"):
            if path.is_file():
                paths.add(path.relative_to(source_path).as_posix())
    return sorted(paths)


def _selected_tensor_fingerprint(report: dict[str, Any]) -> str:
    payload = [
        {
            "key": decision.get("key"),
            "component": decision.get("component"),
            "source_path": decision.get("source_path"),
            "destination_path": decision.get("destination_path"),
            "dtype": decision.get("dtype"),
            "shape": decision.get("shape"),
            "byte_count": decision.get("byte_count"),
        }
        for decision in report.get("tensor_decisions", [])
        if decision.get("keep")
    ]
    return _json_fingerprint(payload)


def _selected_header_fingerprint(report: dict[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for decision in report.get("tensor_decisions", []):
        if not decision.get("keep"):
            continue
        source_path = str(decision.get("source_path"))
        entry = payload.setdefault(
            source_path,
            {
                "component": decision.get("component"),
                "destination_path": decision.get("destination_path"),
                "tensors": [],
            },
        )
        entry["tensors"].append(
            {
                "key": decision.get("key"),
                "dtype": decision.get("dtype"),
                "shape": decision.get("shape"),
                "byte_count": decision.get("byte_count"),
            }
        )
    for entry in payload.values():
        entry["tensors"] = sorted(entry["tensors"], key=lambda item: str(item["key"]))
    return _json_fingerprint(payload)


def _copied_metadata_fingerprint(source_path: Path, copied_metadata: list[str]) -> str:
    payload: list[dict[str, Any]] = []
    for rel_path in sorted(copied_metadata):
        path = source_path / rel_path
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise Krea2TurboMlxError(
                f"Unable to fingerprint copied metadata {path}: {exc}"
            ) from exc
        payload.append(
            {
                "path": rel_path,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return _json_fingerprint(payload)


def _source_layout_summary(plan: dict[str, Any]) -> dict[str, Any]:
    tensor_summary = plan.get("tensor_summary", {})
    selection_summary = plan.get("selection_summary", {})
    return {
        "source_kind": plan.get("source_kind"),
        "required_output_path_count": len(plan.get("required_output_paths", [])),
        "source_tensor_count": tensor_summary.get("tensor_count"),
        "source_tensor_bytes": tensor_summary.get("total_tensor_bytes"),
        "selected_tensor_count": selection_summary.get("selected", {}).get("tensor_count"),
        "selected_tensor_bytes": selection_summary.get("selected", {}).get(
            "total_tensor_bytes"
        ),
    }


def _source_revision(source: dict[str, Any]) -> str | None:
    value = source.get("revision") or source.get("resolved_revision")
    return None if value is None else str(value)


def _json_fingerprint(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _emit_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
