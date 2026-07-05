from __future__ import annotations

import json
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .errors import Krea2TurboMlxError
from .json_io import write_json
from .safetensors_header import SafetensorsHeader, TensorHeader, read_safetensors_header
from .tensor_selection import QUANTIZED_DTYPES

CHUNK_SIZE = 8 * 1024 * 1024


def write_selected_safetensors(
    *,
    source_root: Path,
    output_root: Path,
    decisions: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    decisions_by_source = _group_decisions_by_source(decisions)
    written_files: list[dict[str, Any]] = []

    for source_rel in sorted(decisions_by_source):
        source_decisions = decisions_by_source[source_rel]
        kept_decisions = [decision for decision in source_decisions if decision["keep"]]
        if not kept_decisions:
            continue

        destination_rel = _single_destination_path(source_rel, kept_decisions)
        source_path = source_root / source_rel
        destination_path = output_root / destination_rel
        source_header = read_safetensors_header(source_path)
        _validate_decisions_match_source(source_rel, source_header, source_decisions)
        _reject_quantized_kept_tensors(source_rel, kept_decisions)

        kept_keys = {str(decision["key"]) for decision in kept_decisions}
        copied_verbatim = kept_keys == set(source_header.tensors)
        _emit_progress(
            progress_callback,
            f"{source_rel} -> {destination_rel} ({len(kept_decisions)} tensors)",
        )
        if copied_verbatim:
            _copy_file(source_path, destination_path)
        else:
            _rebuild_safetensors(source_path, destination_path, source_header, kept_keys)

        _verify_written_safetensors(
            source_rel,
            destination_path,
            source_header,
            kept_decisions,
        )
        written_files.append(
            {
                "source_path": source_rel,
                "destination_path": destination_rel,
                "tensor_count": len(kept_decisions),
                "total_tensor_bytes": sum(int(item["byte_count"]) for item in kept_decisions),
                "copied_verbatim": copied_verbatim,
            }
        )

    transformer_index = _write_transformer_index(output_root, decisions)
    return {
        "written_safetensors": written_files,
        "transformer_index": transformer_index,
        "dtype_equivalence_verified": True,
    }


def _emit_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _group_decisions_by_source(
    decisions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        source_path = str(decision.get("source_path", ""))
        if source_path.endswith(".safetensors"):
            grouped[source_path].append(decision)
    return dict(grouped)


def _single_destination_path(
    source_rel: str,
    kept_decisions: list[dict[str, Any]],
) -> str:
    destinations = {
        str(decision.get("destination_path"))
        for decision in kept_decisions
        if decision.get("destination_path")
    }
    if len(destinations) != 1:
        raise Krea2TurboMlxError(
            f"Kept tensors from {source_rel} do not map to one destination file"
        )
    return next(iter(destinations))


def _validate_decisions_match_source(
    source_rel: str,
    source_header: SafetensorsHeader,
    source_decisions: list[dict[str, Any]],
) -> None:
    decision_keys = {str(decision["key"]) for decision in source_decisions}
    source_keys = set(source_header.tensors)
    if decision_keys != source_keys:
        missing = sorted(source_keys - decision_keys)
        extra = sorted(decision_keys - source_keys)
        raise Krea2TurboMlxError(
            f"Selection report does not match {source_rel}; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )


def _reject_quantized_kept_tensors(
    source_rel: str,
    kept_decisions: list[dict[str, Any]],
) -> None:
    quantized = sorted(
        {
            str(decision.get("dtype"))
            for decision in kept_decisions
            if str(decision.get("dtype")) in QUANTIZED_DTYPES
        }
    )
    if quantized:
        raise Krea2TurboMlxError(
            f"Quantized safetensors dtypes are outside the full-precision contract "
            f"for {source_rel}: {', '.join(quantized)}"
        )


def _copy_file(source: Path, destination: Path) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise Krea2TurboMlxError(f"Unable to copy {source} to {destination}: {exc}") from exc


def _rebuild_safetensors(
    source: Path,
    destination: Path,
    source_header: SafetensorsHeader,
    kept_keys: set[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept_tensors = [
        source_header.tensors[key]
        for key in sorted(kept_keys)
    ]
    header_bytes = _rebuilt_header_bytes(source_header, kept_tensors)

    try:
        with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
            destination_handle.write(struct.pack("<Q", len(header_bytes)))
            destination_handle.write(header_bytes)
            for tensor in kept_tensors:
                source_handle.seek(source_header.payload_start + tensor.data_offsets[0])
                _copy_exact_bytes(source_handle, destination_handle, tensor.byte_count)
    except OSError as exc:
        raise Krea2TurboMlxError(
            f"Unable to rebuild selective safetensors file {destination}: {exc}"
        ) from exc


def _rebuilt_header_bytes(
    source_header: SafetensorsHeader,
    kept_tensors: list[TensorHeader],
) -> bytes:
    header: dict[str, Any] = {}
    if source_header.metadata:
        header["__metadata__"] = {
            key: source_header.metadata[key]
            for key in sorted(source_header.metadata)
        }

    offset = 0
    for tensor in kept_tensors:
        next_offset = offset + tensor.byte_count
        header[tensor.key] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, next_offset],
        }
        offset = next_offset
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _copy_exact_bytes(source: BinaryIO, destination: BinaryIO, byte_count: int) -> None:
    remaining = byte_count
    while remaining:
        chunk = source.read(min(CHUNK_SIZE, remaining))
        if not chunk:
            raise Krea2TurboMlxError("Source safetensors payload ended unexpectedly")
        destination.write(chunk)
        remaining -= len(chunk)


def _verify_written_safetensors(
    source_rel: str,
    destination_path: Path,
    source_header: SafetensorsHeader,
    kept_decisions: list[dict[str, Any]],
) -> None:
    written_header = read_safetensors_header(destination_path)
    for decision in kept_decisions:
        key = str(decision["key"])
        if key not in written_header.tensors:
            raise Krea2TurboMlxError(
                f"Written safetensors file {destination_path} is missing tensor {key!r}"
            )
        source_tensor = source_header.tensors[key]
        written_tensor = written_header.tensors[key]
        if (
            source_tensor.dtype != written_tensor.dtype
            or source_tensor.shape != written_tensor.shape
            or source_tensor.byte_count != written_tensor.byte_count
        ):
            raise Krea2TurboMlxError(
                f"Written tensor {key!r} in {destination_path} does not preserve "
                f"source dtype/shape/byte count from {source_rel}"
            )


def _write_transformer_index(
    output_root: Path,
    decisions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    transformer_decisions = [
        decision
        for decision in decisions
        if decision.get("component") == "transformer" and decision.get("keep")
    ]
    if not transformer_decisions:
        return None

    weight_map: dict[str, str] = {}
    total_size = 0
    for decision in sorted(transformer_decisions, key=lambda item: str(item["key"])):
        destination = Path(str(decision["destination_path"]))
        try:
            shard = destination.relative_to("transformer").as_posix()
        except ValueError:
            shard = destination.name
        weight_map[str(decision["key"])] = shard
        total_size += int(decision["byte_count"])

    payload = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = output_root / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    write_json(index_path, payload)
    return {
        "path": "transformer/diffusion_pytorch_model.safetensors.index.json",
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "total_size": total_size,
    }
