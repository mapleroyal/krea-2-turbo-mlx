from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen

from .constants import (
    DIFFUSERS_COMPONENTS,
    EXPECTED_SOURCE_COMPONENTS,
    FULL_PRECISION_ONLY,
    OFFICIAL_HF_REPO_ID,
    OFFICIAL_HF_REVISION,
    OFFICIAL_SOURCE_DTYPES,
    PROJECT_NAME,
)
from .errors import Krea2TurboMlxError
from .hf import _load_hf_tools, resolve_source_revision
from .safetensors_header import (
    MAX_HEADER_LENGTH,
    SafetensorsHeader,
    parse_safetensors_header_bytes,
    read_safetensors_header,
)

SCHEMA_VERSION = 2
MAX_REMOTE_METADATA_BYTES = 16 * 1024 * 1024
REMOTE_RANGE_TIMEOUT_SECONDS = 30
REMOTE_COMPONENTS = ("model_index", *DIFFUSERS_COMPONENTS)

_JSON_METADATA_FILES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model_index.json",
    "scheduler_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
_TEXT_METADATA_FILES = {"chat_template.jinja"}


def generate_manifest(source: str | Path, revision: str | None = None) -> dict[str, Any]:
    source_text = str(source)
    source_path = Path(source_text).expanduser()
    if source_path.exists():
        return _generate_local_manifest(source_path)
    return _generate_remote_manifest(source_text, revision)


def _generate_local_manifest(source_path: Path) -> dict[str, Any]:
    root = source_path if source_path.is_dir() else source_path.parent
    safetensors_paths = _local_safetensors_paths(source_path)
    tensor_inventory, safetensors_headers = _build_local_tensor_inventory(
        safetensors_paths,
        root,
    )
    configs = [
        _summarize_json_file(path, root)
        for path in _local_json_metadata_paths(source_path)
    ]
    text_metadata = [
        _summarize_text_file(path, root)
        for path in _local_text_metadata_paths(source_path)
    ]
    indexes = [
        _summarize_safetensors_index(path, root, safetensors_headers)
        for path in _local_safetensors_index_paths(source_path)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(source_path),
        "source_kind": "local_directory" if source_path.is_dir() else "local_file",
        "official_source": _official_source_payload(),
        "configs": configs,
        "text_metadata": text_metadata,
        "component_configs": _group_component_configs(configs, text_metadata),
        "safetensors_indexes": indexes,
        "safetensors_files": [_relative(path, root) for path in safetensors_paths],
        "safetensors_headers": safetensors_headers,
        "tensor_inventory": tensor_inventory,
        "tensor_summary": _summarize_tensor_inventory(tensor_inventory),
        "status": "local_manifest_generated",
    }


def _generate_remote_manifest(source: str, revision: str | None) -> dict[str, Any]:
    effective_revision = resolve_source_revision(source, revision)
    if effective_revision is None:
        raise Krea2TurboMlxError("Remote manifest requires a resolved revision.")

    HfApi, hf_hub_download = _load_hf_tools()
    info = HfApi().model_info(
        repo_id=source,
        revision=effective_revision,
        files_metadata=True,
    )
    inventory = sorted(
        (
            item
            for item in (_summarize_remote_file(sibling) for sibling in _siblings(info))
            if item.get("path")
        ),
        key=lambda item: str(item["path"]),
    )
    tensor_inventory, safetensors_headers = _build_remote_tensor_inventory(
        source,
        effective_revision,
        inventory,
    )

    configs: list[dict[str, Any]] = []
    text_metadata: list[dict[str, Any]] = []
    safetensors_indexes: list[dict[str, Any]] = []
    remote_files_read: list[str] = []

    for item in inventory:
        path = str(item["path"])
        if not _should_download_remote_metadata(path, item.get("size")):
            continue
        remote_files_read.append(path)
        content = _download_remote_text_file(
            hf_hub_download,
            repo_id=source,
            revision=effective_revision,
            filename=path,
        )
        if path.endswith(".safetensors.index.json"):
            safetensors_indexes.append(_summarize_safetensors_index_text(content, path))
        elif path.endswith(".jinja"):
            text_metadata.append(_summarize_text_text(content, path))
        elif path.endswith(".json"):
            configs.append(_summarize_json_text(content, path))

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_kind": "huggingface_model",
        "repo_id": _repo_id(info, source),
        "requested_revision": revision,
        "resolved_revision": _resolved_revision(info, effective_revision),
        "last_modified": _isoformat(_get_attr(info, "last_modified")),
        "license": _license(info),
        "tags": sorted(str(tag) for tag in (_get_attr(info, "tags") or [])),
        "library_name": _get_attr(info, "library_name"),
        "pipeline_tag": _get_attr(info, "pipeline_tag"),
        "official_source": _official_source_payload(),
        "file_inventory": inventory,
        "remote_files_read": remote_files_read,
        "configs": configs,
        "text_metadata": text_metadata,
        "component_configs": _group_component_configs(configs, text_metadata),
        "safetensors_indexes": safetensors_indexes,
        "safetensors_files": [
            str(item["path"])
            for item in inventory
            if str(item["path"]).endswith(".safetensors")
        ],
        "safetensors_headers": safetensors_headers,
        "tensor_inventory": tensor_inventory,
        "tensor_summary": _summarize_tensor_inventory(tensor_inventory),
        "status": "remote_manifest_generated",
    }


def _local_json_metadata_paths(source_path: Path) -> list[Path]:
    if not source_path.is_dir():
        return []
    root = source_path
    return [
        path
        for path in sorted(root.rglob("*.json"))
        if _is_json_metadata(path, root)
    ]


def _local_text_metadata_paths(source_path: Path) -> list[Path]:
    if not source_path.is_dir():
        return []
    root = source_path
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_text_metadata(path, root)
    ]


def _local_safetensors_index_paths(source_path: Path) -> list[Path]:
    if not source_path.is_dir():
        return []
    return sorted(source_path.rglob("*.safetensors.index.json"))


def _local_safetensors_paths(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path] if source_path.name.endswith(".safetensors") else []
    return [
        path
        for path in sorted(source_path.rglob("*.safetensors"))
        if not path.name.endswith(".index.json")
    ]


def _is_json_metadata(path: Path, root: Path) -> bool:
    if path.name.endswith(".safetensors.index.json"):
        return False
    rel = _relative(path, root)
    parts = rel.split("/")
    if len(parts) == 1:
        return path.name == "model_index.json"

    component = parts[0]
    if component in {"text_encoder", "transformer", "vae"}:
        return path.name in {"config.json", "generation_config.json"}
    if component == "tokenizer":
        return path.name in _JSON_METADATA_FILES
    if component == "scheduler":
        return path.name == "scheduler_config.json"
    return False


def _is_text_metadata(path: Path, root: Path) -> bool:
    rel = _relative(path, root)
    parts = rel.split("/")
    return len(parts) > 1 and parts[0] == "tokenizer" and path.name in _TEXT_METADATA_FILES


def _build_local_tensor_inventory(
    safetensors_paths: Iterable[Path],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    header_summaries: list[dict[str, Any]] = []

    for path in safetensors_paths:
        rel = _relative(path, root)
        component = _component_for_path(rel)
        header_summary: dict[str, Any] = {
            "path": rel,
            "component": component,
        }

        try:
            header = read_safetensors_header(path)
        except Krea2TurboMlxError as exc:
            header_summary["error"] = str(exc)
            header_summaries.append(header_summary)
            continue

        header_summary.update(
            {
                "metadata": header.metadata,
                "payload_start": header.payload_start,
                "tensor_count": len(header.tensors),
                "tensors": sorted(header.tensors),
                "total_tensor_bytes": sum(tensor.byte_count for tensor in header.tensors.values()),
            }
        )
        header_summaries.append(header_summary)

        for key, tensor in sorted(header.tensors.items()):
            inventory.append(
                {
                    "key": key,
                    "component": component,
                    "path": rel,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "byte_count": tensor.byte_count,
                    "preserve_source_dtype": True,
                }
            )

    return inventory, header_summaries


def _build_remote_tensor_inventory(
    repo_id: str,
    revision: str,
    file_inventory: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    header_summaries: list[dict[str, Any]] = []

    for file_summary in file_inventory:
        path = str(file_summary.get("path", ""))
        if not path.endswith(".safetensors"):
            continue
        component = _component_for_path(path)
        header_summary: dict[str, Any] = {
            "path": path,
            "component": component,
            "remote_byte_ranges_only": True,
        }
        size = file_summary.get("size")
        file_size = size if isinstance(size, int) else None

        try:
            header = _read_remote_safetensors_header(
                repo_id,
                revision,
                path,
                file_size=file_size,
            )
        except Krea2TurboMlxError as exc:
            header_summary["error"] = str(exc)
            header_summaries.append(header_summary)
            continue

        header_summary.update(_header_summary_payload(header))
        header_summaries.append(header_summary)
        inventory.extend(_tensor_inventory_entries(path, component, header))

    return inventory, header_summaries


def _summarize_tensor_inventory(inventory: Iterable[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    total = 0
    total_bytes = 0
    for entry in inventory:
        total += 1
        total_bytes += int(entry.get("byte_count", 0))
        component = str(entry.get("component", "other"))
        summary = components.setdefault(
            component,
            {"tensor_count": 0, "total_tensor_bytes": 0, "dtypes": {}},
        )
        summary["tensor_count"] += 1
        summary["total_tensor_bytes"] += int(entry.get("byte_count", 0))
        dtype = str(entry.get("dtype", "unknown"))
        summary["dtypes"][dtype] = summary["dtypes"].get(dtype, 0) + 1
    return {
        "tensor_count": total,
        "total_tensor_bytes": total_bytes,
        "components": components,
    }


def _header_summary_payload(header: SafetensorsHeader) -> dict[str, Any]:
    return {
        "metadata": header.metadata,
        "payload_start": header.payload_start,
        "tensor_count": len(header.tensors),
        "tensors": sorted(header.tensors),
        "total_tensor_bytes": sum(tensor.byte_count for tensor in header.tensors.values()),
    }


def _tensor_inventory_entries(
    path: str,
    component: str,
    header: SafetensorsHeader,
) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "component": component,
            "path": path,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "byte_count": tensor.byte_count,
            "preserve_source_dtype": True,
        }
        for key, tensor in sorted(header.tensors.items())
    ]


def _summarize_json_file(path: Path, root: Path) -> dict[str, Any]:
    rel = _relative(path, root)
    try:
        return _summarize_json_text(path.read_text(encoding="utf-8"), rel)
    except OSError as exc:
        return {
            "path": rel,
            "component": _component_for_path(rel),
            "kind": "json",
            "error": str(exc),
        }


def _summarize_text_file(path: Path, root: Path) -> dict[str, Any]:
    rel = _relative(path, root)
    try:
        return _summarize_text_text(path.read_text(encoding="utf-8"), rel)
    except OSError as exc:
        return {
            "path": rel,
            "component": _component_for_path(rel),
            "kind": "text",
            "error": str(exc),
        }


def _summarize_json_text(content: str, path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": path,
        "component": _component_for_path(path),
        "kind": "json",
    }
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        summary["error"] = str(exc)
        return summary
    if not isinstance(data, dict):
        summary["error"] = "JSON metadata is not an object"
        return summary

    summary["top_level_keys"] = sorted(str(key) for key in data)
    for key in (
        "_class_name",
        "_diffusers_version",
        "architectures",
        "dtype",
        "torch_dtype",
        "model_type",
        "transformers_version",
        "is_distilled",
        "patch_size",
    ):
        if key in data:
            summary[key] = data[key]
    if path.endswith("model_index.json"):
        summary["component_classes"] = _component_classes(data)
        if "text_encoder_select_layers" in data:
            summary["text_encoder_select_layers"] = data["text_encoder_select_layers"]
    summary.update(_config_facts(data))
    return summary


def _summarize_text_text(content: str, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "component": _component_for_path(path),
        "kind": "text",
        "char_count": len(content),
        "contains_system_role": "system" in content,
        "contains_user_role": "user" in content,
        "contains_generation_prompt": "add_generation_prompt" in content,
    }


def _summarize_safetensors_index(
    path: Path,
    root: Path,
    safetensors_headers: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rel = _relative(path, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": rel, "component": _component_for_path(rel), "error": str(exc)}
    summary = _summarize_safetensors_index_payload(data, rel)
    if safetensors_headers is not None:
        summary["validation"] = _validate_index_against_headers(
            path,
            root,
            data.get("weight_map", {}),
            safetensors_headers,
        )
    return summary


def _summarize_safetensors_index_text(content: str, path: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return {"path": path, "component": _component_for_path(path), "error": str(exc)}
    return _summarize_safetensors_index_payload(data, path)


def _summarize_safetensors_index_payload(
    data: Any,
    path: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": path,
        "component": _component_for_path(path),
    }
    if not isinstance(data, dict):
        summary["error"] = "Safetensors index is not a JSON object"
        return summary

    weight_map = data.get("weight_map", {})
    metadata = data.get("metadata", {})
    if not isinstance(weight_map, dict):
        weight_map = {}
    if not isinstance(metadata, dict):
        metadata = {}

    shards = sorted({str(shard) for shard in weight_map.values()})
    summary.update(
        {
            "metadata": metadata,
            "tensor_count": len(weight_map),
            "shard_count": len(shards),
            "shards": shards,
            "total_size": metadata.get("total_size"),
        }
    )
    return summary


def _validate_index_against_headers(
    index_path: Path,
    root: Path,
    weight_map: Any,
    safetensors_headers: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(weight_map, dict):
        return {"status": "error", "reason": "weight_map is not an object"}

    index_component = _component_for_path(_relative(index_path, root))
    index_dir = index_path.parent
    header_by_path = {
        str(header["path"]): header
        for header in safetensors_headers
        if "path" in header and "error" not in header
    }
    normalized_weight_map = {
        str(key): _relative(index_dir / str(shard), root)
        for key, shard in weight_map.items()
    }
    expected_shard_paths = sorted(set(normalized_weight_map.values()))
    missing_shards = [
        shard for shard in expected_shard_paths if shard not in header_by_path
    ]

    missing_tensors: list[dict[str, str]] = []
    for key, shard in sorted(normalized_weight_map.items()):
        header = header_by_path.get(shard)
        if header is None:
            continue
        tensors = set(str(tensor) for tensor in header.get("tensors", []))
        if key not in tensors:
            missing_tensors.append({"key": key, "shard": _shard_for_component_path(shard, index_component)})

    indexed_pairs = set(normalized_weight_map.items())
    extra_tensors: list[dict[str, str]] = []
    for shard in expected_shard_paths:
        header = header_by_path.get(shard)
        if header is None:
            continue
        for key in sorted(str(tensor) for tensor in header.get("tensors", [])):
            if (key, shard) not in indexed_pairs:
                extra_tensors.append(
                    {"key": key, "shard": _shard_for_component_path(shard, index_component)}
                )

    component_shards = {
        str(header["path"])
        for header in safetensors_headers
        if header.get("component") == index_component and "error" not in header
    }
    extra_shards = sorted(component_shards - set(expected_shard_paths))
    status = "ok"
    if missing_shards or missing_tensors or extra_tensors or extra_shards:
        status = "error"
    return {
        "status": status,
        "missing_shards": [
            _shard_for_component_path(shard, index_component) for shard in missing_shards
        ],
        "extra_shards": [
            _shard_for_component_path(shard, index_component) for shard in extra_shards
        ],
        "missing_tensors": missing_tensors,
        "extra_tensors": extra_tensors,
    }


def _config_facts(data: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    text_config = data.get("text_config")
    if isinstance(text_config, dict):
        facts["text_config"] = {
            key: text_config[key]
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
                "max_position_embeddings",
                "model_type",
                "rope_parameters",
                "rope_scaling",
                "rope_theta",
                "vocab_size",
            )
            if key in text_config
        }

    vision_config = data.get("vision_config")
    if isinstance(vision_config, dict):
        facts["vision_config"] = {
            key: vision_config[key]
            for key in (
                "deepstack_visual_indexes",
                "depth",
                "hidden_size",
                "num_heads",
                "out_hidden_size",
                "patch_size",
                "spatial_merge_size",
                "temporal_patch_size",
            )
            if key in vision_config
        }

    for key in (
        "attention_head_dim",
        "axes_dims_rope",
        "base_dim",
        "dim_mult",
        "image_token_id",
        "in_channels",
        "intermediate_size",
        "latents_mean",
        "latents_std",
        "num_attention_heads",
        "num_key_value_heads",
        "num_layers",
        "num_text_layers",
        "rope_theta",
        "text_hidden_dim",
        "text_intermediate_size",
        "video_token_id",
        "vision_end_token_id",
        "vision_start_token_id",
        "z_dim",
    ):
        if key in data:
            facts[key] = data[key]
    return facts


def _component_classes(data: dict[str, Any]) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, str) for item in value)
        ):
            classes[key] = value
    return classes


def _group_component_configs(
    configs: Iterable[dict[str, Any]],
    text_metadata: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {component: [] for component in EXPECTED_SOURCE_COMPONENTS}
    for summary in [*configs, *text_metadata]:
        component = str(summary.get("component", "root"))
        groups.setdefault(component, []).append(summary)
    return groups


def _download_remote_text_file(
    hf_hub_download: Any,
    *,
    repo_id: str,
    revision: str,
    filename: str,
) -> str:
    local_path = hf_hub_download(repo_id=repo_id, revision=revision, filename=filename)
    return Path(local_path).read_text(encoding="utf-8")


def _read_remote_safetensors_header(
    repo_id: str,
    revision: str,
    filename: str,
    *,
    file_size: int | None,
) -> SafetensorsHeader:
    prefix = _read_hf_byte_range(repo_id, revision, filename, 0, 7)
    if len(prefix) != 8:
        raise Krea2TurboMlxError(
            f"{filename} is too small to contain a safetensors header"
        )

    header_length = struct.unpack("<Q", prefix)[0]
    if header_length > MAX_HEADER_LENGTH:
        raise Krea2TurboMlxError(
            f"{filename} declares a safetensors header larger than {MAX_HEADER_LENGTH} bytes"
        )
    if file_size is not None and header_length > file_size - 8:
        raise Krea2TurboMlxError(
            f"{filename} declares a safetensors header longer than the file"
        )

    header_bytes = _read_hf_byte_range(
        repo_id,
        revision,
        filename,
        8,
        8 + header_length - 1,
    )
    if len(header_bytes) != header_length:
        raise Krea2TurboMlxError(
            f"{filename} ended before the declared safetensors header was complete"
        )
    return parse_safetensors_header_bytes(
        filename,
        header_bytes,
        payload_start=8 + header_length,
        file_size=file_size,
    )


def _read_hf_byte_range(
    repo_id: str,
    revision: str,
    filename: str,
    start: int,
    end: int,
) -> bytes:
    if end < start:
        return b""

    expected_length = end - start + 1
    request = Request(
        _hf_resolve_url(repo_id, revision, filename),
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": PROJECT_NAME,
        },
    )
    try:
        with urlopen(request, timeout=REMOTE_RANGE_TIMEOUT_SECONDS) as response:
            data = response.read(expected_length + 1)
    except OSError as exc:
        raise Krea2TurboMlxError(
            f"Unable to read safetensors header range {start}-{end} from {filename}: {exc}"
        ) from exc

    if len(data) != expected_length:
        raise Krea2TurboMlxError(
            f"Remote server did not return the requested byte range for {filename}"
        )
    return data


def _hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    quoted_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    quoted_revision = quote(revision, safe="")
    quoted_filename = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/{quoted_filename}"


def _should_download_remote_metadata(path: str, size: Any) -> bool:
    if path.endswith(".safetensors"):
        return False
    name = path.rsplit("/", 1)[-1]
    is_metadata = (
        path.endswith(".safetensors.index.json")
        or name in _JSON_METADATA_FILES
        or name in _TEXT_METADATA_FILES
    )
    if not is_metadata:
        return False
    if isinstance(size, int) and size > MAX_REMOTE_METADATA_BYTES:
        return False
    return True


def _summarize_remote_file(sibling: Any) -> dict[str, Any]:
    path = _get_attr(sibling, "rfilename") or _get_attr(sibling, "path")
    size = _get_attr(sibling, "size")
    lfs = _get_attr(sibling, "lfs")
    if not isinstance(lfs, dict):
        lfs = {}
    summary: dict[str, Any] = {
        "path": path,
        "size": size,
        "component": _component_for_path(str(path)),
    }
    lfs_sha256 = lfs.get("sha256") or lfs.get("oid")
    if lfs_sha256:
        summary["lfs_sha256"] = lfs_sha256
    return summary


def _component_for_path(path: str) -> str:
    if "/" not in path:
        if path == "model_index.json":
            return "model_index"
        return "root"
    component = path.split("/", 1)[0]
    if component in REMOTE_COMPONENTS:
        return component
    return "other"


def _shard_for_component_path(path: str, component: str) -> str:
    prefix = f"{component}/"
    if component != "model_index" and path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _siblings(info: Any) -> Iterable[Any]:
    siblings = _get_attr(info, "siblings")
    if siblings is None:
        return []
    return siblings


def _repo_id(info: Any, fallback: str) -> str:
    return str(_get_attr(info, "id") or _get_attr(info, "repo_id") or fallback)


def _resolved_revision(info: Any, fallback: str) -> str:
    return str(_get_attr(info, "sha") or fallback)


def _license(info: Any) -> str | None:
    card_data = _get_attr(info, "cardData") or _get_attr(info, "card_data") or {}
    value = _get_attr(card_data, "license_name") or _get_attr(card_data, "license")
    if value:
        return str(value)
    for tag in _get_attr(info, "tags") or []:
        tag_text = str(tag)
        if tag_text.startswith("license:"):
            return tag_text.split(":", 1)[1]
    return None


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _official_source_payload() -> dict[str, Any]:
    return {
        "repo_id": OFFICIAL_HF_REPO_ID,
        "revision": OFFICIAL_HF_REVISION,
        "full_precision_only": FULL_PRECISION_ONLY,
        "source_dtypes": OFFICIAL_SOURCE_DTYPES,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
