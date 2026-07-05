from __future__ import annotations

import hashlib
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import Krea2TurboMlxError
from .safetensors_header import TensorHeader, read_safetensors_header

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None

LOCAL_LORA_SCALE_MIN = 0.0
LOCAL_LORA_SCALE_MAX = 4.0
LOCAL_LORA_DEFAULT_SCALE = 1.0

LoraSourceType = Literal["catalog", "path"]
LoraAdapterType = Literal["weight-diff", "standard", "lokr"]
_HeaderAdapterType = Literal["weight-diff", "standard", "lokr", "unsupported"]

_WEIGHT_DIFF_SUFFIXES = {
    "diff": "diff",
}
_STANDARD_SUFFIXES = {
    "lora_A.weight": "down",
    "lora_A": "down",
    "lora_down.weight": "down",
    "lora_down": "down",
    "lora_B.weight": "up",
    "lora_B": "up",
    "lora_up.weight": "up",
    "lora_up": "up",
    "alpha": "alpha",
    "lora_alpha": "alpha",
}
_LOKR_SUFFIXES = {
    "lokr_w1": "w1",
    "lokr_w1.weight": "w1",
    "lokr_w1_a": "w1_a",
    "lokr_w1_a.weight": "w1_a",
    "lokr_w1_b": "w1_b",
    "lokr_w1_b.weight": "w1_b",
    "lokr_w2": "w2",
    "lokr_w2.weight": "w2",
    "lokr_w2_a": "w2_a",
    "lokr_w2_a.weight": "w2_a",
    "lokr_w2_b": "w2_b",
    "lokr_w2_b.weight": "w2_b",
    "alpha": "alpha",
    "lora_alpha": "alpha",
}
_UNSUPPORTED_SUFFIXES = {
    "dora_scale": "DoRA",
    "hada_w1_a": "LoHa",
    "hada_w1_b": "LoHa",
    "hada_w2_a": "LoHa",
    "hada_w2_b": "LoHa",
    "lokr_t1": "Tucker LoKr",
    "lokr_t2": "Tucker LoKr",
}
_ALL_SUFFIXES = tuple(
    sorted(
        set(_WEIGHT_DIFF_SUFFIXES)
        | set(_STANDARD_SUFFIXES)
        | set(_LOKR_SUFFIXES)
        | set(_UNSUPPORTED_SUFFIXES),
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class LoraReference:
    id: str
    scale: float | None = None

    def to_mapping(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {"id": self.id}
        payload["scale"] = (
            default_lora_scale(self.id) if self.scale is None else float(self.scale)
        )
        return payload


@dataclass(frozen=True)
class LoraCatalogItem:
    id: str
    display_name: str
    source_type: LoraSourceType
    adapter_type: str
    path: str | None
    default_scale: float
    scale_min: float
    scale_max: float
    target_count: int
    skipped_count: int = 0
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "display_name": self.display_name,
            "source_type": self.source_type,
            "adapter_type": self.adapter_type,
            "default_scale": self.default_scale,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "target_count": self.target_count,
            "skipped_count": self.skipped_count,
            "warnings": list(self.warnings),
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class LoraCatalog:
    items: tuple[LoraCatalogItem, ...]
    warnings: tuple[str, ...]
    scanned_at_ms: int
    lora_dir: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "dir": self.lora_dir,
            "items": [item.to_mapping() for item in self.items],
            "warnings": list(self.warnings),
            "scanned_at_ms": self.scanned_at_ms,
        }


@dataclass(frozen=True)
class ResolvedLoraTarget:
    adapter_type: LoraAdapterType
    target: str
    source_key: str
    scale: float
    tensors: Mapping[str, Any]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedLoraPatch:
    id: str
    display_name: str
    scale: float
    source_type: LoraSourceType
    adapter_type: str
    targets: tuple[ResolvedLoraTarget, ...]
    sha256: str
    path: str | None = None
    skipped_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def patch_hash(self) -> str:
        return self.sha256

    @property
    def target(self) -> str:
        return self.targets[0].target if self.targets else ""

    @property
    def source_key(self) -> str:
        return self.targets[0].source_key if self.targets else ""

    @property
    def shape(self) -> tuple[int, ...]:
        return self.targets[0].shape if self.targets else ()

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "display_name": self.display_name,
            "scale": self.scale,
            "source_type": self.source_type,
            "adapter_type": self.adapter_type,
            "target_count": len(self.targets),
            "skipped_count": self.skipped_count,
            "sha256": self.sha256,
            "patch_hash": self.sha256,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def default_lora_scale(lora_id: Any) -> float:
    return LOCAL_LORA_DEFAULT_SCALE


def lora_scale_bounds(lora_id: Any) -> tuple[float, float]:
    return (LOCAL_LORA_SCALE_MIN, LOCAL_LORA_SCALE_MAX)


def parse_lora_scale(
    value: Any,
    *,
    lora_id: Any = None,
    clamp: bool = False,
) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise Krea2TurboMlxError("LoRA scale must be a number.") from exc
    if not math.isfinite(scale):
        raise Krea2TurboMlxError("LoRA scale must be finite.")

    minimum, maximum = lora_scale_bounds(lora_id)
    if clamp:
        return max(minimum, min(maximum, scale))
    if not minimum <= scale <= maximum:
        raise Krea2TurboMlxError(
            f"LoRA scale for {str(lora_id)!r} must be from {minimum:g} to {maximum:g}."
        )
    return scale


def normalize_lora_payload(
    value: Any,
    *,
    clamp_scale: bool = False,
) -> tuple[LoraReference, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Krea2TurboMlxError("loras must be an array.")

    references: list[LoraReference] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise Krea2TurboMlxError(f"LoRA {index} must be a JSON object.")
        unsupported = sorted(set(item) - {"id", "scale"})
        if unsupported:
            label = "field" if len(unsupported) == 1 else "fields"
            raise Krea2TurboMlxError(
                f"LoRA {index} has unsupported {label}: {', '.join(unsupported)}."
            )
        lora_id = _normalized_lora_payload_id(item.get("id"), index=index)
        scale = parse_lora_scale(
            item.get("scale", default_lora_scale(lora_id)),
            lora_id=lora_id,
            clamp=clamp_scale,
        )
        references.append(LoraReference(id=lora_id, scale=scale))

    return tuple(references)


def validate_lora_spec_syntax(specs: Sequence[str] | None) -> None:
    for spec in specs or ():
        source, scale_text = _parse_lora_string_spec(spec)
        if scale_text is not None:
            parse_lora_scale(scale_text, lora_id=source)


def scan_lora_catalog(lora_dir: str | Path) -> LoraCatalog:
    root = Path(lora_dir).expanduser()
    items = []
    warnings: list[str] = []
    if root.exists() and not root.is_dir():
        warnings.append(f"LoRA path is not a directory: {root}")
    elif root.is_dir():
        for path in sorted(root.rglob("*.safetensors")):
            try:
                item = _catalog_item_from_header(path, root)
            except Krea2TurboMlxError as exc:
                warnings.append(f"{_display_path(path, root)}: {exc}")
                continue
            if item.target_count <= 0:
                warnings.extend(
                    item.warnings
                    or (f"{item.id}: no supported Krea 2 Turbo LoRA targets found.",)
                )
                continue
            items.append(item)

    return LoraCatalog(
        items=tuple(items),
        warnings=tuple(warnings),
        scanned_at_ms=int(time.time() * 1000),
        lora_dir=str(root),
    )


def resolve_lora_patches(
    specs: Sequence[str | Mapping[str, Any] | LoraReference | ResolvedLoraPatch] | None,
    *,
    transformer: Any | None = None,
    lora_dir: str | Path | None = None,
) -> tuple[ResolvedLoraPatch, ...]:
    if not specs:
        return ()

    patches: list[ResolvedLoraPatch] = []
    for spec in specs:
        if isinstance(spec, ResolvedLoraPatch):
            patch = spec
            _validate_resolved_patch(patch, transformer)
        else:
            patch = _resolve_lora_spec(
                spec,
                transformer=transformer,
                lora_dir=lora_dir,
            )
        patches.append(patch)
    return tuple(patches)


def lora_metadata(patches: Iterable[ResolvedLoraPatch]) -> list[dict[str, object]]:
    return [patch.metadata() for patch in patches]


@contextmanager
def applied_lora_patches(
    transformer: Any,
    patches: Sequence[ResolvedLoraPatch] | None,
) -> Iterable[None]:
    if not patches:
        yield
        return

    originals: list[tuple[Any, str, Any]] = []
    targets_by_module: dict[str, list[_PreparedTarget]] = defaultdict(list)
    try:
        for patch in patches:
            for target in patch.targets:
                module = _target_value(transformer, target.target)
                dtype = getattr(getattr(module, "weight", None), "dtype", None)
                targets_by_module[target.target].append(
                    _prepare_runtime_target(
                        target,
                        scale=float(patch.scale) * float(target.scale),
                        dtype=dtype,
                    )
                )

        for target_path, prepared_targets in targets_by_module.items():
            parent, attr = _target_parent(transformer, target_path)
            base = _get_child(parent, attr)
            originals.append((parent, attr, base))
            _set_child(parent, attr, _LoraLinearWrapper(base, tuple(prepared_targets)))
        yield
    finally:
        for parent, attr, original in reversed(originals):
            _set_child(parent, attr, original)


def map_lora_target_key(source_key: str) -> str | None:
    parts = [part for part in str(source_key).split(".") if part]
    if not parts:
        return None

    parts = _strip_supported_prefixes(parts)
    if parts[:1] == ["diffusion_model"]:
        parts = parts[1:]
        if len(parts) >= 2 and parts[0] == "blocks":
            parts = ["transformer_blocks", parts[1], *parts[2:]]
        elif len(parts) >= 2 and parts[0] == "txtfusion":
            if parts[1] == "projector":
                parts = ["text_fusion", "projector", *parts[2:]]
            elif len(parts) >= 3 and parts[1] == "layerwise_blocks":
                parts = ["text_fusion", "layerwise_blocks", parts[2], *parts[3:]]
            elif len(parts) >= 3 and parts[1] == "refiner_blocks":
                parts = ["text_fusion", "refiner_blocks", parts[2], *parts[3:]]
            else:
                return None
        else:
            return None

    mapped: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "attn" and index + 1 < len(parts):
            next_part = parts[index + 1]
            attn = {
                "wq": ["attn", "to_q"],
                "wk": ["attn", "to_k"],
                "wv": ["attn", "to_v"],
                "wo": ["attn", "to_out", "0"],
                "gate": ["attn", "to_gate"],
            }.get(next_part)
            if attn is not None:
                mapped.extend(attn)
                index += 2
                continue
        if part == "mlp" and index + 1 < len(parts):
            next_part = parts[index + 1]
            ff = {
                "gate": ["ff", "gate"],
                "up": ["ff", "up"],
                "down": ["ff", "down"],
            }.get(next_part)
            if ff is not None:
                mapped.extend(ff)
                index += 2
                continue
        mapped.append(part)
        index += 1

    return ".".join(mapped)


def _resolve_lora_spec(
    spec: str | Mapping[str, Any] | LoraReference,
    *,
    transformer: Any | None,
    lora_dir: str | Path | None,
) -> ResolvedLoraPatch:
    if isinstance(spec, LoraReference):
        source = spec.id
        scale = (
            default_lora_scale(spec.id) if spec.scale is None else float(spec.scale)
        )
    elif isinstance(spec, Mapping):
        reference = normalize_lora_payload([spec], clamp_scale=False)[0]
        source = reference.id
        scale = (
            default_lora_scale(reference.id)
            if reference.scale is None
            else float(reference.scale)
        )
    elif isinstance(spec, str):
        source, scale_text = _parse_lora_string_spec(spec)
        scale = parse_lora_scale(
            default_lora_scale(source) if scale_text is None else scale_text,
            lora_id=source,
        )
    else:
        raise Krea2TurboMlxError("LoRA specs must be strings or objects.")

    path, lora_id, source_type = _resolve_local_lora_source(source, lora_dir)
    return _load_local_lora(
        path,
        lora_id=lora_id,
        source_type=source_type,
        scale=parse_lora_scale(scale, lora_id=lora_id),
        transformer=transformer,
    )


def _load_local_lora(
    path: Path,
    *,
    lora_id: str,
    source_type: LoraSourceType,
    scale: float,
    transformer: Any | None,
) -> ResolvedLoraPatch:
    header = read_safetensors_header(path)
    groups, warnings = _group_lora_tensors(header.tensors)
    loaded: list[ResolvedLoraTarget] = []
    skipped = 0

    for source_target, group in sorted(groups.items()):
        try:
            target = _load_lora_group(
                path,
                header.payload_start,
                header.tensors,
                header.metadata,
                source_target,
                group,
                transformer=transformer,
            )
        except Krea2TurboMlxError as exc:
            skipped += 1
            warnings.append(f"{source_target}: {exc}")
            continue
        loaded.append(target)

    if not loaded:
        detail = "; ".join(warnings[:5])
        raise Krea2TurboMlxError(
            f"LoRA {path} has no supported Krea 2 Turbo Linear targets"
            + (f": {detail}" if detail else ".")
        )

    adapter_types = sorted({target.adapter_type for target in loaded})
    return ResolvedLoraPatch(
        id=lora_id,
        display_name=path.stem,
        scale=parse_lora_scale(scale, lora_id=lora_id),
        source_type=source_type,
        adapter_type=adapter_types[0] if len(adapter_types) == 1 else "mixed",
        targets=tuple(loaded),
        sha256=_sha256_file(path),
        path=str(path),
        skipped_count=skipped,
        warnings=tuple(warnings),
    )


def _load_lora_group(
    path: Path,
    payload_start: int,
    tensors: Mapping[str, TensorHeader],
    metadata: Mapping[str, str],
    source_target: str,
    group: Mapping[str, str],
    *,
    transformer: Any | None,
) -> ResolvedLoraTarget:
    unsupported = [name for name in group if name.startswith("unsupported:")]
    if unsupported:
        raise Krea2TurboMlxError(f"{unsupported[0].split(':', 1)[1]} is not supported.")
    has_lokr = any(name in group for name in ("w1", "w1_a", "w1_b", "w2", "w2_a", "w2_b"))
    has_standard = any(name in group for name in ("down", "up"))
    if has_lokr and has_standard:
        raise Krea2TurboMlxError("mixed standard LoRA and LoKr tensors are not supported.")
    if "diff" in group:
        if has_lokr or has_standard or set(group) != {"diff"}:
            raise Krea2TurboMlxError(
                "weight-diff adapters must contain only a diff tensor."
            )
        return _load_weight_diff_group(
            path,
            payload_start,
            tensors,
            source_target,
            group,
            transformer=transformer,
        )
    if has_lokr:
        return _load_lokr_group(
            path,
            payload_start,
            tensors,
            metadata,
            source_target,
            group,
            transformer=transformer,
        )
    if has_standard:
        return _load_standard_lora_group(
            path,
            payload_start,
            tensors,
            metadata,
            source_target,
            group,
            transformer=transformer,
        )
    raise Krea2TurboMlxError("no supported adapter tensor pair found.")


def _load_weight_diff_group(
    path: Path,
    payload_start: int,
    tensors: Mapping[str, TensorHeader],
    source_target: str,
    group: Mapping[str, str],
    *,
    transformer: Any | None,
) -> ResolvedLoraTarget:
    diff_key = group["diff"]
    diff = _read_tensor_array(path, payload_start, tensors[diff_key])
    if diff.ndim != 2:
        raise Krea2TurboMlxError("weight-diff tensor must be 2D.")
    target = _runtime_target_or_raise(source_target)
    expected_shape = tuple(int(dim) for dim in diff.shape)
    _validate_target_module_shape(transformer, target, expected_shape)  # type: ignore[arg-type]
    return ResolvedLoraTarget(
        adapter_type="weight-diff",
        target=target,
        source_key=diff_key,
        scale=1.0,
        tensors={"diff": diff},
        shape=expected_shape,
    )


def _load_standard_lora_group(
    path: Path,
    payload_start: int,
    tensors: Mapping[str, TensorHeader],
    metadata: Mapping[str, str],
    source_target: str,
    group: Mapping[str, str],
    *,
    transformer: Any | None,
) -> ResolvedLoraTarget:
    if "down" not in group or "up" not in group:
        raise Krea2TurboMlxError("standard LoRA requires down and up tensors.")
    down = _read_tensor_array(path, payload_start, tensors[group["down"]])
    up = _read_tensor_array(path, payload_start, tensors[group["up"]])
    if down.ndim != 2 or up.ndim != 2:
        raise Krea2TurboMlxError("conv LoRA tensors are not supported.")
    if down.shape[0] != up.shape[1]:
        raise Krea2TurboMlxError(
            f"rank mismatch between down {tuple(down.shape)} and up {tuple(up.shape)}."
        )
    rank = int(down.shape[0])
    if rank <= 0:
        raise Krea2TurboMlxError("standard LoRA rank must be positive.")
    alpha = _group_alpha(
        path,
        payload_start,
        tensors,
        metadata,
        group,
        rank=rank,
    )
    target = _runtime_target_or_raise(source_target)
    expected_shape = (int(up.shape[0]), int(down.shape[1]))
    _validate_target_module_shape(transformer, target, expected_shape)
    return ResolvedLoraTarget(
        adapter_type="standard",
        target=target,
        source_key=source_target,
        scale=float(alpha) / float(rank),
        tensors={"down": down, "up": up},
        shape=expected_shape,
    )


def _load_lokr_group(
    path: Path,
    payload_start: int,
    tensors: Mapping[str, TensorHeader],
    metadata: Mapping[str, str],
    source_target: str,
    group: Mapping[str, str],
    *,
    transformer: Any | None,
) -> ResolvedLoraTarget:
    for unsupported_name, label in (("dora_scale", "DoRA"), ("t1", "Tucker LoKr"), ("t2", "Tucker LoKr")):
        if unsupported_name in group:
            raise Krea2TurboMlxError(f"{label} is not supported.")

    loaded: dict[str, Any] = {}
    for name in ("w1", "w1_a", "w1_b", "w2", "w2_a", "w2_b"):
        if name in group:
            loaded[name] = _read_tensor_array(path, payload_start, tensors[group[name]])
            if loaded[name].ndim != 2:
                raise Krea2TurboMlxError("conv LoKr tensors are not supported.")

    use_w1 = "w1" in loaded
    use_w2 = "w2" in loaded
    if not use_w1 and not {"w1_a", "w1_b"} <= set(loaded):
        raise Krea2TurboMlxError("LoKr requires w1 or w1_a/w1_b tensors.")
    if not use_w2 and not {"w2_a", "w2_b"} <= set(loaded):
        raise Krea2TurboMlxError("LoKr requires w2 or w2_a/w2_b tensors.")

    if use_w1:
        w1_shape = tuple(int(dim) for dim in loaded["w1"].shape)
    else:
        w1_a = loaded["w1_a"]
        w1_b = loaded["w1_b"]
        if w1_a.shape[1] != w1_b.shape[0]:
            raise Krea2TurboMlxError("LoKr w1_a/w1_b rank mismatch.")
        w1_shape = (int(w1_a.shape[0]), int(w1_b.shape[1]))

    if use_w2:
        w2_shape = tuple(int(dim) for dim in loaded["w2"].shape)
    else:
        w2_a = loaded["w2_a"]
        w2_b = loaded["w2_b"]
        if w2_a.shape[1] != w2_b.shape[0]:
            raise Krea2TurboMlxError("LoKr w2_a/w2_b rank mismatch.")
        w2_shape = (int(w2_a.shape[0]), int(w2_b.shape[1]))

    expected_shape = (w1_shape[0] * w2_shape[0], w1_shape[1] * w2_shape[1])
    direct_full_matrix = use_w1 and use_w2
    rank = _lokr_rank(loaded, use_w1=use_w1, use_w2=use_w2)
    alpha = rank if direct_full_matrix else _group_alpha(
        path,
        payload_start,
        tensors,
        metadata,
        group,
        rank=rank,
    )
    target = _runtime_target_or_raise(source_target)
    _validate_target_module_shape(transformer, target, expected_shape)
    return ResolvedLoraTarget(
        adapter_type="lokr",
        target=target,
        source_key=source_target,
        scale=1.0 if direct_full_matrix else float(alpha) / float(rank),
        tensors=loaded,
        shape=expected_shape,
    )


def _group_alpha(
    path: Path,
    payload_start: int,
    tensors: Mapping[str, TensorHeader],
    metadata: Mapping[str, str],
    group: Mapping[str, str],
    *,
    rank: int,
) -> float:
    if "alpha" in group:
        alpha = _read_tensor_array(path, payload_start, tensors[group["alpha"]])
        if alpha.size != 1:
            raise Krea2TurboMlxError("LoRA alpha tensor must be scalar.")
        return float(alpha.reshape(-1)[0])
    for key in ("alpha", "lora_alpha", "ss_network_alpha"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return float(rank)


def _lokr_rank(loaded: Mapping[str, Any], *, use_w1: bool, use_w2: bool) -> int:
    if not use_w1:
        return int(loaded["w1_b"].shape[0])
    if not use_w2:
        return int(loaded["w2_b"].shape[0])
    return 1


def _catalog_item_from_header(path: Path, root: Path) -> LoraCatalogItem:
    header = read_safetensors_header(path)
    groups, warnings = _group_lora_tensors(header.tensors)
    target_count = 0
    skipped = 0
    adapter_types: set[_HeaderAdapterType] = set()
    for source_target, group in groups.items():
        adapter_type, error = _catalog_adapter_type(group)
        if error is not None:
            skipped += 1
            warnings.append(f"{source_target}: {error}")
            continue

        target = map_lora_target_key(source_target)
        if target is None:
            skipped += 1
            warnings.append(f"{source_target}: no Krea 2 Turbo runtime target.")
            continue
        adapter_types.add(adapter_type)
        target_count += 1

    adapter_type = (
        "mixed"
        if len(adapter_types - {"unsupported"}) > 1
        else next(iter(adapter_types - {"unsupported"}), "unsupported")
    )
    return LoraCatalogItem(
        id=_display_path(path, root),
        display_name=path.stem,
        source_type="catalog",
        adapter_type=adapter_type,
        path=str(path),
        default_scale=LOCAL_LORA_DEFAULT_SCALE,
        scale_min=LOCAL_LORA_SCALE_MIN,
        scale_max=LOCAL_LORA_SCALE_MAX,
        target_count=target_count,
        skipped_count=skipped,
        warnings=tuple(warnings),
    )


def _catalog_adapter_type(
    group: Mapping[str, str],
) -> tuple[_HeaderAdapterType, str | None]:
    unsupported = [name for name in group if name.startswith("unsupported:")]
    if unsupported:
        return "unsupported", f"{unsupported[0].split(':', 1)[1]} is not supported."

    names = set(group)
    has_weight_diff = "diff" in names
    has_lokr = any(
        name in names for name in ("w1", "w1_a", "w1_b", "w2", "w2_a", "w2_b")
    )
    has_standard = any(name in group for name in ("down", "up"))
    if has_weight_diff and (has_lokr or has_standard or names != {"diff"}):
        return "unsupported", "weight-diff adapters must contain only a diff tensor."
    if has_weight_diff:
        return "weight-diff", None
    if has_lokr and has_standard:
        return "unsupported", "mixed standard LoRA and LoKr tensors."
    if has_standard:
        if "down" not in group or "up" not in group:
            return "unsupported", "standard LoRA requires down and up tensors."
        return "standard", None
    if has_lokr:
        if "w1" not in names and not {"w1_a", "w1_b"} <= names:
            return "unsupported", "LoKr requires w1 or w1_a/w1_b tensors."
        if "w2" not in names and not {"w2_a", "w2_b"} <= names:
            return "unsupported", "LoKr requires w2 or w2_a/w2_b tensors."
        return "lokr", None
    return "unsupported", "no supported adapter tensor pair found."


def _group_lora_tensors(
    tensors: Mapping[str, TensorHeader],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    warnings: list[str] = []
    for key in tensors:
        split = _split_adapter_key(key)
        if split is None:
            continue
        source_target, name = split
        if name in groups[source_target]:
            warnings.append(f"{source_target}: duplicate tensor role {name}.")
            continue
        groups[source_target][name] = key
    return dict(groups), warnings


def _split_adapter_key(key: str) -> tuple[str, str] | None:
    for suffix in _ALL_SUFFIXES:
        dotted = f".{suffix}"
        if not key.endswith(dotted):
            continue
        source_target = key[: -len(dotted)]
        if not source_target:
            return None
        if suffix in _UNSUPPORTED_SUFFIXES:
            return source_target, f"unsupported:{_UNSUPPORTED_SUFFIXES[suffix]}"
        if suffix in _WEIGHT_DIFF_SUFFIXES:
            return source_target, _WEIGHT_DIFF_SUFFIXES[suffix]
        if suffix in _LOKR_SUFFIXES and (
            ".lokr_" in key or suffix.startswith("lokr_")
        ):
            return source_target, _LOKR_SUFFIXES[suffix]
        if suffix in _STANDARD_SUFFIXES:
            return source_target, _STANDARD_SUFFIXES[suffix]
    return None


def _runtime_target_or_raise(source_target: str) -> str:
    target = map_lora_target_key(source_target)
    if target is None:
        raise Krea2TurboMlxError("no Krea 2 Turbo runtime target.")
    return target


def _validate_target_module_shape(
    transformer: Any | None,
    target: str,
    expected_shape: tuple[int, int],
) -> None:
    if transformer is None:
        return
    module = _target_value(transformer, target)
    actual_shape = _linear_weight_shape(module)
    if actual_shape != expected_shape:
        raise Krea2TurboMlxError(
            f"target {target} has shape {actual_shape}; LoRA expects {expected_shape}."
        )


def _validate_resolved_patch(
    patch: ResolvedLoraPatch,
    transformer: Any | None,
) -> None:
    if transformer is None:
        return
    for target in patch.targets:
        _validate_target_module_shape(transformer, target.target, target.shape)  # type: ignore[arg-type]


def _resolve_local_lora_source(
    source: str,
    lora_dir: str | Path | None,
) -> tuple[Path, str, LoraSourceType]:
    # Trusted local LoRA inputs: direct paths and symlinks inside the catalog are
    # allowed for pre-release use; stronger symlink containment is deferred.
    path = Path(source).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise Krea2TurboMlxError(f"LoRA file must end in .safetensors: {path}")
        return path, path.name, "path"
    if path.is_absolute():
        if not path.is_file():
            raise Krea2TurboMlxError(f"LoRA file not found: {path}")

    if lora_dir is not None:
        root = Path(lora_dir).expanduser()
        candidates = [root / source]
        if Path(source).suffix.lower() != ".safetensors":
            candidates.append(root / f"{source}.safetensors")
        for candidate in candidates:
            if not _path_is_inside_directory(candidate, root):
                continue
            if candidate.is_file() and candidate.suffix.lower() == ".safetensors":
                return candidate, _display_path(candidate, root), "catalog"

    raise Krea2TurboMlxError(
        f"Unknown LoRA {source!r}; expected a catalog id or a .safetensors path."
    )


def _parse_lora_string_spec(value: str) -> tuple[str, str | None]:
    text = str(value).strip()
    if not text:
        raise Krea2TurboMlxError("LoRA spec must not be empty.")
    source = text
    scale_text = None
    if ":" in text:
        source, scale_text = text.rsplit(":", 1)
        source = source.strip()
        scale_text = scale_text.strip()
        if not source or not scale_text:
            raise Krea2TurboMlxError("LoRA spec must use SOURCE[:SCALE].")
    if not source:
        raise Krea2TurboMlxError("LoRA spec must include a source.")
    return source, scale_text


def _normalized_lora_payload_id(value: Any, *, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise Krea2TurboMlxError(f"LoRA {index} has an unknown id.")
    if Path(text).expanduser().is_absolute():
        raise Krea2TurboMlxError(f"LoRA {index} must use a catalog id, not an absolute path.")
    return text


def _read_tensor_array(
    path: Path,
    payload_start: int,
    tensor: TensorHeader,
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime installs NumPy.
        raise Krea2TurboMlxError("LoRA tensor loading requires NumPy.") from exc

    try:
        with path.open("rb") as handle:
            handle.seek(payload_start + tensor.data_offsets[0])
            data = handle.read(tensor.byte_count)
    except OSError as exc:
        raise Krea2TurboMlxError(
            f"Unable to read LoRA tensor payload from {path}: {exc}"
        ) from exc
    if len(data) != tensor.byte_count:
        raise Krea2TurboMlxError(f"LoRA tensor payload is truncated: {path}")

    if tensor.dtype == "F32":
        array = np.frombuffer(data, dtype="<f4").copy()
    elif tensor.dtype == "F16":
        array = np.frombuffer(data, dtype="<f2").astype(np.float32)
    elif tensor.dtype == "BF16":
        raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
        array = (raw << 16).view(np.float32)
    else:
        raise Krea2TurboMlxError(
            f"LoRA tensor {tensor.key!r} must use BF16, F16, or F32 dtype."
        )
    return array.reshape(tensor.shape)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Krea2TurboMlxError(f"Unable to hash LoRA file {path}: {exc}") from exc
    return digest.hexdigest()


def _strip_supported_prefixes(parts: list[str]) -> list[str]:
    changed = True
    while changed:
        changed = False
        if parts[:1] == ["transformer"]:
            parts = parts[1:]
            changed = True
        if parts[:2] == ["base_model", "model"]:
            parts = parts[2:]
            changed = True
    return parts


def _display_path(path: Path, root: Path) -> str:
    try:
        return _absolute_path_without_symlink_resolution(path).relative_to(
            _absolute_path_without_symlink_resolution(root)
        ).as_posix()
    except (OSError, ValueError):
        return path.name


def _path_is_inside_directory(path: Path, root: Path) -> bool:
    try:
        _absolute_path_without_symlink_resolution(path).relative_to(
            _absolute_path_without_symlink_resolution(root)
        )
    except (OSError, ValueError):
        return False
    return True


def _absolute_path_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _target_value(root: Any, target: str) -> Any:
    parent, attr = _target_parent(root, target)
    return _get_child(parent, attr)


def _target_parent(root: Any, target: str) -> tuple[Any, str]:
    parts = target.split(".")
    current = root
    for part in parts[:-1]:
        try:
            current = _get_child(current, part)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise Krea2TurboMlxError(f"Transformer target is missing {target}.") from exc
    return current, parts[-1]


def _get_child(parent: Any, attr: str) -> Any:
    if isinstance(parent, (list, tuple)) and attr.isdigit():
        return parent[int(attr)]
    return getattr(parent, attr)


def _set_child(parent: Any, attr: str, value: Any) -> None:
    if isinstance(parent, list) and attr.isdigit():
        parent[int(attr)] = value
        return
    setattr(parent, attr, value)


def _linear_weight_shape(module: Any) -> tuple[int, int]:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        raise Krea2TurboMlxError("LoRA target is not a Linear module with a weight.")
    shape_tuple = tuple(int(dim) for dim in shape)
    if len(shape_tuple) != 2:
        raise Krea2TurboMlxError(f"LoRA target weight must be 2D, got {shape_tuple}.")
    return shape_tuple  # type: ignore[return-value]


@dataclass(frozen=True)
class _PreparedTarget:
    adapter_type: LoraAdapterType
    scale: float
    tensors: Mapping[str, Any]


def _prepare_runtime_target(
    target: ResolvedLoraTarget,
    *,
    scale: float,
    dtype: Any | None,
) -> _PreparedTarget:
    use_mx = mx is not None and not _is_numpy_dtype(dtype)
    return _PreparedTarget(
        adapter_type=target.adapter_type,
        scale=scale,
        tensors={
            name: _array_to_backend(value, dtype=dtype, use_mx=use_mx)
            for name, value in target.tensors.items()
        },
    )


def _array_to_backend(value: Any, *, dtype: Any | None, use_mx: bool) -> Any:
    if use_mx:
        array = mx.array(value)
        return array.astype(dtype) if dtype is not None else array
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime installs NumPy.
        raise Krea2TurboMlxError("LoRA patch application requires NumPy.") from exc
    return np.array(value, dtype=dtype if _is_numpy_dtype(dtype) else np.float32)


class _LoraLinearWrapper:
    def __init__(self, base: Any, targets: tuple[_PreparedTarget, ...]) -> None:
        self._base = base
        self._targets = targets

    @property
    def weight(self) -> Any:
        return getattr(self._base, "weight")

    @property
    def bias(self) -> Any:
        return getattr(self._base, "bias", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        out = self._base(x, *args, **kwargs)
        for target in self._targets:
            out = out + _target_delta(target, x)
        return out


def _target_delta(target: _PreparedTarget, x: Any) -> Any:
    if target.adapter_type == "weight-diff":
        delta = _linear(x, target.tensors["diff"])
    elif target.adapter_type == "standard":
        hidden = _linear(x, target.tensors["down"])
        delta = _linear(hidden, target.tensors["up"])
    elif target.adapter_type == "lokr":
        delta = _lokr_bypass_delta(x, target.tensors)
    else:  # pragma: no cover - dataclass type guard.
        raise Krea2TurboMlxError(f"Unsupported LoRA adapter type: {target.adapter_type}")
    return delta * target.scale


def _lokr_bypass_delta(x: Any, tensors: Mapping[str, Any]) -> Any:
    use_w1 = "w1" in tensors
    use_w2 = "w2" in tensors
    c = tensors["w1"] if use_w1 else _matmul(tensors["w1_a"], tensors["w1_b"])
    uq = int(c.shape[1])
    h_in_group = _reshape(x, (*x.shape[:-1], uq, -1))
    if use_w2:
        hb = _linear(h_in_group, tensors["w2"])
    else:
        ha = _linear(h_in_group, tensors["w2_b"])
        hb = _linear(ha, tensors["w2_a"])
    h_cross_group = _swap_last_two(hb)
    hc = _linear(h_cross_group, c)
    hc = _swap_last_two(hc)
    return _reshape(hc, (*hc.shape[:-2], -1))


def _linear(x: Any, weight: Any) -> Any:
    return _matmul(x, _swap_last_two(weight))


def _matmul(left: Any, right: Any) -> Any:
    if mx is not None and _is_mlx_array(left):
        return mx.matmul(left, right)
    return left @ right


def _reshape(value: Any, shape: tuple[int, ...]) -> Any:
    return value.reshape(shape)


def _swap_last_two(value: Any) -> Any:
    if len(value.shape) < 2:
        return value
    if mx is not None and _is_mlx_array(value):
        axes = list(range(len(value.shape)))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return value.transpose(*axes)
    return value.swapaxes(-1, -2)


def _is_mlx_array(value: Any) -> bool:
    return type(value).__module__.startswith("mlx.")


def _is_numpy_dtype(dtype: Any | None) -> bool:
    return dtype is not None and type(dtype).__module__.startswith("numpy")


__all__ = [
    "LOCAL_LORA_DEFAULT_SCALE",
    "LOCAL_LORA_SCALE_MAX",
    "LOCAL_LORA_SCALE_MIN",
    "LoraCatalog",
    "LoraCatalogItem",
    "LoraReference",
    "ResolvedLoraPatch",
    "ResolvedLoraTarget",
    "applied_lora_patches",
    "default_lora_scale",
    "lora_metadata",
    "lora_scale_bounds",
    "map_lora_target_key",
    "normalize_lora_payload",
    "parse_lora_scale",
    "resolve_lora_patches",
    "scan_lora_catalog",
    "validate_lora_spec_syntax",
]
