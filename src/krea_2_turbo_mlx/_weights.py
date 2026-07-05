from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import Krea2TurboMlxError
from .json_io import read_json_object

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None


def flatten_parameter_shapes(parameters: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape)
        for name, value in flatten_parameters(parameters)
    }


def flatten_parameters(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_parameters(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten_parameters(child, child_prefix)
    else:
        yield prefix, value


def load_mapped_safetensors_weights(
    weights_dir: str | Path,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
    map_key: Callable[[str], str | None],
    transform_value: Callable[[str, Any], Any] | None = None,
    index_filename: str,
    single_filename: str,
    label: str,
    dtype: Any | None = None,
) -> dict[str, Any]:
    _require_mlx(label)
    selected: dict[str, Any] = {}
    unexpected: list[str] = []
    duplicates: list[str] = []

    for official_key, array in iter_safetensors_weight_arrays(
        weights_dir,
        index_filename=index_filename,
        single_filename=single_filename,
        label=label,
    ):
        try:
            local_key = map_key(official_key)
        except Krea2TurboMlxError:
            unexpected.append(official_key)
            continue
        if local_key is None:
            continue
        if local_key in selected:
            duplicates.append(official_key)
            continue
        if transform_value is not None:
            array = transform_value(local_key, array)
        if dtype is not None:
            array = array.astype(dtype)
        selected[local_key] = array

    selected_keys = set(selected)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - selected_keys)
    extra = sorted(selected_keys - expected_keys)
    if missing:
        raise Krea2TurboMlxError(
            f"{label} artifact is missing runtime tensors: " + ", ".join(missing[:10])
        )
    if unexpected or duplicates or extra:
        bad = unexpected + duplicates + extra
        raise Krea2TurboMlxError(
            f"{label} artifact has unexpected runtime tensors: "
            + ", ".join(bad[:10])
        )

    mismatched = [
        f"{key}: expected {expected_shapes[key]}, got {tuple(array.shape)}"
        for key, array in sorted(selected.items())
        if tuple(array.shape) != expected_shapes[key]
    ]
    if mismatched:
        raise Krea2TurboMlxError(
            f"{label} artifact has shape-mismatched tensors: "
            + "; ".join(mismatched[:10])
        )
    return selected


def iter_safetensors_weight_arrays(
    weights_dir: str | Path,
    *,
    index_filename: str,
    single_filename: str,
    label: str,
) -> Iterable[tuple[str, Any]]:
    _require_mlx(label)
    root = Path(weights_dir).expanduser()
    index_path = root / index_filename
    single_path = root / single_filename
    if index_path.is_file():
        index = read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise Krea2TurboMlxError(f"Invalid {label.lower()} weight index: {index_path}")
        keys_by_shard: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            keys_by_shard.setdefault(str(shard), []).append(str(key))
        for shard, keys in sorted(keys_by_shard.items()):
            shard_path = index_path.parent / shard
            if not shard_path.is_file():
                raise Krea2TurboMlxError(f"Missing {label.lower()} weight shard: {shard_path}")
            arrays = mx.load(str(shard_path))
            shard_keys = set(keys)
            missing = sorted(shard_keys - set(arrays))
            extra = sorted(set(arrays) - shard_keys)
            if missing:
                raise Krea2TurboMlxError(
                    f"{shard_path} is missing indexed tensors: " + ", ".join(missing[:10])
                )
            if extra:
                raise Krea2TurboMlxError(
                    f"{shard_path} contains tensors not listed in its index: "
                    + ", ".join(extra[:10])
                )
            for key in sorted(keys):
                yield key, arrays[key]
        return

    if single_path.is_file():
        arrays = mx.load(str(single_path))
        for key in sorted(arrays):
            yield key, arrays[key]
        return

    raise Krea2TurboMlxError(
        f"Missing {label.lower()} weights: expected {single_path} or {index_path}"
    )


def _require_mlx(label: str) -> None:
    if mx is None:
        raise Krea2TurboMlxError(
            f"{label} loading requires MLX. Install `krea-2-turbo-mlx[runtime]` "
            "on an MLX-supported machine."
        )


__all__ = [
    "flatten_parameter_shapes",
    "flatten_parameters",
    "iter_safetensors_weight_arrays",
    "load_mapped_safetensors_weights",
]
