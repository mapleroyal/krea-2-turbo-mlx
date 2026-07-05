from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._version import __version__
from .constants import (
    DEFAULT_DISTILLED_SHIFT,
    DEFAULT_GUIDANCE_SCALE,
    PROJECT_NAME,
)
from .errors import Krea2TurboMlxError, ValidationError
from .json_io import read_json_object

PNG_METADATA_KEY = PROJECT_NAME
PNG_PARAMETERS_KEY = "parameters"

_DTYPE_LABELS = {
    "BF16": "bf16",
    "F16": "fp16",
    "F32": "fp32",
    "F64": "fp64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
}


def save_generation_png(
    image: Any,
    output: str | Path,
    *,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> None:
    try:
        from PIL.PngImagePlugin import PngInfo
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise Krea2TurboMlxError(
            "PNG metadata output requires pillow>=10. Run `./setup.sh` or install "
            "`krea-2-turbo-mlx[runtime]`."
        ) from exc

    output_path = Path(output).expanduser()
    if output_path.suffix.lower() != ".png":
        raise ValidationError("output path must end in .png")
    if output_path.exists() and not overwrite:
        raise ValidationError(
            f"output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, list):
        if not image:
            raise Krea2TurboMlxError("No image was produced")
        image = image[0]
    if not hasattr(image, "save"):
        raise Krea2TurboMlxError("PNG output requires a PIL-compatible image")

    pnginfo = PngInfo()
    pnginfo.add_itxt(PNG_METADATA_KEY, json.dumps(metadata, sort_keys=True))
    pnginfo.add_itxt(PNG_PARAMETERS_KEY, generation_parameters_text(metadata))

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp.png",
        dir=output_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path, format="PNG", pnginfo=pnginfo)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def generation_metadata_payload(
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    shift: float = DEFAULT_DISTILLED_SHIFT,
    model_path: str | Path,
    elapsed_seconds: float | None,
    truncation_warnings: tuple[dict[str, int], ...] = (),
    loras: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> dict[str, Any]:
    model = Path(model_path).expanduser()
    artifact = _artifact_payload(model)
    model_precision = artifact_precision_label(artifact)
    payload = {
        "schema_version": 1,
        "generator": PROJECT_NAME,
        "generator_version": __version__,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "model_path": str(model),
        "model_precision": model_precision,
        "artifact_fingerprint": artifact_fingerprint(model),
        "source_revision": _source_revision(artifact),
        "elapsed_seconds": elapsed_seconds,
        "prompt_truncation": [dict(item) for item in truncation_warnings],
    }
    if loras:
        payload["loras"] = [dict(item) for item in loras]
    return payload


def generation_parameters_text(payload: dict[str, Any]) -> str:
    fields = [
        f"Steps: {payload['steps']}",
        f"Seed: {payload['seed']}",
        f"Size: {payload['width']}x{payload['height']}",
        f"Guidance scale: {payload['guidance_scale']}",
        f"Shift: {payload['shift']}",
        f"Model: {payload['model_path']}",
        f"Artifact: {payload['artifact_fingerprint']}",
        f"Source revision: {payload['source_revision']}",
        f"Generator: {payload['generator']} {payload['generator_version']}",
    ]
    loras = payload.get("loras")
    if isinstance(loras, list) and loras:
        names = []
        for item in loras:
            if isinstance(item, Mapping):
                name = str(item.get("display_name") or item.get("id") or "").strip()
                scale = item.get("scale")
                if name:
                    names.append(f"{name} x{scale}")
        if names:
            fields.append("LoRAs: " + "; ".join(names))
    return f"{payload['prompt']}\n" + ", ".join(fields)


def artifact_fingerprint(model_path: str | Path) -> str:
    root = Path(model_path).expanduser()
    digest = hashlib.sha256()
    for rel_path in ("artifact.json", "conversion_report.json", "model_index.json"):
        path = root / rel_path
        if not path.is_file():
            continue
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def model_precision_label(model_path: str | Path, fallback: str = "model") -> str:
    return artifact_precision_label(
        _artifact_payload(Path(model_path).expanduser()),
        fallback=fallback,
    )


def artifact_precision_label(
    artifact: Mapping[str, Any],
    *,
    fallback: str = "model",
) -> str:
    precision = artifact.get("precision")
    if isinstance(precision, Mapping):
        label = _dtype_histogram_label(precision.get("selected_dtype_histogram"))
        if label:
            return label

    selected = artifact.get("selected")
    if isinstance(selected, Mapping):
        label = _dtype_histogram_label(selected.get("dtypes"))
        if label:
            return label

    return fallback


def _artifact_payload(model: Path) -> dict[str, Any]:
    artifact_path = model / "artifact.json"
    if not artifact_path.is_file():
        return {}
    return read_json_object(artifact_path)


def _dtype_histogram_label(value: Any) -> str | None:
    if not isinstance(value, Mapping) or not value:
        return None

    counts: list[tuple[str, int]] = []
    for dtype, count in value.items():
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            continue
        if normalized_count <= 0:
            continue
        counts.append((str(dtype).upper(), normalized_count))

    if not counts:
        return None

    dtype, _ = max(counts, key=lambda item: (item[1], item[0]))
    return _DTYPE_LABELS.get(dtype, dtype.lower())


def _source_revision(artifact: dict[str, Any]) -> str | None:
    provenance = artifact.get("provenance", {})
    if isinstance(provenance, dict) and provenance.get("source_revision") is not None:
        return str(provenance["source_revision"])
    source = artifact.get("source", {})
    if isinstance(source, dict):
        value = source.get("revision") or source.get("resolved_revision")
        if value is not None:
            return str(value)
    return None


__all__ = [
    "PNG_METADATA_KEY",
    "PNG_PARAMETERS_KEY",
    "artifact_fingerprint",
    "generation_metadata_payload",
    "generation_parameters_text",
    "save_generation_png",
]
