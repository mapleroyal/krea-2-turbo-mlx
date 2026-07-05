from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import Krea2TurboMlxError

DTYPE_BYTE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
MAX_HEADER_LENGTH = 100 * 1024 * 1024


@dataclass(frozen=True)
class TensorHeader:
    key: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def byte_count(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def expected_byte_count(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTE_SIZES[self.dtype]


@dataclass(frozen=True)
class SafetensorsHeader:
    metadata: dict[str, str]
    tensors: dict[str, TensorHeader]
    payload_start: int


def read_safetensors_header(path: Path) -> SafetensorsHeader:
    """Read only the safetensors header, not the tensor payload."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header_length_bytes = handle.read(8)
            if len(header_length_bytes) != 8:
                raise Krea2TurboMlxError(f"{path} is too small to contain a safetensors header")

            header_length = struct.unpack("<Q", header_length_bytes)[0]
            if header_length > MAX_HEADER_LENGTH:
                raise Krea2TurboMlxError(
                    f"{path} declares a safetensors header larger than {MAX_HEADER_LENGTH} bytes"
                )
            if header_length > file_size - 8:
                raise Krea2TurboMlxError(
                    f"{path} declares a safetensors header longer than the file"
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise Krea2TurboMlxError(
                    f"{path} ended before the declared safetensors header was complete"
                )
    except OSError as exc:
        raise Krea2TurboMlxError(f"Unable to read safetensors header from {path}: {exc}") from exc

    return parse_safetensors_header_bytes(
        path,
        header_bytes,
        payload_start=8 + header_length,
        file_size=file_size,
    )


def parse_safetensors_header_bytes(
    source: str | Path,
    header_bytes: bytes,
    *,
    payload_start: int,
    file_size: int | None,
) -> SafetensorsHeader:
    """Parse and validate a safetensors JSON header already read from any source."""
    try:
        raw_header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise Krea2TurboMlxError(
            f"Invalid safetensors header JSON in {source}: {exc}"
        ) from exc

    if not isinstance(raw_header, dict):
        raise Krea2TurboMlxError(f"Safetensors header in {source} is not a JSON object")

    metadata = raw_header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}

    payload_size = None if file_size is None else file_size - payload_start
    if payload_size is not None and payload_size < 0:
        raise Krea2TurboMlxError(
            f"{source} declares a safetensors header longer than the file"
        )

    tensors: dict[str, TensorHeader] = {}
    for key, value in raw_header.items():
        if key == "__metadata__":
            continue
        tensors[str(key)] = _parse_tensor_header(
            source,
            str(key),
            value,
            payload_size=payload_size,
        )

    return SafetensorsHeader(
        metadata={str(key): str(value) for key, value in metadata.items()},
        tensors=tensors,
        payload_start=payload_start,
    )


def _parse_tensor_header(
    source: str | Path,
    key: str,
    value: Any,
    *,
    payload_size: int | None,
) -> TensorHeader:
    if not isinstance(value, dict):
        raise Krea2TurboMlxError(f"Tensor {key!r} in {source} has a non-object header")

    dtype = value.get("dtype")
    shape = value.get("shape")
    data_offsets = value.get("data_offsets")
    if not isinstance(dtype, str):
        raise Krea2TurboMlxError(f"Tensor {key!r} in {source} is missing a string dtype")
    if dtype not in DTYPE_BYTE_SIZES:
        raise Krea2TurboMlxError(
            f"Tensor {key!r} in {source} has unsupported dtype {dtype!r}"
        )
    if not (
        isinstance(shape, list)
        and all(isinstance(dim, int) and dim >= 0 for dim in shape)
    ):
        raise Krea2TurboMlxError(f"Tensor {key!r} in {source} has an invalid shape")
    if not (
        isinstance(data_offsets, list)
        and len(data_offsets) == 2
        and all(isinstance(offset, int) and offset >= 0 for offset in data_offsets)
        and data_offsets[0] <= data_offsets[1]
    ):
        raise Krea2TurboMlxError(f"Tensor {key!r} in {source} has invalid data_offsets")
    if payload_size is not None and data_offsets[1] > payload_size:
        raise Krea2TurboMlxError(
            f"Tensor {key!r} in {source} points beyond the safetensors payload"
        )

    tensor = TensorHeader(
        key=key,
        dtype=dtype,
        shape=tuple(shape),
        data_offsets=(data_offsets[0], data_offsets[1]),
    )
    if tensor.byte_count != tensor.expected_byte_count:
        raise Krea2TurboMlxError(
            f"Tensor {key!r} in {source} declares {tensor.byte_count} bytes but "
            f"shape {list(tensor.shape)} and dtype {tensor.dtype} require "
            f"{tensor.expected_byte_count}"
        )
    return tensor
