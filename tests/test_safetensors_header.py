from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.safetensors_header import (
    MAX_HEADER_LENGTH,
    parse_safetensors_header_bytes,
    read_safetensors_header,
)
from safetensors_fixtures import write_safetensors_fixture


def test_reads_dtype_shape_and_byte_count_without_optional_dependencies(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors_fixture(
        path,
        {
            "decoder.conv.weight": ("F32", [2, 3]),
            "text.embed.weight": ("BF16", [4, 5]),
        },
    )

    header = read_safetensors_header(path)

    assert header.metadata["format"] == "pt"
    assert header.payload_start > 8
    assert header.tensors["decoder.conv.weight"].dtype == "F32"
    assert header.tensors["decoder.conv.weight"].shape == (2, 3)
    assert header.tensors["decoder.conv.weight"].byte_count == 24
    assert header.tensors["text.embed.weight"].byte_count == 40


def test_rejects_invalid_headers_before_tensor_payload_use(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.safetensors"
    oversized.write_bytes(struct.pack("<Q", MAX_HEADER_LENGTH + 1))
    with pytest.raises(Krea2TurboMlxError, match="header larger"):
        read_safetensors_header(oversized)

    unknown_dtype = tmp_path / "unknown.safetensors"
    _write_raw_safetensors(
        unknown_dtype,
        {"bad.weight": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 1]}},
        b"\0",
    )
    with pytest.raises(Krea2TurboMlxError, match="unsupported dtype"):
        read_safetensors_header(unknown_dtype)

    overrun = tmp_path / "overrun.safetensors"
    _write_raw_safetensors(
        overrun,
        {"bad.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}},
        b"\0" * 4,
    )
    with pytest.raises(Krea2TurboMlxError, match="beyond"):
        read_safetensors_header(overrun)


def test_shared_header_parser_validates_remote_style_bytes() -> None:
    header = {
        "remote.weight": {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [0, 4],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    parsed = parse_safetensors_header_bytes(
        "remote/model.safetensors",
        header_bytes,
        payload_start=8 + len(header_bytes),
        file_size=8 + len(header_bytes) + 4,
    )

    assert parsed.payload_start == 8 + len(header_bytes)
    assert parsed.tensors["remote.weight"].byte_count == 4


def _write_raw_safetensors(path: Path, header: dict[str, object], payload: bytes) -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
