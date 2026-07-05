from __future__ import annotations

from pathlib import Path

import pytest

from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.manifest import generate_manifest
from krea_2_turbo_mlx.safetensors_copy import write_selected_safetensors
from krea_2_turbo_mlx.safetensors_header import read_safetensors_header
from krea_2_turbo_mlx.tensor_selection import select_manifest_tensors
from safetensors_fixtures import write_safetensors_fixture


def test_copy_writes_verbatim_transformer_and_selective_text_vae(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write_source_metadata(source)
    write_safetensors_fixture(
        source / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors",
        {"transformer_blocks.0.weight": ("F32", [1])},
        payloads={"transformer_blocks.0.weight": b"TRNS"},
    )
    write_safetensors_fixture(
        source / "text_encoder" / "model.safetensors",
        {
            "language_model.layers.0.weight": ("BF16", [2]),
            "visual.patch_embed.weight": ("BF16", [2]),
        },
        payloads={
            "language_model.layers.0.weight": b"LANG",
            "visual.patch_embed.weight": b"VISN",
        },
    )
    write_safetensors_fixture(
        source / "vae" / "diffusion_pytorch_model.safetensors",
        {
            "decoder.conv.weight": ("F32", [1]),
            "encoder.conv.weight": ("F32", [1]),
        },
        payloads={
            "decoder.conv.weight": b"DECO",
            "encoder.conv.weight": b"ENCO",
        },
    )

    selection = select_manifest_tensors(generate_manifest(source))
    summary = write_selected_safetensors(
        source_root=source,
        output_root=output,
        decisions=selection["decisions"],
    )

    transformer_rel = "transformer/diffusion_pytorch_model-00001-of-00001.safetensors"
    assert (output / transformer_rel).read_bytes() == (source / transformer_rel).read_bytes()
    assert _tensor_payload(output / "text_encoder" / "model.safetensors", "language_model.layers.0.weight") == b"LANG"
    assert "visual.patch_embed.weight" not in read_safetensors_header(
        output / "text_encoder" / "model.safetensors"
    ).tensors
    assert _tensor_payload(output / "vae" / "diffusion_pytorch_model.safetensors", "decoder.conv.weight") == b"DECO"
    assert "encoder.conv.weight" not in read_safetensors_header(
        output / "vae" / "diffusion_pytorch_model.safetensors"
    ).tensors
    assert summary["transformer_index"]["tensor_count"] == 1


def test_selective_rebuild_preserves_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write_source_metadata(source)
    write_safetensors_fixture(
        source / "vae" / "diffusion_pytorch_model.safetensors",
        {
            "decoder.conv.weight": ("F32", [1]),
            "encoder.conv.weight": ("F32", [1]),
        },
        payloads={
            "decoder.conv.weight": b"DECO",
            "encoder.conv.weight": b"ENCO",
        },
    )

    selection = select_manifest_tensors(generate_manifest(source))
    write_selected_safetensors(
        source_root=source,
        output_root=output,
        decisions=selection["decisions"],
    )

    source_header = read_safetensors_header(
        source / "vae" / "diffusion_pytorch_model.safetensors"
    )
    written_header = read_safetensors_header(
        output / "vae" / "diffusion_pytorch_model.safetensors"
    )
    assert source_header.metadata == {"format": "pt"}
    assert written_header.metadata == source_header.metadata


def test_copy_rejects_quantized_kept_tensors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write_source_metadata(source)
    write_safetensors_fixture(
        source / "transformer" / "diffusion_pytorch_model.safetensors",
        {"transformer_blocks.0.weight": ("I8", [4])},
        payloads={"transformer_blocks.0.weight": b"int8"},
    )

    selection = select_manifest_tensors(generate_manifest(source))

    with pytest.raises(Krea2TurboMlxError, match="Quantized"):
        write_selected_safetensors(
            source_root=source,
            output_root=output,
            decisions=selection["decisions"],
        )


def _write_source_metadata(root: Path) -> None:
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "config.json").write_text("{}", encoding="utf-8")


def _tensor_payload(path: Path, key: str) -> bytes:
    header = read_safetensors_header(path)
    tensor = header.tensors[key]
    raw = path.read_bytes()
    start = header.payload_start + tensor.data_offsets[0]
    end = header.payload_start + tensor.data_offsets[1]
    return raw[start:end]
