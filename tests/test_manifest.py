from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from krea_2_turbo_mlx import manifest as manifest_module
from krea_2_turbo_mlx.constants import OFFICIAL_HF_REPO_ID, OFFICIAL_HF_REVISION
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.manifest import generate_manifest
from krea_2_turbo_mlx.safetensors_header import SafetensorsHeader, TensorHeader
from safetensors_fixtures import write_safetensors_fixture


@dataclass
class FakeSibling:
    rfilename: str
    size: int
    lfs: dict[str, str] | None = None


@dataclass
class FakeCardData:
    license: str = "other"
    license_name: str = "krea-2-community-license"
    license_link: str = "https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf"


class FakeModelInfo:
    id = OFFICIAL_HF_REPO_ID
    sha = OFFICIAL_HF_REVISION
    last_modified = datetime(2026, 6, 23, 16, 23, tzinfo=timezone.utc)
    tags = ["license:other", "diffusers:Krea2Pipeline"]
    cardData = FakeCardData()
    library_name = "diffusers"
    pipeline_tag = "text-to-image"
    siblings = [
        FakeSibling("README.md", 4096),
        FakeSibling("images/00.jpg", 100_000),
        FakeSibling("model_index.json", 512),
        FakeSibling("scheduler/scheduler_config.json", 512),
        FakeSibling("text_encoder/config.json", 4096),
        FakeSibling("text_encoder/model.safetensors", 5_000_000_000, {"sha256": "text-sha"}),
        FakeSibling("tokenizer/chat_template.jinja", 2048),
        FakeSibling("tokenizer/tokenizer_config.json", 2048),
        FakeSibling("turbo.safetensors", 13_000_000_000),
        FakeSibling("transformer/config.json", 4096),
        FakeSibling("transformer/diffusion_pytorch_model.safetensors.index.json", 4096),
        FakeSibling("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 7_000_000_000),
        FakeSibling("vae/config.json", 2048),
        FakeSibling("vae/diffusion_pytorch_model.safetensors", 1_000_000_000),
    ]


class FakeHfApi:
    calls: list[dict[str, object]] = []

    def model_info(self, **kwargs: object) -> FakeModelInfo:
        self.calls.append(kwargs)
        return FakeModelInfo()


def test_official_remote_manifest_uses_pinned_revision_and_metadata_only(tmp_path: Path) -> None:
    FakeHfApi.calls = []
    downloaded: list[str] = []
    header_reads: list[str] = []
    payloads: dict[str, object] = {
        "model_index.json": {
            "_class_name": "Krea2Pipeline",
            "_diffusers_version": "0.39.0.dev0",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3VLModel"],
            "text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "Krea2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
            "is_distilled": True,
            "patch_size": 2,
        },
        "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
        "text_encoder/config.json": {
            "architectures": ["Qwen3VLModel"],
            "dtype": "bfloat16",
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
            },
        },
        "tokenizer/chat_template.jinja": "{% for message in messages %}{{ message.role }}{% endfor %}",
        "tokenizer/tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "transformer/config.json": {
            "_class_name": "Krea2Transformer2DModel",
            "num_layers": 28,
            "text_hidden_dim": 2560,
        },
        "transformer/diffusion_pytorch_model.safetensors.index.json": {
            "metadata": {"total_size": 42},
            "weight_map": {
                "transformer_blocks.0.attn.to_q.weight": "diffusion_pytorch_model-00001-of-00003.safetensors"
            },
        },
        "vae/config.json": {
            "_class_name": "AutoencoderKLQwenImage",
            "z_dim": 16,
        },
    }

    def fake_download(*, repo_id: str, revision: str, filename: str) -> str:
        assert repo_id == OFFICIAL_HF_REPO_ID
        assert revision == OFFICIAL_HF_REVISION
        downloaded.append(filename)
        path = tmp_path / filename.replace("/", "__")
        payload = payloads[filename]
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(str(payload), encoding="utf-8")
        return str(path)

    def fake_remote_header(
        repo_id: str,
        revision: str,
        filename: str,
        *,
        file_size: int | None,
    ) -> SafetensorsHeader:
        assert repo_id == OFFICIAL_HF_REPO_ID
        assert revision == OFFICIAL_HF_REVISION
        assert file_size is not None
        header_reads.append(filename)
        tensors = {
            "text_encoder/model.safetensors": {
                "language_model.embed_tokens.weight": TensorHeader(
                    "language_model.embed_tokens.weight", "BF16", (2, 2), (0, 8)
                ),
                "visual.patch_embed.weight": TensorHeader(
                    "visual.patch_embed.weight", "BF16", (2, 2), (8, 16)
                ),
            },
            "turbo.safetensors": {
                "duplicate.weight": TensorHeader("duplicate.weight", "BF16", (1,), (0, 2)),
            },
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": {
                "transformer_blocks.0.attn.to_q.weight": TensorHeader(
                    "transformer_blocks.0.attn.to_q.weight", "F32", (1,), (0, 4)
                ),
            },
            "vae/diffusion_pytorch_model.safetensors": {
                "decoder.conv.weight": TensorHeader("decoder.conv.weight", "F32", (1,), (0, 4)),
                "encoder.conv.weight": TensorHeader("encoder.conv.weight", "F32", (1,), (4, 8)),
            },
        }[filename]
        return SafetensorsHeader(metadata={"format": "pt"}, tensors=tensors, payload_start=128)

    with mock.patch.object(
        manifest_module,
        "_load_hf_tools",
        return_value=(FakeHfApi, fake_download),
    ), mock.patch.object(
        manifest_module,
        "_read_remote_safetensors_header",
        side_effect=fake_remote_header,
    ):
        manifest = generate_manifest(OFFICIAL_HF_REPO_ID)

    assert manifest["schema_version"] == 2
    assert FakeHfApi.calls[0]["revision"] == OFFICIAL_HF_REVISION
    assert manifest["status"] == "remote_manifest_generated"
    assert manifest["resolved_revision"] == OFFICIAL_HF_REVISION
    assert manifest["license"] == "krea-2-community-license"
    assert manifest["library_name"] == "diffusers"
    assert "text_encoder/model.safetensors" not in downloaded
    assert "vae/diffusion_pytorch_model.safetensors" not in downloaded
    assert "transformer/diffusion_pytorch_model.safetensors.index.json" in downloaded
    assert "text_encoder/model.safetensors" in manifest["safetensors_files"]
    assert sorted(header_reads) == [
        "text_encoder/model.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "turbo.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    assert manifest["tensor_summary"]["tensor_count"] == 6

    model_index = manifest["component_configs"]["model_index"][0]
    assert model_index["component_classes"]["text_encoder"] == ["transformers", "Qwen3VLModel"]
    assert model_index["text_encoder_select_layers"] == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

    text_config = manifest["component_configs"]["text_encoder"][0]
    assert text_config["text_config"]["hidden_size"] == 2560
    assert text_config["text_config"]["num_hidden_layers"] == 36

    index = manifest["safetensors_indexes"][0]
    assert index["component"] == "transformer"
    assert index["tensor_count"] == 1
    assert index["total_size"] == 42


def test_remote_safetensors_header_uses_exact_byte_ranges(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors_fixture(path, {"decoder.weight": ("F32", [2])})
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    ranges: list[str] = []

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self.payload if size < 0 else self.payload[:size]

    def fake_urlopen(request: object, *, timeout: int) -> FakeResponse:
        range_header = request.get_header("Range")
        ranges.append(range_header)
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = int(end_text)
        return FakeResponse(raw[start : end + 1])

    with mock.patch.object(manifest_module, "urlopen", side_effect=fake_urlopen):
        header = manifest_module._read_remote_safetensors_header(
            OFFICIAL_HF_REPO_ID,
            OFFICIAL_HF_REVISION,
            "vae/diffusion_pytorch_model.safetensors",
            file_size=len(raw),
        )

    assert ranges == ["bytes=0-7", f"bytes=8-{8 + header_length - 1}"]
    assert header.tensors["decoder.weight"].dtype == "F32"


def test_non_official_remote_manifest_requires_revision() -> None:
    with pytest.raises(Krea2TurboMlxError, match="requires --revision"):
        generate_manifest("someone/model")


def test_local_manifest_reads_headers_and_validates_indexes(tmp_path: Path) -> None:
    _write_minimal_source(tmp_path)
    transformer = tmp_path / "transformer"
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "transformer_blocks.0.weight": "diffusion_pytorch_model.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    write_safetensors_fixture(
        transformer / "diffusion_pytorch_model.safetensors",
        {"transformer_blocks.0.weight": ("F32", [2, 2])},
    )
    write_safetensors_fixture(
        tmp_path / "text_encoder" / "model.safetensors",
        {"model.embed_tokens.weight": ("BF16", [2, 3])},
    )

    manifest = generate_manifest(tmp_path)

    assert manifest["schema_version"] == 2
    assert manifest["status"] == "local_manifest_generated"
    assert manifest["tensor_summary"]["tensor_count"] == 2
    assert manifest["tensor_summary"]["components"]["transformer"]["dtypes"] == {"F32": 1}
    assert manifest["tensor_summary"]["components"]["text_encoder"]["dtypes"] == {"BF16": 1}
    assert manifest["safetensors_indexes"][0]["validation"]["status"] == "ok"


def test_local_manifest_reports_index_header_mismatch(tmp_path: Path) -> None:
    _write_minimal_source(tmp_path)
    transformer = tmp_path / "transformer"
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "missing.weight": "diffusion_pytorch_model.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    write_safetensors_fixture(
        transformer / "diffusion_pytorch_model.safetensors",
        {"actual.weight": ("F32", [2, 2])},
    )

    manifest = generate_manifest(tmp_path)

    validation = manifest["safetensors_indexes"][0]["validation"]
    assert validation["status"] == "error"
    assert validation["missing_tensors"] == [
        {"key": "missing.weight", "shard": "diffusion_pytorch_model.safetensors"}
    ]
    assert validation["extra_tensors"] == [
        {"key": "actual.weight", "shard": "diffusion_pytorch_model.safetensors"}
    ]


def _write_minimal_source(root: Path) -> None:
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline"}),
        encoding="utf-8",
    )
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "config.json").write_text("{}", encoding="utf-8")
