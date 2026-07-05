from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from krea_2_turbo_mlx import convert as convert_module
from krea_2_turbo_mlx.constants import OFFICIAL_HF_REPO_ID, OFFICIAL_HF_REVISION
from krea_2_turbo_mlx.convert import build_conversion_plan, run_conversion
from krea_2_turbo_mlx.doctor import run_doctor
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.json_io import read_json_object
from krea_2_turbo_mlx.safetensors_header import read_safetensors_header
from safetensors_fixtures import write_safetensors_fixture


def test_remote_conversion_builds_dry_plan_without_writes(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 2,
        "status": "remote_manifest_generated",
        "source_kind": "huggingface_model",
        "repo_id": OFFICIAL_HF_REPO_ID,
        "resolved_revision": OFFICIAL_HF_REVISION,
        "safetensors_headers": [],
        "tensor_inventory": [
            _tensor("transformer", "transformer/model.safetensors", "blocks.0.weight", "BF16", 2),
            _tensor(
                "text_encoder",
                "text_encoder/model.safetensors",
                "language_model.layers.0.weight",
                "BF16",
                2,
            ),
            _tensor("vae", "vae/diffusion_pytorch_model.safetensors", "decoder.conv.weight", "F32", 4),
        ],
        "tensor_summary": {"tensor_count": 3},
    }

    with mock.patch.object(convert_module, "generate_manifest", return_value=manifest):
        plan = build_conversion_plan(
            OFFICIAL_HF_REPO_ID,
            revision=OFFICIAL_HF_REVISION,
            output=tmp_path / "artifact",
        )

    assert plan["status"] == "dry_conversion_plan_generated"
    assert plan["writes_artifact"] is False
    assert plan["source"]["revision"] == OFFICIAL_HF_REVISION
    assert not (tmp_path / "artifact").exists()


def test_local_dry_plan_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write_diffusers_source(source)

    plan = build_conversion_plan(source, output=output)

    assert plan["status"] == "dry_conversion_plan_generated"
    assert plan["selection_summary"]["selected"]["tensor_count"] == 4
    assert not output.exists()


def test_local_conversion_writes_full_precision_artifact_and_doctor_accepts_it(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifact = tmp_path / "artifact"
    _write_diffusers_source(source)

    result = run_conversion(source, output=artifact)

    assert result["status"] == "artifact_written"
    metadata = read_json_object(artifact / "artifact.json")
    assert metadata["format"] == "krea-2-turbo-mlx-artifact"
    assert metadata["full_precision_only"] is True
    assert "quantization" not in metadata
    assert metadata["precision"]["dtype_equivalence_verified"] is True
    assert metadata["precision"]["quantized_dtypes_present"] is False
    assert metadata["provenance"]["selected_tensor_fingerprint"].startswith("sha256:")
    assert metadata["provenance"]["copied_metadata_fingerprint"].startswith("sha256:")

    text_header = read_safetensors_header(artifact / "text_encoder" / "model.safetensors")
    assert set(text_header.tensors) == {"language_model.layers.0.weight"}
    vae_header = read_safetensors_header(artifact / "vae" / "diffusion_pytorch_model.safetensors")
    assert set(vae_header.tensors) == {"decoder.conv.weight", "post_quant_conv.weight"}
    assert not (artifact / "turbo.safetensors").exists()

    report = read_json_object(artifact / "conversion_report.json")
    dropped = {
        decision["key"]
        for decision in report["tensor_decisions"]
        if decision["decision"] == "drop"
    }
    assert {"visual.patch_embed.weight", "encoder.conv.weight", "duplicate.weight"} <= dropped
    assert run_doctor(model=artifact)["status"] == "ok"


def test_remote_conversion_downloads_source_then_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "downloaded"
    artifact = tmp_path / "artifact"
    progress: list[str] = []

    def fake_download(*args: object, **kwargs: object) -> Path:
        assert args == (OFFICIAL_HF_REPO_ID,)
        assert kwargs["revision"] == OFFICIAL_HF_REVISION
        assert kwargs["local_dir"] == tmp_path / "source-dir"
        kwargs["progress_callback"](
            f"downloading {OFFICIAL_HF_REPO_ID}@{OFFICIAL_HF_REVISION} into "
            f"{tmp_path / 'source-dir'}"
        )
        _write_diffusers_source(source)
        return source

    monkeypatch.setattr(convert_module, "download_source", fake_download)

    result = run_conversion(
        OFFICIAL_HF_REPO_ID,
        revision=OFFICIAL_HF_REVISION,
        source_dir=tmp_path / "source-dir",
        output=artifact,
        progress_callback=progress.append,
    )

    assert result["status"] == "artifact_written"
    assert artifact.joinpath("artifact.json").is_file()
    # Progress is surfaced to the caller (the download step is prefixed and
    # forwarded), but the exact wording is not a contract worth freezing.
    assert progress, "conversion should report progress to the callback"
    assert any("download" in message for message in progress)


def test_conversion_preflight_failure_leaves_no_artifact_or_temp_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifact = tmp_path / "artifact"
    _write_diffusers_source(source)

    with mock.patch.object(
        convert_module,
        "preflight_free_space",
        side_effect=Krea2TurboMlxError("not enough test space"),
    ):
        with pytest.raises(Krea2TurboMlxError, match="not enough test space"):
            run_conversion(source, output=artifact)

    assert not artifact.exists()
    assert not any(path.name.startswith(".artifact.tmp-") for path in tmp_path.iterdir())


def _write_diffusers_source(root: Path) -> None:
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline"}),
        encoding="utf-8",
    )
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "config.json").write_text("{}", encoding="utf-8")
    write_safetensors_fixture(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        {"transformer_blocks.0.weight": ("BF16", [1])},
        payloads={"transformer_blocks.0.weight": b"tr"},
    )
    write_safetensors_fixture(
        root / "text_encoder" / "model.safetensors",
        {
            "language_model.layers.0.weight": ("BF16", [1]),
            "visual.patch_embed.weight": ("BF16", [1]),
        },
        payloads={
            "language_model.layers.0.weight": b"lm",
            "visual.patch_embed.weight": b"vi",
        },
    )
    write_safetensors_fixture(
        root / "vae" / "diffusion_pytorch_model.safetensors",
        {
            "decoder.conv.weight": ("F32", [1]),
            "post_quant_conv.weight": ("F32", [1]),
            "encoder.conv.weight": ("F32", [1]),
        },
        payloads={
            "decoder.conv.weight": b"deco",
            "post_quant_conv.weight": b"post",
            "encoder.conv.weight": b"enco",
        },
    )
    write_safetensors_fixture(
        root / "turbo.safetensors",
        {"duplicate.weight": ("BF16", [1])},
        payloads={"duplicate.weight": b"du"},
    )


def _tensor(
    component: str,
    path: str,
    key: str,
    dtype: str,
    byte_count: int,
) -> dict[str, object]:
    return {
        "component": component,
        "path": path,
        "key": key,
        "dtype": dtype,
        "shape": [1],
        "byte_count": byte_count,
    }
