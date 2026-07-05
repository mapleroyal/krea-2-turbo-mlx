from __future__ import annotations

import json
from pathlib import Path

import pytest

from krea_2_turbo_mlx.png import (
    PNG_METADATA_KEY,
    PNG_PARAMETERS_KEY,
    generation_metadata_payload,
    save_generation_png,
)
from krea_2_turbo_mlx.errors import ValidationError


def test_save_generation_png_writes_reproducibility_itxt_metadata(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "artifact.json").write_text(
        json.dumps(
            {
                "source": {"revision": "abc123"},
                "provenance": {"source_revision": "abc123"},
                "precision": {
                    "selected_dtype_histogram": {
                        "BF16": 12,
                        "F32": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (model / "conversion_report.json").write_text("{}", encoding="utf-8")
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "image.png"
    metadata = generation_metadata_payload(
        prompt="a glass observatory",
        seed=42,
        width=32,
        height=16,
        steps=2,
        model_path=model,
        elapsed_seconds=1.25,
        truncation_warnings=({"prompt_index": 0, "truncated_tokens": 3},),
    )

    save_generation_png(
        Image.new("RGB", (1, 1)),
        output,
        metadata=metadata,
    )

    saved = Image.open(output)
    payload = json.loads(saved.info[PNG_METADATA_KEY])
    assert payload["prompt"] == "a glass observatory"
    assert payload["seed"] == 42
    assert payload["model_precision"] == "bf16"
    assert payload["source_revision"] == "abc123"
    assert payload["prompt_truncation"][0]["truncated_tokens"] == 3
    assert "Steps: 2" in saved.info[PNG_PARAMETERS_KEY]


def test_save_generation_png_refuses_existing_path_without_overwrite(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    output = tmp_path / "image.png"
    output.write_bytes(b"existing")
    metadata = {
        "prompt": "a glass observatory",
        "seed": 42,
        "width": 32,
        "height": 16,
        "steps": 2,
        "guidance_scale": 0.0,
        "shift": 1.15,
        "model_path": "artifact",
        "artifact_fingerprint": "sha256:test",
        "source_revision": "abc123",
        "generator": "krea-2-turbo-mlx",
        "generator_version": "0.1.0",
    }

    with pytest.raises(ValidationError, match="Pass --overwrite"):
        save_generation_png(
            Image.new("RGB", (1, 1)),
            output,
            metadata=metadata,
            overwrite=False,
        )

    assert output.read_bytes() == b"existing"
