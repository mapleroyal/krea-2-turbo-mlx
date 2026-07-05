from __future__ import annotations

import pytest

from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.tensor_selection import select_manifest_tensors


def test_m1_selection_keeps_runtime_tensors_and_drops_verified_branches() -> None:
    selection = select_manifest_tensors(
        {
            "tensor_inventory": [
                _tensor("transformer", "transformer/model.safetensors", "blocks.0.weight", "BF16", 2),
                _tensor(
                    "text_encoder",
                    "text_encoder/model.safetensors",
                    "language_model.layers.0.weight",
                    "BF16",
                    4,
                ),
                _tensor(
                    "text_encoder",
                    "text_encoder/model.safetensors",
                    "visual.patch_embed.weight",
                    "BF16",
                    4,
                ),
                _tensor("vae", "vae/diffusion_pytorch_model.safetensors", "decoder.conv.weight", "F32", 8),
                _tensor(
                    "vae",
                    "vae/diffusion_pytorch_model.safetensors",
                    "post_quant_conv.weight",
                    "F32",
                    8,
                ),
                _tensor("vae", "vae/diffusion_pytorch_model.safetensors", "encoder.conv.weight", "F32", 8),
                _tensor("vae", "vae/diffusion_pytorch_model.safetensors", "quant_conv.weight", "F32", 8),
                _tensor("root", "turbo.safetensors", "duplicate.weight", "BF16", 2),
            ]
        }
    )

    kept = {decision["key"] for decision in selection["decisions"] if decision["keep"]}
    dropped = {decision["key"] for decision in selection["decisions"] if not decision["keep"]}

    assert kept == {
        "blocks.0.weight",
        "language_model.layers.0.weight",
        "decoder.conv.weight",
        "post_quant_conv.weight",
    }
    assert dropped == {
        "visual.patch_embed.weight",
        "encoder.conv.weight",
        "quant_conv.weight",
        "duplicate.weight",
    }
    assert selection["rule_matches"]["text_encoder.drop_lm_head"] == 0
    assert selection["summary"]["selected"]["dtypes"] == {"BF16": 2, "F32": 2}


def test_m1_selection_drops_lm_head_when_present() -> None:
    selection = select_manifest_tensors(
        {
            "tensor_inventory": [
                _tensor(
                    "text_encoder",
                    "text_encoder/model.safetensors",
                    "language_model.layers.0.weight",
                    "BF16",
                    2,
                ),
                _tensor(
                    "text_encoder",
                    "text_encoder/model.safetensors",
                    "lm_head.weight",
                    "BF16",
                    2,
                ),
            ]
        }
    )

    kept = {decision["key"] for decision in selection["decisions"] if decision["keep"]}
    dropped = {decision["key"] for decision in selection["decisions"] if not decision["keep"]}

    assert kept == {"language_model.layers.0.weight"}
    assert dropped == {"lm_head.weight"}
    assert selection["rule_matches"]["text_encoder.drop_lm_head"] == 1


def test_m1_selection_drops_language_model_lm_head_when_present() -> None:
    manifest = {
        "tensor_inventory": [
            _tensor(
                "text_encoder",
                "text_encoder/model.safetensors",
                "language_model.layers.0.weight",
                "BF16",
                2,
            ),
            _tensor(
                "text_encoder",
                "text_encoder/model.safetensors",
                "language_model.lm_head.weight",
                "BF16",
                2,
            ),
        ]
    }

    selection = select_manifest_tensors(manifest)
    dropped = {
        decision["key"]
        for decision in selection["decisions"]
        if decision["decision"] == "drop"
    }

    assert dropped == {"language_model.lm_head.weight"}


def test_m1_selection_fails_on_unknown_text_branch() -> None:
    with pytest.raises(Krea2TurboMlxError, match="text_encoder tensor selection rule"):
        select_manifest_tensors(
            {
                "tensor_inventory": [
                    _tensor(
                        "text_encoder",
                        "text_encoder/model.safetensors",
                        "mystery.weight",
                        "BF16",
                        2,
                    )
                ]
            }
        )


def test_m1_selection_fails_on_unknown_vae_branch() -> None:
    with pytest.raises(Krea2TurboMlxError, match="vae tensor selection rule"):
        select_manifest_tensors(
            {
                "tensor_inventory": [
                    _tensor(
                        "vae",
                        "vae/diffusion_pytorch_model.safetensors",
                        "mystery.weight",
                        "F32",
                        4,
                    )
                ]
            }
        )


def test_m1_selection_fails_on_unknown_component() -> None:
    with pytest.raises(Krea2TurboMlxError, match="No M1 tensor selection rule"):
        select_manifest_tensors(
            {
                "tensor_inventory": [
                    _tensor(
                        "controlnet",
                        "controlnet/model.safetensors",
                        "blocks.0.weight",
                        "BF16",
                        2,
                    )
                ]
            }
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
        "shape": [byte_count],
        "byte_count": byte_count,
    }
