from __future__ import annotations

import json
from pathlib import Path

import pytest

from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.text_encoder import (
    Qwen3VLTextConfig,
    Qwen3VLTextModel,
    build_text_rotary_embeddings,
    build_text_position_ids,
    causal_padding_attention_mask,
    flatten_parameters,
    flatten_parameter_shapes,
    load_text_encoder_weights,
    map_text_encoder_weight_key,
)


def test_config_accepts_krea_decoupled_head_dim() -> None:
    config = Qwen3VLTextConfig.from_model_config(_model_config())

    assert config.hidden_size == 2560
    assert config.attention_dim == 4096
    assert config.num_attention_heads * config.head_dim != config.hidden_size


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"hidden_act": "gelu"}, "hidden_act='silu'"),
        ({"attention_bias": True}, "attention_bias=false"),
        ({"num_attention_heads": 30}, "divisible"),
        ({"rope_parameters": {"rope_type": "linear"}}, "default RoPE"),
        ({"rope_parameters": {"mrope_interleaved": False}}, "interleaved"),
        ({"rope_parameters": {"mrope_section": [24, 20, 19]}}, "sum"),
    ],
)
def test_config_rejects_unsupported_text_variants(
    override: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Qwen3VLTextConfig.from_model_config(_model_config(**override))


def test_weight_key_mapping_strips_runtime_prefix_and_ignores_unreachable_keys() -> None:
    assert (
        map_text_encoder_weight_key("language_model.layers.0.self_attn.q_proj.weight")
        == "layers.0.self_attn.q_proj.weight"
    )
    assert map_text_encoder_weight_key("visual.patch_embed.proj.weight") is None
    assert map_text_encoder_weight_key("lm_head.weight") is None
    assert map_text_encoder_weight_key("language_model.lm_head.weight") is None

    with pytest.raises(Krea2TurboMlxError, match="Unexpected text encoder tensor"):
        map_text_encoder_weight_key("other.unexpected.weight")


def test_tiny_mlx_forward_returns_transformers_hidden_state_order() -> None:
    mx = pytest.importorskip("mlx.core")

    config = _tiny_config(num_hidden_layers=2)
    model = Qwen3VLTextModel(config)
    input_ids = mx.array([[1, 2, 0, 3]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 0, 1]], dtype=mx.bool_)
    position_ids = build_text_position_ids(attention_mask)

    output = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    mx.eval(output.last_hidden_state, *output.hidden_states)

    assert output.last_hidden_state.shape == (1, 4, config.hidden_size)
    assert len(output.hidden_states) == config.num_hidden_layers + 2
    assert output.hidden_states[0].shape == (1, 4, config.hidden_size)
    assert output.hidden_states[-1].shape == output.last_hidden_state.shape


def test_text_only_mrope_matches_1d_rope_when_axes_share_positions() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    config = _tiny_config(num_hidden_layers=1)
    positions = mx.array([[0, 1, 2, 5]], dtype=mx.int32)
    position_ids = mx.stack([positions, positions, positions], axis=0)

    cos, sin = build_text_rotary_embeddings(config, position_ids, dtype=mx.float32)
    inv_freq = 1.0 / (
        config.rope_theta
        ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
    )
    freqs = positions[:, :, None].astype(mx.float32) * inv_freq[None, None, :]
    expected = mx.concatenate([freqs, freqs], axis=-1)
    expected_cos = mx.cos(expected)
    expected_sin = mx.sin(expected)
    mx.eval(cos, sin, expected_cos, expected_sin)

    np.testing.assert_allclose(np.array(cos), np.array(expected_cos), atol=1e-6)
    np.testing.assert_allclose(np.array(sin), np.array(expected_sin), atol=1e-6)


def test_causal_padding_attention_mask_blocks_future_and_padded_keys() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    attention_mask = mx.array([[1, 1, 0, 1]], dtype=mx.bool_)

    mask = causal_padding_attention_mask(attention_mask)
    mx.eval(mask)

    expected = np.array(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, False, False],
                    [True, True, False, True],
                ]
            ]
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(np.array(mask), expected)


def test_load_text_encoder_weights_accepts_single_file_and_indexed_shards(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")

    model = Qwen3VLTextModel(_tiny_config(num_hidden_layers=1))
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = {
        f"language_model.{key}": value
        for key, value in flatten_parameters(model.parameters())
    }
    official_arrays["visual.patch_embed.weight"] = mx.zeros((1,), dtype=mx.float32)
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    mx.save_safetensors(str(single_dir / "model.safetensors"), official_arrays)

    single = load_text_encoder_weights(single_dir, expected_shapes=expected_shapes)
    assert set(single) == set(expected_shapes)

    indexed_dir = tmp_path / "indexed"
    indexed_dir.mkdir()
    first_items = dict(list(official_arrays.items())[: len(official_arrays) // 2])
    second_items = dict(list(official_arrays.items())[len(official_arrays) // 2 :])
    mx.save_safetensors(str(indexed_dir / "shard-a.safetensors"), first_items)
    mx.save_safetensors(str(indexed_dir / "shard-b.safetensors"), second_items)
    weight_map = {
        **{key: "shard-a.safetensors" for key in first_items},
        **{key: "shard-b.safetensors" for key in second_items},
    }
    (indexed_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    indexed = load_text_encoder_weights(indexed_dir, expected_shapes=expected_shapes)
    assert set(indexed) == set(expected_shapes)


def test_load_text_encoder_weights_fails_when_runtime_tensor_is_missing(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")

    model = Qwen3VLTextModel(_tiny_config(num_hidden_layers=1))
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = {
        f"language_model.{key}": value
        for key, value in flatten_parameters(model.parameters())
    }
    official_arrays.pop("language_model.embed_tokens.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), official_arrays)

    with pytest.raises(Krea2TurboMlxError, match="missing runtime tensors"):
        load_text_encoder_weights(tmp_path, expected_shapes=expected_shapes)


def _model_config(**text_overrides: object) -> dict[str, object]:
    text_config: dict[str, object] = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "pad_token_id": None,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 5_000_000,
            "rope_type": "default",
        },
        "vocab_size": 151936,
    }
    text_config.update(text_overrides)
    if "rope_parameters" in text_overrides:
        rope = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 5_000_000,
            "rope_type": "default",
        }
        rope.update(text_overrides["rope_parameters"])  # type: ignore[arg-type]
        text_config["rope_parameters"] = rope
    return {"model_type": "qwen3_vl", "text_config": text_config}


def _tiny_config(*, num_hidden_layers: int) -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        mrope_section=(1, 1, 1),
    )
