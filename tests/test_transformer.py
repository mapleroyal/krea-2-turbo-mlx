from __future__ import annotations

import json
from pathlib import Path

import pytest

from krea_2_turbo_mlx._weights import flatten_parameters, flatten_parameter_shapes
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.transformer import (
    Krea2Attention,
    Krea2RMSNorm,
    Krea2Transformer2DModel,
    Krea2TransformerConfig,
    apply_rotary_emb,
    build_attention_masks,
    build_krea2_rotary_embeddings,
    load_transformer,
    load_transformer_weights,
    prepare_position_ids,
)


def test_transformer_config_accepts_pinned_krea_values() -> None:
    config = Krea2TransformerConfig.from_model_config(_pinned_config())

    assert config.in_channels == 64
    assert config.hidden_size == 6144
    assert config.num_key_value_groups == 4
    assert config.axes_dims_rope == (32, 48, 48)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"_class_name": "Other"}, "_class_name='Krea2Transformer2DModel'"),
        ({"num_attention_heads": 47}, "divisible"),
        ({"hidden_size": 17}, "hidden_size"),
        ({"attention_head_dim": 127}, "attention_head_dim must be even"),
        ({"axes_dims_rope": [32, 48, 46]}, "sum"),
        ({"axes_dims_rope": [31, 49, 48]}, "positive even"),
        ({"norm_eps": 0}, "norm_eps"),
        ({"rope_theta": 0}, "rope_theta"),
    ],
)
def test_transformer_config_rejects_unsupported_variants(
    override: dict[str, object],
    match: str,
) -> None:
    payload = _pinned_config()
    payload.update(override)

    with pytest.raises(ValueError, match=match):
        Krea2TransformerConfig.from_model_config(payload)


def test_zero_centered_rms_norm_matches_fp32_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    norm = Krea2RMSNorm(4, eps=1e-5)
    norm.weight = mx.array([0.0, 0.5, -0.25, 1.0], dtype=mx.float32)
    hidden = mx.array(
        [[[1.0, -2.0, 3.0, -4.0], [2.0, 0.0, -1.0, 1.0]]],
        dtype=mx.float16,
    )

    actual = norm(hidden)
    mx.eval(actual)

    hidden_np = np.array(hidden, dtype=np.float32)
    scale = np.array([1.0, 1.5, 0.75, 2.0], dtype=np.float32)
    expected = hidden_np / np.sqrt(np.mean(hidden_np * hidden_np, axis=-1, keepdims=True) + 1e-5)
    expected = expected * scale
    np.testing.assert_allclose(np.array(actual, dtype=np.float32), expected, rtol=1e-3, atol=1e-3)


def test_prepare_position_ids_places_text_at_origin_and_images_on_grid() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    position_ids = prepare_position_ids(3, 2, 3)
    mx.eval(position_ids)

    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.array(position_ids), expected)


def test_rotary_embeddings_and_adjacent_pair_rotation_match_diffusers_layout() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    position_ids_np = np.array([[0, 0, 0], [0, 1, 2], [0, 3, 4]], dtype=np.int32)
    position_ids = mx.array(position_ids_np, dtype=mx.int32)
    axes_dims = (2, 4, 2)

    cos, sin = build_krea2_rotary_embeddings(
        position_ids,
        axes_dims_rope=axes_dims,
        theta=1000.0,
        dtype=mx.float32,
    )
    mx.eval(cos, sin)

    expected_cos, expected_sin = _numpy_rotary(position_ids_np, axes_dims, theta=1000.0)
    np.testing.assert_allclose(np.array(cos), expected_cos, atol=1e-6)
    np.testing.assert_allclose(np.array(sin), expected_sin, atol=1e-6)

    hidden_np = np.arange(24, dtype=np.float32).reshape(1, 3, 1, 8) / 10.0
    hidden = mx.array(hidden_np, dtype=mx.float32)
    actual = apply_rotary_emb(hidden, (cos, sin), sequence_dim=1)
    mx.eval(actual)

    paired = hidden_np.reshape(1, 3, 1, 4, 2)
    rotated = np.stack([-paired[..., 1], paired[..., 0]], axis=-1).reshape(hidden_np.shape)
    expected = hidden_np * expected_cos[None, :, None, :] + rotated * expected_sin[None, :, None, :]
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)


def test_attention_uses_sigmoid_gate_gqa_and_key_padding_mask() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    attn = Krea2Attention(4, 2, 1, eps=1e-6)
    attn.to_q.weight = mx.array(np.eye(4, dtype=np.float32))
    attn.to_k.weight = mx.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=mx.float32)
    attn.to_v.weight = mx.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=mx.float32)
    attn.to_gate.weight = mx.zeros((4, 4), dtype=mx.float32)
    attn.to_out[0].weight = mx.array(np.eye(4, dtype=np.float32))

    hidden_np = np.array([[[1, 2, 3, 4], [2, 0, 0, 1]]], dtype=np.float32)
    hidden = mx.array(hidden_np, dtype=mx.float32)
    mask = mx.array([[[[True, False]]]], dtype=mx.bool_)

    actual = attn(hidden, attention_mask=mask)
    mx.eval(actual)

    query = _rms_norm(hidden_np.reshape(1, 2, 2, 2), eps=1e-6)
    key = _rms_norm(hidden_np[:, :, :2].reshape(1, 2, 1, 2), eps=1e-6)
    value = hidden_np[:, :, 2:].reshape(1, 2, 1, 2)
    query = np.transpose(query, (0, 2, 1, 3))
    key = np.repeat(np.transpose(key, (0, 2, 1, 3)), 2, axis=1)
    value = np.repeat(np.transpose(value, (0, 2, 1, 3)), 2, axis=1)
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(2.0)
    scores[:, :, :, 1] = -1e9
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    expected = np.matmul(probs, value)
    expected = np.transpose(expected, (0, 2, 1, 3)).reshape(1, 2, 4) * 0.5
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)


def test_attention_mask_helper_returns_none_for_all_valid_and_masks_text_only() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    all_valid = mx.ones((2, 3), dtype=mx.bool_)
    assert build_attention_masks(all_valid, image_seq_len=2) == (None, None)

    text_mask, combined_mask = build_attention_masks(
        mx.array([[1, 0, 1]], dtype=mx.bool_),
        image_seq_len=2,
    )
    mx.eval(text_mask, combined_mask)

    assert text_mask.shape == (1, 1, 1, 3)
    assert combined_mask.shape == (1, 1, 1, 5)
    np.testing.assert_array_equal(
        np.array(text_mask),
        np.array([[[[True, False, True]]]]),
    )
    np.testing.assert_array_equal(
        np.array(combined_mask),
        np.array([[[[True, False, True, True, True]]]]),
    )


def test_tiny_transformer_forward_is_deterministic_and_returns_image_tokens() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    config = _tiny_config()
    model = Krea2Transformer2DModel(config)
    hidden = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4) / 10.0
    encoder = mx.arange(48, dtype=mx.float32).reshape(1, 4, 3, 4) / 100.0
    timestep = mx.array([0.5], dtype=mx.float32)
    position_ids = prepare_position_ids(4, 1, 2)
    mask = mx.array([[1, 1, 0, 1]], dtype=mx.bool_)

    first = model(hidden, encoder, timestep, position_ids, mask)
    second = model(hidden, encoder, timestep, position_ids, mask, return_dict=False)[0]
    mx.eval(first.sample, second)

    assert first.sample.shape == (1, 2, 4)
    np.testing.assert_allclose(np.array(first.sample), np.array(second), atol=0)


def test_load_transformer_weights_accepts_single_file_and_indexed_shards(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")

    model = Krea2Transformer2DModel(_tiny_config())
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = {
        key: value.astype(mx.bfloat16)
        for key, value in flatten_parameters(model.parameters())
    }

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    mx.save_safetensors(str(single_dir / "diffusion_pytorch_model.safetensors"), official_arrays)

    single = load_transformer_weights(single_dir, expected_shapes=expected_shapes)
    assert set(single) == set(expected_shapes)
    assert single["img_in.weight"].dtype == mx.bfloat16

    recast = load_transformer_weights(
        single_dir,
        expected_shapes=expected_shapes,
        dtype=mx.float32,
    )
    assert recast["img_in.weight"].dtype == mx.float32

    indexed_dir = tmp_path / "indexed"
    indexed_dir.mkdir()
    items = list(official_arrays.items())
    first_items = dict(items[: len(items) // 2])
    second_items = dict(items[len(items) // 2 :])
    mx.save_safetensors(str(indexed_dir / "shard-a.safetensors"), first_items)
    mx.save_safetensors(str(indexed_dir / "shard-b.safetensors"), second_items)
    weight_map = {
        **{key: "shard-a.safetensors" for key in first_items},
        **{key: "shard-b.safetensors" for key in second_items},
    }
    (indexed_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    indexed = load_transformer_weights(indexed_dir, expected_shapes=expected_shapes)
    assert set(indexed) == set(expected_shapes)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing runtime tensors"),
        ("extra", "unexpected runtime tensors"),
        ("shape", "shape-mismatched tensors"),
    ],
)
def test_load_transformer_weights_fails_strictly(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    mx = pytest.importorskip("mlx.core")

    model = Krea2Transformer2DModel(_tiny_config())
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = dict(flatten_parameters(model.parameters()))
    if mutation == "missing":
        official_arrays.pop("img_in.weight")
    elif mutation == "extra":
        official_arrays["unexpected.weight"] = mx.zeros((1,), dtype=mx.float32)
    elif mutation == "shape":
        official_arrays["img_in.weight"] = mx.zeros((1,), dtype=mx.float32)
    mx.save_safetensors(
        str(tmp_path / "diffusion_pytorch_model.safetensors"),
        official_arrays,
    )

    with pytest.raises(Krea2TurboMlxError, match=match):
        load_transformer_weights(tmp_path, expected_shapes=expected_shapes)


def test_load_transformer_reads_config_and_weights_from_artifact_root(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    config = _tiny_config()
    (transformer_dir / "config.json").write_text(
        json.dumps(_config_payload(config)),
        encoding="utf-8",
    )
    model = Krea2Transformer2DModel(config)
    mx.save_safetensors(
        str(transformer_dir / "diffusion_pytorch_model.safetensors"),
        dict(flatten_parameters(model.parameters())),
    )

    loaded = load_transformer(tmp_path, dtype=mx.float32)

    assert loaded.config == config
    assert loaded.img_in.weight.dtype == mx.float32


def _numpy_rotary(
    position_ids: object,
    axes_dims: tuple[int, int, int],
    *,
    theta: float,
) -> tuple[object, object]:
    np = pytest.importorskip("numpy")
    cos_parts = []
    sin_parts = []
    positions = np.asarray(position_ids, dtype=np.float32)
    for axis, axis_dim in enumerate(axes_dims):
        inv_freq = 1.0 / (theta ** (np.arange(0, axis_dim, 2, dtype=np.float32) / axis_dim))
        freqs = positions[:, axis : axis + 1] * inv_freq[None, :]
        cos_parts.append(np.repeat(np.cos(freqs), 2, axis=1))
        sin_parts.append(np.repeat(np.sin(freqs), 2, axis=1))
    return np.concatenate(cos_parts, axis=-1), np.concatenate(sin_parts, axis=-1)


def _rms_norm(value: object, *, eps: float) -> object:
    np = pytest.importorskip("numpy")
    value = np.asarray(value, dtype=np.float32)
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)


def _tiny_config() -> Krea2TransformerConfig:
    return Krea2TransformerConfig(
        in_channels=4,
        num_layers=1,
        attention_head_dim=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=4,
        text_hidden_dim=4,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        text_intermediate_size=8,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 2),
    )


def _config_payload(config: Krea2TransformerConfig) -> dict[str, object]:
    return {
        "_class_name": "Krea2Transformer2DModel",
        "in_channels": config.in_channels,
        "num_layers": config.num_layers,
        "attention_head_dim": config.attention_head_dim,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "intermediate_size": config.intermediate_size,
        "timestep_embed_dim": config.timestep_embed_dim,
        "text_hidden_dim": config.text_hidden_dim,
        "num_text_layers": config.num_text_layers,
        "text_num_attention_heads": config.text_num_attention_heads,
        "text_num_key_value_heads": config.text_num_key_value_heads,
        "text_intermediate_size": config.text_intermediate_size,
        "num_layerwise_text_blocks": config.num_layerwise_text_blocks,
        "num_refiner_text_blocks": config.num_refiner_text_blocks,
        "axes_dims_rope": list(config.axes_dims_rope),
        "rope_theta": config.rope_theta,
        "norm_eps": config.norm_eps,
    }


def _pinned_config() -> dict[str, object]:
    return {
        "_class_name": "Krea2Transformer2DModel",
        "_diffusers_version": "0.39.0.dev0",
        "attention_head_dim": 128,
        "axes_dims_rope": [32, 48, 48],
        "in_channels": 64,
        "intermediate_size": 16384,
        "norm_eps": 1e-05,
        "num_attention_heads": 48,
        "num_key_value_heads": 12,
        "num_layers": 28,
        "num_layerwise_text_blocks": 2,
        "num_refiner_text_blocks": 2,
        "num_text_layers": 12,
        "rope_theta": 1000.0,
        "text_hidden_dim": 2560,
        "text_intermediate_size": 6912,
        "text_num_attention_heads": 20,
        "text_num_key_value_heads": 20,
        "timestep_embed_dim": 256,
    }
