from __future__ import annotations

import json
from pathlib import Path

import pytest

from krea_2_turbo_mlx._weights import flatten_parameter_shapes, flatten_parameters
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.vae import (
    AutoencoderKLQwenImage,
    QwenImageCausalConv3d,
    QwenImageRMSNorm,
    QwenImageResample,
    QwenImageVAEConfig,
    decode_latents,
    denormalize_latents,
    load_vae,
    load_vae_weights,
    postprocess_decoded_image,
)


def test_vae_config_accepts_pinned_qwen_image_values() -> None:
    config = QwenImageVAEConfig.from_model_config(_pinned_config())

    assert config.base_dim == 96
    assert config.z_dim == 16
    assert config.dim_mult == (1, 2, 4, 4)
    assert config.temperal_upsample == (True, True, False)
    assert config.spatial_compression_ratio == 8


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"_class_name": "Other"}, "_class_name='AutoencoderKLQwenImage'"),
        ({"z_dim": 0}, "z_dim"),
        ({"dim_mult": []}, "dim_mult"),
        ({"attn_scales": [1.0]}, "attn_scales"),
        ({"dropout": 0.1}, "dropout"),
        ({"temperal_downsample": [True]}, "temperal_downsample"),
        ({"latents_mean": [0.0]}, "latents_mean"),
        ({"latents_std": [0.0] * 16}, "latents_std entries"),
    ],
)
def test_vae_config_rejects_unsupported_variants(
    override: dict[str, object],
    match: str,
) -> None:
    payload = _pinned_config()
    payload.update(override)

    with pytest.raises(ValueError, match=match):
        QwenImageVAEConfig.from_model_config(payload)


def test_denormalize_latents_broadcasts_mean_and_std() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    config = _tiny_config(z_dim=2, latents_mean=(10.0, -10.0), latents_std=(2.0, 4.0))
    latents = mx.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=mx.float32)

    actual = denormalize_latents(latents, config)
    mx.eval(actual)

    assert actual.shape == (1, 2, 1, 1, 2)
    expected = np.array([[[[[12.0, 14.0]]], [[[2.0, 6.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(np.array(actual), expected, atol=0)


def test_rms_norm_matches_qwen_l2_normalize_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    norm = QwenImageRMSNorm(3)
    norm.gamma = mx.array([1.0, 0.5, 2.0], dtype=mx.float32)
    hidden = mx.array([[[[[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]]]]], dtype=mx.float32)

    actual = norm(hidden)
    mx.eval(actual)

    hidden_np = np.array(hidden, dtype=np.float32)
    expected = hidden_np / np.sqrt(np.sum(hidden_np * hidden_np, axis=-1, keepdims=True))
    expected = expected * np.sqrt(3.0) * np.array([1.0, 0.5, 2.0], dtype=np.float32)
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)


def test_causal_conv3d_pads_only_past_temporal_context() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    conv = QwenImageCausalConv3d(1, 1, (3, 1, 1), padding=(1, 0, 0))
    conv.weight = mx.ones((1, 3, 1, 1, 1), dtype=mx.float32)
    conv.bias = mx.zeros((1,), dtype=mx.float32)
    hidden = mx.array([1.0, 2.0], dtype=mx.float32).reshape(1, 2, 1, 1, 1)

    actual = conv(hidden)
    mx.eval(actual)

    expected = np.array([1.0, 3.0], dtype=np.float32).reshape(1, 2, 1, 1, 1)
    np.testing.assert_allclose(np.array(actual), expected, atol=0)


def test_resample_upsamples_spatially_and_keeps_single_frame_time() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    resample = QwenImageResample(2, "upsample3d")
    weight = np.zeros((1, 3, 3, 2), dtype=np.float32)
    weight[0, 1, 1, 0] = 1.0
    resample.resample[1].weight = mx.array(weight, dtype=mx.float32)
    resample.resample[1].bias = mx.zeros((1,), dtype=mx.float32)
    hidden = mx.array([[[[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]]])

    actual = resample(hidden)
    mx.eval(actual)

    assert actual.shape == (1, 1, 4, 4, 1)
    expected = np.repeat(np.repeat(np.array(hidden)[..., :1], 2, axis=2), 2, axis=3)
    np.testing.assert_allclose(np.array(actual), expected, atol=0)


def test_tiny_decode_returns_single_frame_clamped_image_tensor() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    config = _tiny_config()
    model = AutoencoderKLQwenImage(config)
    latents = mx.ones((1, config.z_dim, 1, 1), dtype=mx.float32)

    output = model.decode(latents)
    raw = decode_latents(model, latents, output_type="raw")
    mx.eval(output.sample, raw)

    assert output.sample.shape == (1, 3, 1, 8, 8)
    np.testing.assert_allclose(np.array(output.sample), np.array(raw), atol=0)
    assert float(mx.min(output.sample)) >= -1.0
    assert float(mx.max(output.sample)) <= 1.0


def test_tiled_decode_covers_partial_edge_tiles() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    config = _tiny_config(z_dim=1, dim_mult=(1,), temperal_downsample=())
    model = AutoencoderKLQwenImage(config)

    def fake_decode_denormalized(tile: object) -> object:
        return mx.concatenate([tile, tile, tile], axis=1)

    model._decode_denormalized = fake_decode_denormalized
    model.enable_tiling(
        tile_sample_min_height=2,
        tile_sample_min_width=2,
        tile_sample_stride_height=2,
        tile_sample_stride_width=2,
    )
    latents = (mx.arange(25, dtype=mx.float32).reshape(1, 1, 1, 5, 5) / 24.0) - 0.5

    actual = model.decode(latents, return_dict=False)[0]
    mx.eval(actual)

    expected = np.repeat(np.array(latents), 3, axis=1)
    np.testing.assert_allclose(np.array(actual), expected, atol=0)


def test_enable_tiling_rejects_stride_larger_than_tile_without_mutating() -> None:
    pytest.importorskip("mlx.core")

    model = AutoencoderKLQwenImage(_tiny_config(z_dim=1, dim_mult=(1,), temperal_downsample=()))

    with pytest.raises(ValueError, match="stride_height"):
        model.enable_tiling(
            tile_sample_min_height=2,
            tile_sample_min_width=2,
        )

    assert model.use_tiling is False
    assert model.tile_sample_min_height == 256
    assert model.tile_sample_stride_height == 224


def test_postprocess_decoded_image_matches_diffusers_denormalization_semantics() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    decoded = mx.array(
        [
            [
                [[[-1.0, 0.0], [1.0, 2.0]]],
                [[[0.0, 1.0], [-1.0, -2.0]]],
                [[[1.0, -1.0], [0.0, 0.5]]],
            ]
        ],
        dtype=mx.float32,
    )

    mlx_image = postprocess_decoded_image(decoded, output_type="mlx")
    np_image = postprocess_decoded_image(decoded, output_type="np")
    pil_image = postprocess_decoded_image(decoded, output_type="pil")
    mx.eval(mlx_image)

    assert mlx_image.shape == (1, 3, 2, 2)
    expected = np.clip(np.array(decoded[:, :, 0]) / 2.0 + 0.5, 0.0, 1.0)
    np.testing.assert_allclose(np.array(mlx_image), expected, atol=0)
    np.testing.assert_allclose(np_image, np.transpose(expected, (0, 2, 3, 1)), atol=0)
    assert len(pil_image) == 1
    assert pil_image[0].size == (2, 2)


def test_load_vae_weights_transforms_official_layouts_and_ignores_encode_tensors(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")

    model = AutoencoderKLQwenImage(_tiny_config(z_dim=2, dim_mult=(1,), temperal_downsample=()))
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = _official_arrays_from_model(model)
    post_quant = mx.arange(4, dtype=mx.float32).reshape(2, 2, 1, 1, 1)
    official_arrays["post_quant_conv.weight"] = post_quant
    official_arrays["encoder.conv.weight"] = mx.zeros((1,), dtype=mx.float32)
    official_arrays["quant_conv.weight"] = mx.zeros((1,), dtype=mx.float32)
    mx.save_safetensors(str(tmp_path / "diffusion_pytorch_model.safetensors"), official_arrays)

    weights = load_vae_weights(tmp_path, expected_shapes=expected_shapes)
    mx.eval(weights["post_quant_conv.weight"])

    assert set(weights) == set(expected_shapes)
    expected = np.array(post_quant).transpose(0, 2, 3, 4, 1)
    np.testing.assert_allclose(np.array(weights["post_quant_conv.weight"]), expected, atol=0)


def test_load_vae_weights_preserves_dtype_and_supports_override(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")

    model = AutoencoderKLQwenImage(_tiny_config(z_dim=2, dim_mult=(1,), temperal_downsample=()))
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = {
        key: value.astype(mx.bfloat16)
        for key, value in _official_arrays_from_model(model).items()
    }
    mx.save_safetensors(str(tmp_path / "diffusion_pytorch_model.safetensors"), official_arrays)

    weights = load_vae_weights(tmp_path, expected_shapes=expected_shapes)
    assert weights["post_quant_conv.weight"].dtype == mx.bfloat16

    recast = load_vae_weights(tmp_path, expected_shapes=expected_shapes, dtype=mx.float32)
    assert recast["post_quant_conv.weight"].dtype == mx.float32


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing runtime tensors"),
        ("extra", "unexpected runtime tensors"),
        ("unknown", "unexpected runtime tensors"),
        ("shape", "shape-mismatched tensors"),
    ],
)
def test_load_vae_weights_fails_strictly(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    mx = pytest.importorskip("mlx.core")

    model = AutoencoderKLQwenImage(_tiny_config(z_dim=2, dim_mult=(1,), temperal_downsample=()))
    expected_shapes = flatten_parameter_shapes(model.parameters())
    official_arrays = _official_arrays_from_model(model)
    if mutation == "missing":
        official_arrays.pop("post_quant_conv.weight")
    elif mutation == "extra":
        official_arrays["decoder.extra.weight"] = mx.zeros((1,), dtype=mx.float32)
    elif mutation == "unknown":
        official_arrays["other.weight"] = mx.zeros((1,), dtype=mx.float32)
    elif mutation == "shape":
        official_arrays["post_quant_conv.weight"] = mx.zeros((1,), dtype=mx.float32)
    mx.save_safetensors(str(tmp_path / "diffusion_pytorch_model.safetensors"), official_arrays)

    with pytest.raises(Krea2TurboMlxError, match=match):
        load_vae_weights(tmp_path, expected_shapes=expected_shapes)


def test_load_vae_reads_config_and_weights_from_artifact_root(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")

    config = _tiny_config(z_dim=2, dim_mult=(1,), temperal_downsample=())
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps(_config_payload(config)), encoding="utf-8")
    model = AutoencoderKLQwenImage(config)
    mx.save_safetensors(
        str(vae_dir / "diffusion_pytorch_model.safetensors"),
        _official_arrays_from_model(model),
    )

    loaded = load_vae(tmp_path, dtype=mx.float32)

    assert loaded.config == config
    assert loaded.post_quant_conv.weight.dtype == mx.float32


def _official_arrays_from_model(model: AutoencoderKLQwenImage) -> dict[str, object]:
    return {
        key: _to_official_array(key, value)
        for key, value in flatten_parameters(model.parameters())
    }


def _to_official_array(key: str, value: object) -> object:
    if key.endswith(".weight") and len(value.shape) == 5:
        return value.transpose(0, 4, 1, 2, 3)
    if key.endswith(".weight") and len(value.shape) == 4:
        return value.transpose(0, 3, 1, 2)
    if key.endswith(".gamma"):
        return value.reshape((value.shape[0], 1, 1, 1))
    return value


def _tiny_config(
    *,
    z_dim: int = 2,
    dim_mult: tuple[int, ...] = (1, 2, 2, 2),
    temperal_downsample: tuple[bool, ...] = (False, True, True),
    latents_mean: tuple[float, ...] | None = None,
    latents_std: tuple[float, ...] | None = None,
) -> QwenImageVAEConfig:
    return QwenImageVAEConfig(
        base_dim=2,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=0,
        temperal_downsample=temperal_downsample,
        input_channels=3,
        latents_mean=(0.0,) * z_dim if latents_mean is None else latents_mean,
        latents_std=(1.0,) * z_dim if latents_std is None else latents_std,
    )


def _config_payload(config: QwenImageVAEConfig) -> dict[str, object]:
    return {
        "_class_name": "AutoencoderKLQwenImage",
        "base_dim": config.base_dim,
        "z_dim": config.z_dim,
        "dim_mult": list(config.dim_mult),
        "num_res_blocks": config.num_res_blocks,
        "attn_scales": list(config.attn_scales),
        "temperal_downsample": list(config.temperal_downsample),
        "dropout": config.dropout,
        "input_channels": config.input_channels,
        "latents_mean": list(config.latents_mean),
        "latents_std": list(config.latents_std),
    }


def _pinned_config() -> dict[str, object]:
    return {
        "_class_name": "AutoencoderKLQwenImage",
        "_diffusers_version": "0.39.0.dev0",
        "_name_or_path": "Qwen/Qwen-Image",
        "attn_scales": [],
        "base_dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "input_channels": 3,
        "latents_mean": [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        "latents_std": [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.916,
        ],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "z_dim": 16,
    }
