from __future__ import annotations

import os
from pathlib import Path

import pytest

from krea_2_turbo_mlx.vae import load_vae, postprocess_decoded_image


@pytest.mark.skipif(
    not os.environ.get("KREA2_TURBO_MLX_VAE_ORACLE_SOURCE"),
    reason="set KREA2_TURBO_MLX_VAE_ORACLE_SOURCE to run the real-weight VAE oracle",
)
def test_vae_decode_matches_pinned_diffusers_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    image_processor = pytest.importorskip("diffusers.image_processor")

    source = Path(os.environ["KREA2_TURBO_MLX_VAE_ORACLE_SOURCE"]).expanduser()
    latents_np = np.random.default_rng(1234).standard_normal(
        (1, 16, 1, 16, 16),
    ).astype(np.float32)

    mlx_vae = load_vae(source, dtype=mx.float32)
    mlx_raw = mlx_vae.decode(mx.array(latents_np, dtype=mx.float32)).sample
    mlx_post = postprocess_decoded_image(mlx_raw, output_type="np")
    mx.eval(mlx_raw)

    torch_vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        source / "vae",
        torch_dtype=torch.float32,
    )
    torch_vae.eval()
    with torch.no_grad():
        torch_latents = torch.from_numpy(latents_np)
        mean = torch.tensor(torch_vae.config.latents_mean).view(1, 16, 1, 1, 1)
        std = torch.tensor(torch_vae.config.latents_std).view(1, 16, 1, 1, 1)
        torch_raw = torch_vae.decode(
            torch_latents * std + mean,
            return_dict=False,
        )[0]
        torch_post = image_processor.VaeImageProcessor().postprocess(
            torch_raw[:, :, 0],
            output_type="np",
        )

    raw_metrics = _metrics(np.array(mlx_raw), torch_raw.cpu().numpy())
    post_metrics = _metrics(mlx_post, torch_post)
    assert raw_metrics["mean_abs"] <= 0.01, raw_metrics
    assert raw_metrics["p99_abs"] <= 0.05, raw_metrics
    assert raw_metrics["cosine"] >= 0.9999, raw_metrics
    assert post_metrics["mean_abs"] <= 0.005, post_metrics
    assert post_metrics["p99_abs"] <= 0.03, post_metrics


@pytest.mark.skipif(
    not os.environ.get("KREA2_TURBO_MLX_VAE_ORACLE_SOURCE"),
    reason="set KREA2_TURBO_MLX_VAE_ORACLE_SOURCE to run the real-weight VAE oracle",
)
def test_vae_tiled_decode_matches_pinned_diffusers_postprocess() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    image_processor = pytest.importorskip("diffusers.image_processor")

    source = Path(os.environ["KREA2_TURBO_MLX_VAE_ORACLE_SOURCE"]).expanduser()
    latents_np = np.random.default_rng(20260628).standard_normal(
        (1, 16, 1, 64, 64),
    ).astype(np.float32)

    # Pin the stride explicitly on both sides: the MLX default stride (224, overlap
    # 0.125) intentionally differs from diffusers' default, so parity is only
    # meaningful when both decode with identical tiling geometry.
    mlx_vae = load_vae(source, dtype=mx.float32)
    mlx_vae.enable_tiling(tile_sample_stride_height=224, tile_sample_stride_width=224)
    mlx_raw = mlx_vae.decode(mx.array(latents_np, dtype=mx.float32)).sample
    mlx_post = postprocess_decoded_image(mlx_raw, output_type="np")
    mx.eval(mlx_raw)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        source / "vae",
        torch_dtype=torch.float32,
    )
    torch_vae.enable_tiling(tile_sample_stride_height=224, tile_sample_stride_width=224)
    torch_vae.to(device)
    torch_vae.eval()
    with torch.no_grad():
        torch_latents = torch.from_numpy(latents_np).to(device=device, dtype=torch.float32)
        mean = torch.tensor(torch_vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        std = torch.tensor(torch_vae.config.latents_std, device=device).view(1, 16, 1, 1, 1)
        torch_raw = torch_vae.decode(
            torch_latents * std + mean,
            return_dict=False,
        )[0]
        torch_post = image_processor.VaeImageProcessor().postprocess(
            torch_raw[:, :, 0],
            output_type="np",
        )

    post_metrics = _metrics(mlx_post, torch_post)
    assert post_metrics["mean_abs"] <= 0.001, post_metrics
    assert post_metrics["p99_abs"] <= 0.005, post_metrics
    assert post_metrics["cosine"] >= 0.99999, post_metrics


def _metrics(actual: object, expected: object) -> dict[str, float]:
    np = pytest.importorskip("numpy")
    actual_np = np.asarray(actual, dtype=np.float64)
    expected_np = np.asarray(expected, dtype=np.float64)
    diff = np.abs(actual_np - expected_np)
    actual_flat = actual_np.reshape(-1)
    expected_flat = expected_np.reshape(-1)
    denom = np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "cosine": float(np.dot(actual_flat, expected_flat) / denom),
    }
