from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest

from krea_2_turbo_mlx.pipeline import KreaTurboPipeline, pack_latents

ORACLE_SOURCE_ENV = "KREA2_TURBO_MLX_GENERATE_ORACLE_SOURCE"
ORACLE_ARTIFACT_ENV = "KREA2_TURBO_MLX_GENERATE_ORACLE_ARTIFACT"
ORACLE_PROMPT = "a glass observatory above a quiet fjord at sunrise"


pytestmark = pytest.mark.skipif(
    not os.environ.get(ORACLE_SOURCE_ENV) or not os.environ.get(ORACLE_ARTIFACT_ENV),
    reason=(
        f"set {ORACLE_SOURCE_ENV} and {ORACLE_ARTIFACT_ENV} "
        "to run the end-to-end generation oracle"
    ),
)


def test_generate_matches_pinned_diffusers_reference_with_fixed_latents() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    mx = pytest.importorskip("mlx.core")

    source = Path(os.environ[ORACLE_SOURCE_ENV]).expanduser()
    artifact = Path(os.environ[ORACLE_ARTIFACT_ENV]).expanduser()
    _require_oracle_source(source)
    _require_oracle_artifact(artifact)
    width = height = 256
    steps = 8
    packed_latents = _packed_latents(np, height=height, width=width)

    ref = _diffusers_output(
        torch,
        diffusers,
        source,
        packed_latents,
        width=width,
        height=height,
        steps=steps,
    )
    _collect(torch)

    actual = _mlx_output(
        np,
        mx,
        artifact,
        packed_latents,
        width=width,
        height=height,
        steps=steps,
    )
    metrics = _metrics(actual, ref)
    message = json.dumps(
        {
            "metrics": metrics,
            "schedule": _shifted_schedule(steps),
            "packed_latents": _summary(np, packed_latents),
            "actual": _summary(np, actual),
            "expected": _summary(np, ref),
        },
        indent=2,
        sort_keys=True,
    )
    print("GENERATE_ORACLE_METRICS " + json.dumps(metrics, sort_keys=True))
    assert metrics["max_abs"] <= 0.002, message
    assert metrics["mean_abs"] <= 0.000025, message
    assert metrics["p99_abs"] <= 0.00025, message
    assert metrics["cosine"] >= 0.99999999, message


def _diffusers_output(
    torch: object,
    diffusers: object,
    source: Path,
    packed_latents: object,
    *,
    width: int,
    height: int,
    steps: int,
) -> object:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = diffusers.Krea2Pipeline.from_pretrained(
        source,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipe.to(device)
    latents = torch.from_numpy(packed_latents).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = pipe(
            ORACLE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,
            latents=latents,
            output_type="np",
        ).images
    output_np = output.astype("float32")
    del pipe, latents, output
    return output_np


def _mlx_output(
    np: object,
    mx: object,
    source: Path,
    packed_latents: object,
    *,
    width: int,
    height: int,
    steps: int,
) -> object:
    pipe = KreaTurboPipeline.from_artifact(source, dtype=mx.float32)
    output = pipe(
        ORACLE_PROMPT,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=0.0,
        seed=0,
        output_type="np",
        _latents=mx.array(packed_latents, dtype=mx.float32),
    )
    images = np.asarray(output.images, dtype=np.float32)
    del pipe, output
    return images


def _packed_latents(np: object, *, height: int, width: int) -> object:
    raw = np.random.default_rng(20260625).standard_normal(
        (1, 16, height // 8, width // 8),
    ).astype(np.float32)
    return pack_latents(raw, patch_size=2)


def _shifted_schedule(steps: int) -> list[float]:
    import math

    base = np_linspace(1.0, 1.0 / steps, steps)
    return [
        math.exp(1.15) / (math.exp(1.15) + (1.0 / sigma - 1.0))
        for sigma in base
    ]


def np_linspace(start: float, stop: float, steps: int) -> list[float]:
    if steps == 1:
        return [float(start)]
    delta = (stop - start) / (steps - 1)
    return [start + index * delta for index in range(steps)]


def _require_oracle_source(source: Path) -> None:
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json",
    ]
    missing = [rel_path for rel_path in required if not (source / rel_path).is_file()]
    if missing:
        pytest.skip(f"oracle source is missing required files: {missing}")


def _require_oracle_artifact(artifact: Path) -> None:
    required = [
        "artifact.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json",
    ]
    missing = [rel_path for rel_path in required if not (artifact / rel_path).is_file()]
    if missing:
        pytest.skip(f"oracle artifact is missing required files: {missing}")


def _collect(torch: object) -> None:
    gc.collect()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _metrics(actual: object, expected: object) -> dict[str, float]:
    np = pytest.importorskip("numpy")
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    diff = np.abs(actual_array - expected_array)
    actual_flat = actual_array.reshape(-1)
    expected_flat = expected_array.reshape(-1)
    denom = np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "cosine": float(np.dot(actual_flat, expected_flat) / denom),
    }


def _summary(np: object, value: object) -> dict[str, object]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
