from __future__ import annotations

import gc
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

from .constants import (
    ARTIFACT_FORMAT,
    DEFAULT_DISTILLED_SHIFT,
    DEFAULT_GENERATION_STEPS,
    DEFAULT_GENERATION_HEIGHT,
    DEFAULT_GENERATION_WIDTH,
    DEFAULT_GUIDANCE_SCALE,
    EXPECTED_COMPONENT_CLASSES,
    FULL_PRECISION_ONLY,
    MAX_GENERATION_SEED,
)
from .errors import Krea2TurboMlxError, ValidationError
from .generation_validation import validate_generation_dimensions
from .json_io import read_json_object
from .lora import ResolvedLoraPatch, applied_lora_patches
from .scheduler import FlowMatchEulerDiscreteScheduler, load_scheduler
from .tensor_selection import SELECTION_POLICY_VERSION
from .text_conditioning import load_text_conditioner
from .transformer import load_transformer, prepare_position_ids
from .vae import load_vae, postprocess_decoded_image

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None

PipelineProgressKind = Literal[
    "load_start",
    "load_component_start",
    "load_component_end",
    "load_complete",
    "encode_start",
    "encode_end",
    "prompt_truncated",
    "denoise_step_start",
    "denoise_step_end",
    "decode_start",
    "decode_end",
    "save_start",
    "save_end",
    "complete",
]
LivePreviewMode: TypeAlias = Literal["off", "latent", "vae"]
LIVE_PREVIEW_MODES: tuple[LivePreviewMode, ...] = ("off", "latent", "vae")
DEFAULT_PREVIEW_INTERVAL_STEPS = 2
MAX_PREVIEW_INTERVAL_STEPS = 100

PipelineProgressCallback = Callable[["PipelineProgressEvent"], None]
PipelinePreviewCallback = Callable[["PipelinePreviewFrame"], None]


@dataclass(frozen=True)
class PipelineProgressEvent:
    kind: PipelineProgressKind
    stage: str
    message: str
    progress: float | None = None
    step_index: int | None = None
    step_count: int | None = None
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PipelinePreviewFrame:
    mode: LivePreviewMode
    step_index: int
    step_count: int
    width: int
    height: int
    image: Any


@dataclass(frozen=True)
class KreaTurboPipelineOutput:
    images: Any
    seed: int
    latents: Any | None = None
    truncation_warnings: tuple[dict[str, int], ...] = ()
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class KreaTurboPipeline:
    scheduler: FlowMatchEulerDiscreteScheduler
    text_conditioner: Any
    transformer: Any
    vae: Any
    patch_size: int = 2
    vae_scale_factor: int = 8
    is_distilled: bool = True
    artifact_path: Path | None = None

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        dtype: Any | None = None,
        progress_callback: PipelineProgressCallback | None = None,
    ) -> "KreaTurboPipeline":
        root = Path(path).expanduser()
        _emit_progress(
            progress_callback,
            PipelineProgressEvent("load_start", "load", f"Loading artifact {root}", progress=0.0),
        )
        _validate_artifact_metadata(root)
        _validate_model_index(read_json_object(root / "model_index.json"))
        _emit_component_progress(progress_callback, "scheduler", start=True)
        scheduler = load_scheduler(root)
        _emit_component_progress(progress_callback, "scheduler", start=False)
        _emit_component_progress(progress_callback, "text_encoder", start=True)
        text_conditioner = load_text_conditioner(root, dtype=dtype)
        _emit_component_progress(progress_callback, "text_encoder", start=False)
        _emit_component_progress(progress_callback, "transformer", start=True)
        transformer = load_transformer(root, dtype=dtype)
        _emit_component_progress(progress_callback, "transformer", start=False)
        _emit_component_progress(progress_callback, "vae", start=True)
        vae = load_vae(root, dtype=dtype)
        _enable_vae_tiling(vae)
        _emit_component_progress(progress_callback, "vae", start=False)
        _emit_progress(
            progress_callback,
            PipelineProgressEvent("load_complete", "load", "Artifact loaded", progress=0.05),
        )
        return cls(
            scheduler=scheduler,
            text_conditioner=text_conditioner,
            transformer=transformer,
            vae=vae,
            patch_size=2,
            vae_scale_factor=int(getattr(vae, "spatial_compression_ratio", 8)),
            is_distilled=True,
            artifact_path=root,
        )

    def __call__(
        self,
        prompt: str,
        *,
        width: int = DEFAULT_GENERATION_WIDTH,
        height: int = DEFAULT_GENERATION_HEIGHT,
        steps: int = DEFAULT_GENERATION_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int | None = None,
        output_type: str = "pil",
        _latents: Any | None = None,
        loras: tuple[ResolvedLoraPatch, ...] | None = None,
        progress_callback: PipelineProgressCallback | None = None,
        live_preview: LivePreviewMode = "off",
        preview_interval_steps: int = DEFAULT_PREVIEW_INTERVAL_STEPS,
        preview_callback: PipelinePreviewCallback | None = None,
    ) -> KreaTurboPipelineOutput:
        try:
            return self._generate(
                prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_type=output_type,
                _latents=_latents,
                loras=loras,
                progress_callback=progress_callback,
                live_preview=live_preview,
                preview_interval_steps=preview_interval_steps,
                preview_callback=preview_callback,
            )
        finally:
            clear_runtime_caches()

    def _generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        output_type: str,
        _latents: Any | None,
        loras: tuple[ResolvedLoraPatch, ...] | None,
        progress_callback: PipelineProgressCallback | None,
        live_preview: LivePreviewMode,
        preview_interval_steps: int,
        preview_callback: PipelinePreviewCallback | None,
    ) -> KreaTurboPipelineOutput:
        start = time.perf_counter()
        live_preview = validate_live_preview_mode(live_preview)
        preview_interval_steps = validate_preview_interval_steps(preview_interval_steps)
        width, height = _validate_call_inputs(
            prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
        )
        _require_mlx()
        effective_seed = resolve_seed(seed)
        _seed_mlx(effective_seed)

        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                "encode_start",
                "encode",
                "Encoding prompt",
                progress=0.05,
            ),
        )
        conditioning = self.text_conditioner.encode(prompt)
        truncation_warnings = tuple(
            warning.to_dict() for warning in conditioning.truncation_warnings
        )
        for warning in truncation_warnings:
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    "prompt_truncated",
                    "encode",
                    "Prompt was truncated",
                    progress=0.08,
                    details=warning,
                ),
            )
        prompt_embeds = conditioning.hidden_states
        prompt_mask = conditioning.attention_mask
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                "encode_end",
                "encode",
                "Prompt encoded",
                progress=0.15,
                details={"sequence_length": int(prompt_embeds.shape[1])},
            ),
        )
        latents = self.prepare_latents(
            batch_size=1,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            latents=_latents,
        )
        grid_height, grid_width = self.image_grid_shape(height=height, width=width)
        position_ids = prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width)

        self.scheduler.set_timesteps(steps, mu=DEFAULT_DISTILLED_SHIFT)
        with applied_lora_patches(self.transformer, loras):
            for step_index in range(steps):
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        "denoise_step_start",
                        "denoise",
                        f"Denoising step {step_index + 1}/{steps}",
                        progress=0.15 + (0.7 * step_index / max(steps, 1)),
                        step_index=step_index,
                        step_count=steps,
                    ),
                )
                sigma = self.scheduler.sigmas[step_index]
                timestep = mx.array([sigma] * latents.shape[0], dtype=latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_mask,
                    return_dict=False,
                )
                noise_pred = _extract_sample(noise_pred)
                latents = self.scheduler.step(noise_pred, step_index, latents)
                mx.eval(latents)
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        "denoise_step_end",
                        "denoise",
                        f"Finished step {step_index + 1}/{steps}",
                        progress=0.15 + (0.7 * (step_index + 1) / max(steps, 1)),
                        step_index=step_index,
                        step_count=steps,
                        details={"sigma": float(sigma)},
                    ),
                )
                if _should_emit_preview(
                    live_preview=live_preview,
                    preview_interval_steps=preview_interval_steps,
                    preview_callback=preview_callback,
                    step_index=step_index,
                    step_count=steps,
                    output_type=output_type,
                ):
                    _emit_preview(
                        preview_callback,
                        PipelinePreviewFrame(
                            mode=live_preview,
                            step_index=step_index,
                            step_count=steps,
                            width=width,
                            height=height,
                            image=self.preview_latents(
                                latents,
                                mode=live_preview,
                                height=height,
                                width=width,
                            ),
                        ),
                    )

        if output_type == "latent":
            images = latents
            decoded_latents = None
        else:
            _emit_progress(
                progress_callback,
                PipelineProgressEvent("decode_start", "decode", "Decoding latents", progress=0.88),
            )
            decoded_latents = unpack_latents(
                latents,
                height=height,
                width=width,
                vae_scale_factor=self.vae_scale_factor,
                patch_size=self.patch_size,
            )
            decoded_latents = _astype(decoded_latents, _model_dtype(self.vae))
            decoded = self.vae.decode(decoded_latents, return_dict=False)[0]
            images = postprocess_decoded_image(decoded, output_type=output_type)
            _emit_progress(
                progress_callback,
                PipelineProgressEvent("decode_end", "decode", "Latents decoded", progress=0.97),
            )
        elapsed = time.perf_counter() - start
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                "complete",
                "complete",
                "Generation complete",
                progress=1.0,
                details={"elapsed_seconds": elapsed},
            ),
        )
        return KreaTurboPipelineOutput(
            images=images,
            seed=effective_seed,
            latents=None,
            truncation_warnings=truncation_warnings,
            elapsed_seconds=elapsed,
        )

    def prepare_latents(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        dtype: Any,
        latents: Any | None = None,
    ) -> Any:
        num_channels = self.transformer.config.in_channels // (self.patch_size**2)
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        grid_height, grid_width = self.image_grid_shape(height=height, width=width)
        expected_shape = (
            batch_size,
            grid_height * grid_width,
            self.transformer.config.in_channels,
        )
        if latents is not None:
            packed = _mx_array(latents, dtype=dtype)
            if tuple(packed.shape) != expected_shape:
                raise ValueError(
                    "packed latents must be shaped "
                    f"{expected_shape}; got {tuple(packed.shape)}"
                )
            return packed

        noise = _random_normal(
            (batch_size, num_channels, latent_height, latent_width),
            dtype=dtype,
        )
        return pack_latents(noise, patch_size=self.patch_size)

    def image_grid_shape(self, *, height: int, width: int) -> tuple[int, int]:
        factor = self.vae_scale_factor * self.patch_size
        return height // factor, width // factor

    def preview_latents(
        self,
        latents: Any,
        *,
        mode: LivePreviewMode,
        height: int,
        width: int,
    ) -> Any:
        mode = validate_live_preview_mode(mode)
        if mode == "latent":
            return latent_preview_rgb(
                latents,
                height=height,
                width=width,
                vae_scale_factor=self.vae_scale_factor,
                patch_size=self.patch_size,
            )
        if mode == "vae":
            return vae_preview_rgb(
                self.vae,
                latents,
                height=height,
                width=width,
                vae_scale_factor=self.vae_scale_factor,
                patch_size=self.patch_size,
            )
        raise ValueError("live preview mode 'off' does not produce preview frames")


def pack_latents(latents: Any, *, patch_size: int = 2) -> Any:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if len(latents.shape) != 4:
        raise ValueError("latents must be shaped [B, C, H, W]")
    batch_size, channels, height, width = latents.shape
    if height % patch_size or width % patch_size:
        raise ValueError("latent height and width must be divisible by patch_size")
    latents = latents.reshape(
        batch_size,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.transpose(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


def unpack_latents(
    latents: Any,
    *,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    patch_size: int = 2,
) -> Any:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor must be positive")
    if len(latents.shape) != 3:
        raise ValueError("latents must be shaped [B, image_seq_len, channels]")
    batch_size, sequence_length, packed_channels = latents.shape
    patch_area = patch_size * patch_size
    if packed_channels % patch_area:
        raise ValueError("packed channel count must be divisible by patch_size**2")

    latent_height = patch_size * (int(height) // (vae_scale_factor * patch_size))
    latent_width = patch_size * (int(width) // (vae_scale_factor * patch_size))
    grid_height = latent_height // patch_size
    grid_width = latent_width // patch_size
    if sequence_length != grid_height * grid_width:
        raise ValueError(
            "packed latent sequence length does not match requested image size"
        )

    channels = packed_channels // patch_area
    latents = latents.reshape(
        batch_size,
        grid_height,
        grid_width,
        channels,
        patch_size,
        patch_size,
    )
    latents = latents.transpose(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels, 1, latent_height, latent_width)


_WAN21_LATENT_RGB_FACTORS = (
    (-0.1299, -0.1692, 0.2932),
    (0.0671, 0.0406, 0.0442),
    (0.3568, 0.2548, 0.1747),
    (0.0372, 0.2344, 0.1420),
    (0.0313, 0.0189, -0.0328),
    (0.0296, -0.0956, -0.0665),
    (-0.3477, -0.4059, -0.2925),
    (0.0166, 0.1902, 0.1975),
    (-0.0412, 0.0267, -0.1364),
    (-0.1293, 0.0740, 0.1636),
    (0.0680, 0.3019, 0.1128),
    (0.0032, 0.0581, 0.0639),
    (-0.1251, 0.0927, 0.1699),
    (0.0060, -0.0633, 0.0005),
    (0.3477, 0.2275, 0.2950),
    (0.1984, 0.0913, 0.1861),
)
_WAN21_LATENT_RGB_BIAS = (-0.1835, -0.0868, -0.3360)


def latent_preview_rgb(
    latents: Any,
    *,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    patch_size: int = 2,
) -> Any:
    unpacked = unpack_latents(
        latents,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
        patch_size=patch_size,
    )
    return _project_unpacked_latents_to_rgb(unpacked)


def vae_preview_rgb(
    vae: Any,
    latents: Any,
    *,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    patch_size: int = 2,
) -> Any:
    decoded_latents = unpack_latents(
        latents,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
        patch_size=patch_size,
    )
    decoded_latents = _astype(decoded_latents, _model_dtype(vae))
    decoded = vae.decode(decoded_latents, return_dict=False)[0]
    return postprocess_decoded_image(decoded, output_type="np")[0]


def validate_live_preview_mode(mode: Any) -> LivePreviewMode:
    if mode not in LIVE_PREVIEW_MODES:
        raise ValueError("live_preview must be one of 'off', 'latent', or 'vae'")
    return mode


def validate_preview_interval_steps(interval: Any) -> int:
    if isinstance(interval, bool):
        raise ValueError("preview_interval_steps must be an integer")
    if isinstance(interval, int):
        parsed = interval
    elif isinstance(interval, str):
        stripped = interval.strip()
        if not stripped:
            raise ValueError("preview_interval_steps must be an integer")
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError("preview_interval_steps must be an integer") from exc
    else:
        raise ValueError("preview_interval_steps must be an integer")
    if parsed < 1 or parsed > MAX_PREVIEW_INTERVAL_STEPS:
        raise ValueError(
            f"preview_interval_steps must be from 1 to {MAX_PREVIEW_INTERVAL_STEPS}"
        )
    return parsed


def _project_unpacked_latents_to_rgb(latents: Any) -> Any:
    _require_mlx()
    if len(latents.shape) != 5:
        raise ValueError("unpacked latents must be shaped [B, C, 1, H, W]")
    if latents.shape[1] != len(_WAN21_LATENT_RGB_FACTORS):
        raise ValueError("latent preview requires 16 latent channels")
    if latents.shape[2] != 1:
        raise ValueError("latent preview supports only single-frame latents")

    factors = mx.array(_WAN21_LATENT_RGB_FACTORS, dtype=latents.dtype)
    bias = mx.array(_WAN21_LATENT_RGB_BIAS, dtype=latents.dtype)
    image = latents[0, :, 0].transpose(1, 2, 0)
    rgb = image @ factors + bias
    rgb = (rgb + 1.0) / 2.0
    mx.eval(rgb)

    import numpy as np

    return np.clip(np.array(rgb, dtype=np.float32), 0.0, 1.0)


def _should_emit_preview(
    *,
    live_preview: LivePreviewMode,
    preview_interval_steps: int,
    preview_callback: PipelinePreviewCallback | None,
    step_index: int,
    step_count: int,
    output_type: str,
) -> bool:
    if live_preview == "off" or preview_callback is None:
        return False
    completed_step = step_index + 1
    if completed_step % preview_interval_steps != 0:
        return False
    if output_type != "latent" and completed_step == step_count:
        return False
    return True


def resolve_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbelow(MAX_GENERATION_SEED + 1)
    if not 0 <= seed <= MAX_GENERATION_SEED:
        raise ValueError(f"seed must be from 0 to {MAX_GENERATION_SEED}")
    return int(seed)


def _validate_artifact_metadata(root: Path) -> None:
    artifact_path = root / "artifact.json"
    if not artifact_path.is_file():
        raise Krea2TurboMlxError(
            f"Missing converted artifact metadata: {artifact_path}. "
            "Run `./setup.sh` or `krea-2-turbo-mlx convert` before using this path as --model."
        )
    artifact = read_json_object(artifact_path)
    if artifact.get("schema_version") != 2:
        raise Krea2TurboMlxError("artifact.json must use schema_version=2")
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise Krea2TurboMlxError(
            f"artifact.json has unsupported format: {artifact.get('format')}"
        )
    if artifact.get("full_precision_only") is not FULL_PRECISION_ONLY:
        raise Krea2TurboMlxError(
            "Krea 2 Turbo generation requires a full-precision converted artifact"
        )
    if artifact.get("selection_policy_version") != SELECTION_POLICY_VERSION:
        raise Krea2TurboMlxError(
            "artifact.json has unsupported selection_policy_version: "
            f"{artifact.get('selection_policy_version')}"
        )
    if "quantization" in artifact:
        raise Krea2TurboMlxError(
            "This build supports only full-precision artifacts; quantized artifacts are not supported."
        )

    precision = artifact.get("precision")
    if not isinstance(precision, Mapping):
        raise Krea2TurboMlxError("artifact.json precision proof is missing")
    if precision.get("preserves_source_dtypes") is not True:
        raise Krea2TurboMlxError(
            "artifact.json does not prove source dtype preservation"
        )
    if precision.get("dtype_equivalence_verified") is not True:
        raise Krea2TurboMlxError(
            "artifact.json does not prove written tensor dtype equivalence"
        )
    if precision.get("quantized_dtypes_present") is not False:
        raise Krea2TurboMlxError("artifact.json declares quantized tensor dtypes")


def _validate_model_index(payload: dict[str, Any]) -> None:
    if payload.get("_class_name") != "Krea2Pipeline":
        raise Krea2TurboMlxError("model_index.json must describe Krea2Pipeline")
    for component, expected in EXPECTED_COMPONENT_CLASSES.items():
        if payload.get(component) != expected:
            raise Krea2TurboMlxError(
                f"model_index.json has unsupported {component}: {payload.get(component)}"
            )
    if payload.get("is_distilled") is not True:
        raise Krea2TurboMlxError("Krea 2 Turbo generation requires is_distilled=true")
    if int(payload.get("patch_size", 0)) != 2:
        raise Krea2TurboMlxError("Krea 2 Turbo generation requires patch_size=2")


def _validate_call_inputs(
    prompt: str,
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
) -> tuple[int, int]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    width, height = validate_generation_dimensions(width, height)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if guidance_scale != DEFAULT_GUIDANCE_SCALE:
        raise ValueError("Krea 2 Turbo runs at guidance_scale=0.0 only")
    return width, height


def _extract_sample(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    sample = getattr(output, "sample", None)
    if sample is not None:
        return sample
    return output


def _model_dtype(model: Any) -> Any | None:
    post_quant = getattr(model, "post_quant_conv", None)
    weight = getattr(post_quant, "weight", None)
    return getattr(weight, "dtype", None)


def _enable_vae_tiling(vae: Any) -> None:
    enable_tiling = getattr(vae, "enable_tiling", None)
    if callable(enable_tiling):
        enable_tiling()


def _mx_array(value: Any, *, dtype: Any | None = None) -> Any:
    _require_mlx()
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype=dtype)


def _astype(value: Any, dtype: Any | None) -> Any:
    if dtype is None or not hasattr(value, "astype"):
        return value
    return value.astype(dtype)


def _seed_mlx(seed: int) -> None:
    _require_mlx()
    mx.random.seed(int(seed))


def _random_normal(shape: tuple[int, ...], *, dtype: Any) -> Any:
    _require_mlx()
    try:
        return mx.random.normal(shape, dtype=dtype)
    except TypeError:
        return mx.random.normal(shape).astype(dtype)


def _require_mlx() -> None:
    if mx is None:
        raise Krea2TurboMlxError(
            "Krea 2 Turbo generation requires MLX. Install "
            "`krea-2-turbo-mlx[runtime]` on an MLX-supported machine."
        )


def clear_runtime_caches() -> None:
    gc.collect()
    if mx is None:
        return
    for owner in (mx, getattr(mx, "metal", None)):
        clear = getattr(owner, "clear_cache", None)
        if callable(clear):
            clear()


def _emit_component_progress(
    callback: PipelineProgressCallback | None,
    component: str,
    *,
    start: bool,
) -> None:
    kind: PipelineProgressKind = "load_component_start" if start else "load_component_end"
    verb = "Loading" if start else "Loaded"
    _emit_progress(
        callback,
        PipelineProgressEvent(
            kind,
            "load",
            f"{verb} {component}",
            details={"component": component},
        ),
    )


def _emit_progress(
    callback: PipelineProgressCallback | None,
    event: PipelineProgressEvent,
) -> None:
    if callback is not None:
        callback(event)


def _emit_preview(
    callback: PipelinePreviewCallback | None,
    frame: PipelinePreviewFrame,
) -> None:
    if callback is not None:
        callback(frame)


__all__ = [
    "DEFAULT_PREVIEW_INTERVAL_STEPS",
    "KreaTurboPipeline",
    "KreaTurboPipelineOutput",
    "LIVE_PREVIEW_MODES",
    "MAX_PREVIEW_INTERVAL_STEPS",
    "PipelinePreviewFrame",
    "PipelineProgressEvent",
    "clear_runtime_caches",
    "latent_preview_rgb",
    "pack_latents",
    "resolve_seed",
    "validate_live_preview_mode",
    "validate_preview_interval_steps",
    "vae_preview_rgb",
    "unpack_latents",
]
