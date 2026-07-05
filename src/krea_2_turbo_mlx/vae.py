from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._weights import flatten_parameter_shapes, load_mapped_safetensors_weights
from .errors import Krea2TurboMlxError
from .json_io import read_json_object

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None
    nn = None

_MlxModuleBase = object if nn is None else nn.Module

_PINNED_LATENTS_MEAN = (
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
)
_PINNED_LATENTS_STD = (
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
)
_DEFAULT_TILE_SAMPLE_MIN_HEIGHT = 256
_DEFAULT_TILE_SAMPLE_MIN_WIDTH = 256
# Stride = tile * (1 - overlap). overlap=0.125 (stride 224) was chosen from the
# tiled-decode benchmark: it is visually seam-free (the seam only shows at overlap 0)
# while running ~25% faster than the previous overlap=0.25 (stride 192) default.
# This intentionally diverges from diffusers' default stride; plain enable_tiling()
# is no longer pixel-identical to diffusers' default tiling unless both decoders
# are pinned to the same stride.
_DEFAULT_TILE_SAMPLE_STRIDE_HEIGHT = 224
_DEFAULT_TILE_SAMPLE_STRIDE_WIDTH = 224


@dataclass(frozen=True)
class QwenImageVAEDecodeOutput:
    sample: Any


@dataclass(frozen=True)
class QwenImageVAEConfig:
    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    input_channels: int = 3
    latents_mean: tuple[float, ...] = _PINNED_LATENTS_MEAN
    latents_std: tuple[float, ...] = _PINNED_LATENTS_STD

    @classmethod
    def from_model_config(cls, payload: Mapping[str, Any]) -> "QwenImageVAEConfig":
        if payload.get("_class_name") != "AutoencoderKLQwenImage":
            raise ValueError("Qwen Image VAE requires _class_name='AutoencoderKLQwenImage'")
        config = cls(
            base_dim=int(payload.get("base_dim", cls.base_dim)),
            z_dim=int(payload.get("z_dim", cls.z_dim)),
            dim_mult=_tuple_ints(payload.get("dim_mult", cls.dim_mult), "dim_mult"),
            num_res_blocks=int(payload.get("num_res_blocks", cls.num_res_blocks)),
            attn_scales=_tuple_floats(
                payload.get("attn_scales", cls.attn_scales),
                "attn_scales",
            ),
            temperal_downsample=_tuple_bools(
                payload.get("temperal_downsample", cls.temperal_downsample),
                "temperal_downsample",
            ),
            dropout=float(payload.get("dropout", cls.dropout)),
            input_channels=int(payload.get("input_channels", cls.input_channels)),
            latents_mean=_tuple_floats(
                payload.get("latents_mean", cls.latents_mean),
                "latents_mean",
            ),
            latents_std=_tuple_floats(
                payload.get("latents_std", cls.latents_std),
                "latents_std",
            ),
        )
        config.validate()
        return config

    @property
    def temperal_upsample(self) -> tuple[bool, ...]:
        return self.temperal_downsample[::-1]

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** len(self.temperal_downsample)

    def validate(self) -> None:
        _require_positive(self.base_dim, "base_dim")
        _require_positive(self.z_dim, "z_dim")
        _require_positive(self.input_channels, "input_channels")
        if not self.dim_mult:
            raise ValueError("dim_mult must not be empty")
        if any(mult <= 0 for mult in self.dim_mult):
            raise ValueError("dim_mult entries must be positive")
        if self.num_res_blocks < 0:
            raise ValueError("num_res_blocks must be non-negative")
        if self.attn_scales:
            raise ValueError("Qwen Image VAE decode supports only attn_scales=[]")
        if self.dropout != 0.0:
            raise ValueError("Qwen Image VAE decode supports only dropout=0")
        if len(self.temperal_downsample) != len(self.dim_mult) - 1:
            raise ValueError("temperal_downsample length must equal len(dim_mult) - 1")
        if len(self.latents_mean) != self.z_dim:
            raise ValueError("latents_mean length must equal z_dim")
        if len(self.latents_std) != self.z_dim:
            raise ValueError("latents_std length must equal z_dim")
        if any(std <= 0 for std in self.latents_std):
            raise ValueError("latents_std entries must be positive")


class QwenImageCausalConv3d(_MlxModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        *,
        bias: bool = True,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuple3(kernel_size)
        self.stride = _tuple3(stride)
        self.padding = _tuple3(padding)
        self.dilation = (1, 1, 1)
        scale = math.sqrt(
            1
            / (
                in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
        )
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *self.kernel_size, in_channels),
            dtype=mx.float32,
        )
        if bias:
            self.bias = mx.zeros((out_channels,), dtype=mx.float32)

    def __call__(self, hidden_states: Any) -> Any:
        pad_t, pad_h, pad_w = self.padding
        if pad_t or pad_h or pad_w:
            hidden_states = mx.pad(
                hidden_states,
                [
                    (0, 0),
                    (2 * pad_t, 0),
                    (pad_h, pad_h),
                    (pad_w, pad_w),
                    (0, 0),
                ],
            )
        hidden_states = mx.conv3d(
            hidden_states,
            self.weight,
            self.stride,
            (0, 0, 0),
            self.dilation,
        )
        if hasattr(self, "bias"):
            hidden_states = hidden_states + self.bias
        return hidden_states


class QwenImageRMSNorm(_MlxModuleBase):
    def __init__(self, dim: int) -> None:
        _require_mlx()
        super().__init__()
        self.dim = dim
        self.scale = math.sqrt(dim)
        self.gamma = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, hidden_states: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.astype(mx.float32)
        norm = mx.sqrt(mx.sum(mx.square(hidden_float), axis=-1, keepdims=True))
        norm = mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
        hidden_states = (hidden_float / norm).astype(input_dtype)
        return hidden_states * self.scale * self.gamma.astype(input_dtype)


class QwenImageUpsample2d(_MlxModuleBase):
    def __init__(self, scale_factor: int = 2) -> None:
        _require_mlx()
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = mx.repeat(hidden_states, self.scale_factor, axis=1)
        return mx.repeat(hidden_states, self.scale_factor, axis=2)


class QwenImageResample(_MlxModuleBase):
    def __init__(self, dim: int, mode: str) -> None:
        _require_mlx()
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode in {"upsample2d", "upsample3d"}:
            self.resample = [
                QwenImageUpsample2d(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            ]
            if mode == "upsample3d":
                self.time_conv = QwenImageCausalConv3d(
                    dim,
                    dim * 2,
                    (3, 1, 1),
                    padding=(1, 0, 0),
                )
        elif mode == "none":
            self.resample = []
        else:
            raise ValueError(f"Unsupported Qwen Image VAE resample mode: {mode}")

    def __call__(self, hidden_states: Any) -> Any:
        if self.mode == "none":
            return hidden_states

        batch_size, frames, height, width, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * frames, height, width, channels)
        for layer in self.resample:
            hidden_states = layer(hidden_states)
        return hidden_states.reshape(
            batch_size,
            frames,
            hidden_states.shape[1],
            hidden_states.shape[2],
            hidden_states.shape[3],
        )


class QwenImageResidualBlock(_MlxModuleBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        _require_mlx()
        super().__init__()
        if dropout != 0.0:
            raise ValueError("Qwen Image VAE decode supports only dropout=0")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = QwenImageRMSNorm(in_dim)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else None
        )

    def __call__(self, hidden_states: Any) -> Any:
        residual = (
            self.conv_shortcut(hidden_states)
            if self.conv_shortcut is not None
            else hidden_states
        )
        hidden_states = self.conv1(_silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(_silu(self.norm2(hidden_states)))
        return hidden_states + residual


class QwenImageAttentionBlock(_MlxModuleBase):
    def __init__(self, dim: int) -> None:
        _require_mlx()
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, hidden_states: Any) -> Any:
        identity = hidden_states
        batch_size, frames, height, width, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * frames, height, width, channels)
        hidden_states = self.norm(hidden_states)

        qkv = self.to_qkv(hidden_states).reshape(
            batch_size * frames,
            height * width,
            channels * 3,
        )
        query = qkv[..., :channels]
        key = qkv[..., channels : 2 * channels]
        value = qkv[..., 2 * channels :]
        hidden_states = mx.fast.scaled_dot_product_attention(
            query[:, None, :, :],
            key[:, None, :, :],
            value[:, None, :, :],
            scale=channels**-0.5,
        )
        hidden_states = hidden_states[:, 0, :, :].reshape(
            batch_size * frames,
            height,
            width,
            channels,
        )
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, frames, height, width, channels)
        return hidden_states + identity


class QwenImageMidBlock(_MlxModuleBase):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        *,
        num_layers: int = 1,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.resnets = [QwenImageResidualBlock(dim, dim, dropout)]
        self.attentions = []
        for _ in range(num_layers):
            self.attentions.append(QwenImageAttentionBlock(dim))
            self.resnets.append(QwenImageResidualBlock(dim, dim, dropout))

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = self.resnets[0](hidden_states)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attention(hidden_states)
            hidden_states = resnet(hidden_states)
        return hidden_states


class QwenImageUpBlock(_MlxModuleBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        *,
        upsample_mode: str | None = None,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        current_dim = in_dim
        self.resnets = []
        for _ in range(num_res_blocks + 1):
            self.resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.upsamplers = (
            [QwenImageResample(out_dim, mode=upsample_mode)]
            if upsample_mode is not None
            else None
        )

    def __call__(self, hidden_states: Any) -> Any:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class QwenImageDecoder3d(_MlxModuleBase):
    def __init__(self, config: QwenImageVAEConfig) -> None:
        _require_mlx()
        super().__init__()
        self.config = config
        dims = [
            config.base_dim * mult
            for mult in (config.dim_mult[-1], *config.dim_mult[::-1])
        ]
        self.conv_in = QwenImageCausalConv3d(config.z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], config.dropout, num_layers=1)
        self.up_blocks = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if index > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if index != len(config.dim_mult) - 1:
                upsample_mode = (
                    "upsample3d"
                    if config.temperal_upsample[index]
                    else "upsample2d"
                )
            self.up_blocks.append(
                QwenImageUpBlock(
                    in_dim,
                    out_dim,
                    config.num_res_blocks,
                    config.dropout,
                    upsample_mode=upsample_mode,
                )
            )
        self.norm_out = QwenImageRMSNorm(dims[-1])
        self.conv_out = QwenImageCausalConv3d(
            dims[-1],
            config.input_channels,
            3,
            padding=1,
        )

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)
        hidden_states = _silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class AutoencoderKLQwenImage(_MlxModuleBase):
    def __init__(self, config: QwenImageVAEConfig | None = None) -> None:
        if nn is not None:
            super().__init__()
        self.config = config or QwenImageVAEConfig()
        self.config.validate()
        self.z_dim = self.config.z_dim
        self.temperal_downsample = self.config.temperal_downsample
        self.temperal_upsample = self.config.temperal_upsample
        self.spatial_compression_ratio = self.config.spatial_compression_ratio
        self.use_tiling = False
        self.tile_sample_min_height = _DEFAULT_TILE_SAMPLE_MIN_HEIGHT
        self.tile_sample_min_width = _DEFAULT_TILE_SAMPLE_MIN_WIDTH
        self.tile_sample_stride_height = _DEFAULT_TILE_SAMPLE_STRIDE_HEIGHT
        self.tile_sample_stride_width = _DEFAULT_TILE_SAMPLE_STRIDE_WIDTH
        if nn is None:
            return

        self.post_quant_conv = QwenImageCausalConv3d(self.config.z_dim, self.config.z_dim, 1)
        self.decoder = QwenImageDecoder3d(self.config)

    def enable_tiling(
        self,
        *,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
    ) -> None:
        next_min_height = _positive_or_existing(
            tile_sample_min_height,
            self.tile_sample_min_height,
            "tile_sample_min_height",
        )
        next_min_width = _positive_or_existing(
            tile_sample_min_width,
            self.tile_sample_min_width,
            "tile_sample_min_width",
        )
        next_stride_height = _positive_or_existing(
            tile_sample_stride_height,
            self.tile_sample_stride_height,
            "tile_sample_stride_height",
        )
        next_stride_width = _positive_or_existing(
            tile_sample_stride_width,
            self.tile_sample_stride_width,
            "tile_sample_stride_width",
        )
        if next_stride_height > next_min_height:
            raise ValueError("tile_sample_stride_height must not exceed tile_sample_min_height")
        if next_stride_width > next_min_width:
            raise ValueError("tile_sample_stride_width must not exceed tile_sample_min_width")
        self.use_tiling = True
        self.tile_sample_min_height = next_min_height
        self.tile_sample_min_width = next_min_width
        self.tile_sample_stride_height = next_stride_height
        self.tile_sample_stride_width = next_stride_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        dtype: Any | None = None,
    ) -> "AutoencoderKLQwenImage":
        return load_vae(path, dtype=dtype)

    def decode(
        self,
        latents: Any,
        *,
        return_dict: bool = True,
    ) -> QwenImageVAEDecodeOutput | tuple[Any]:
        _require_mlx()
        latents = denormalize_latents(latents, self.config)
        if self.use_tiling and self._should_tile(latents):
            sample = self._tiled_decode_denormalized(latents)
        else:
            sample = self._decode_denormalized(latents)
        sample = mx.clip(sample, -1.0, 1.0)
        if not return_dict:
            return (sample,)
        return QwenImageVAEDecodeOutput(sample=sample)

    def _decode_denormalized(self, latents: Any) -> Any:
        hidden_states = latents.transpose(0, 2, 3, 4, 1)
        hidden_states = self.post_quant_conv(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states.transpose(0, 4, 1, 2, 3)

    def _should_tile(self, latents: Any) -> bool:
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        return latents.shape[3] > tile_latent_min_height or latents.shape[4] > tile_latent_min_width

    def _tiled_decode_denormalized(self, latents: Any) -> Any:
        _, _, frames, height, width = latents.shape
        if frames != 1:
            raise ValueError("Qwen Image VAE tiled decode supports only single-frame latents")

        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        _require_positive(tile_latent_min_height, "tile_latent_min_height")
        _require_positive(tile_latent_min_width, "tile_latent_min_width")
        _require_positive(tile_latent_stride_height, "tile_latent_stride_height")
        _require_positive(tile_latent_stride_width, "tile_latent_stride_width")

        # Cache/eval policy (chosen from the tiled-decode benchmark; neither affects
        # output pixels, only speed/peak-memory):
        #   * cache: clear once per row (at row start), not per tile -- per-tile clearing
        #     cost ~13% decode time for no peak-memory benefit; per-row costs ~1%.
        #   * eval: defer all materialization to a single mx.eval at the very end --
        #     fastest, with negligible/inconsistent peak-memory difference.
        rows = []
        for top in range(0, height, tile_latent_stride_height):
            _clear_mlx_cache()
            row = []
            for left in range(0, width, tile_latent_stride_width):
                tile = latents[
                    :,
                    :,
                    :,
                    top : top + tile_latent_min_height,
                    left : left + tile_latent_min_width,
                ]
                row.append(self._decode_denormalized(tile))
            rows.append(row)

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width
        result_rows = []
        for row_index, row in enumerate(rows):
            result_row = []
            for col_index, tile in enumerate(row):
                if row_index > 0:
                    tile = _blend_vertical(rows[row_index - 1][col_index], tile, blend_height)
                if col_index > 0:
                    tile = _blend_horizontal(row[col_index - 1], tile, blend_width)
                row[col_index] = tile
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : self.tile_sample_stride_height,
                        : self.tile_sample_stride_width,
                    ]
                )
            result_rows.append(mx.concatenate(result_row, axis=-1))

        decoded = mx.concatenate(result_rows, axis=3)[
            :,
            :,
            :,
            :sample_height,
            :sample_width,
        ]
        mx.eval(decoded)
        return decoded

    def __call__(
        self,
        latents: Any,
        *,
        return_dict: bool = True,
    ) -> QwenImageVAEDecodeOutput | tuple[Any]:
        return self.decode(latents, return_dict=return_dict)


def load_vae(
    path: str | Path,
    *,
    dtype: Any | None = None,
) -> AutoencoderKLQwenImage:
    _require_mlx()
    root = Path(path).expanduser()
    vae_dir = _resolve_vae_dir(root)
    config = QwenImageVAEConfig.from_model_config(read_json_object(vae_dir / "config.json"))
    model = AutoencoderKLQwenImage(config)
    expected_shapes = flatten_parameter_shapes(model.parameters())
    weights = load_vae_weights(vae_dir, expected_shapes=expected_shapes, dtype=dtype)
    model.load_weights(sorted(weights.items()), strict=True)
    mx.eval(*weights.values())
    return model


def load_vae_weights(
    vae_dir: str | Path,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
    dtype: Any | None = None,
) -> dict[str, Any]:
    _require_mlx()

    def transform_value(local_key: str, array: Any) -> Any:
        return transform_vae_weight_value(local_key, array, expected_shapes=expected_shapes)

    return load_mapped_safetensors_weights(
        vae_dir,
        expected_shapes=expected_shapes,
        map_key=map_vae_weight_key,
        transform_value=transform_value,
        index_filename="diffusion_pytorch_model.safetensors.index.json",
        single_filename="diffusion_pytorch_model.safetensors",
        label="VAE",
        dtype=dtype,
    )


def map_vae_weight_key(key: str) -> str | None:
    if key.startswith("encoder.") or key.startswith("quant_conv."):
        return None
    if key.startswith("decoder.") or key.startswith("post_quant_conv."):
        return key
    raise Krea2TurboMlxError(f"Unexpected VAE tensor {key!r}")


def transform_vae_weight_value(
    local_key: str,
    array: Any,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> Any:
    expected_shape = expected_shapes.get(local_key)
    if expected_shape is None:
        return array
    actual_shape = tuple(array.shape)
    if actual_shape == expected_shape:
        return array
    if local_key.endswith(".weight"):
        if _is_conv3d_shape_pair(actual_shape, expected_shape):
            return array.transpose(0, 2, 3, 4, 1)
        if _is_conv2d_shape_pair(actual_shape, expected_shape):
            return array.transpose(0, 2, 3, 1)
    if local_key.endswith(".gamma") and len(expected_shape) == 1:
        if actual_shape and actual_shape[0] == expected_shape[0]:
            return array.reshape((actual_shape[0],))
    return array


def denormalize_latents(latents: Any, config: QwenImageVAEConfig | None = None) -> Any:
    _require_mlx()
    config = config or QwenImageVAEConfig()
    latents = _normalize_latents_shape(latents, config.z_dim)
    mean = mx.array(config.latents_mean, dtype=latents.dtype).reshape(
        1,
        config.z_dim,
        1,
        1,
        1,
    )
    std = mx.array(config.latents_std, dtype=latents.dtype).reshape(
        1,
        config.z_dim,
        1,
        1,
        1,
    )
    return latents * std + mean


def decode_latents(
    vae: AutoencoderKLQwenImage,
    latents: Any,
    *,
    output_type: str = "mlx",
) -> Any:
    decoded = vae.decode(latents).sample
    if output_type in {"latent", "raw"}:
        return decoded
    return postprocess_decoded_image(decoded, output_type=output_type)


def postprocess_decoded_image(decoded: Any, *, output_type: str = "pil") -> Any:
    _require_mlx()
    if decoded.ndim == 5:
        if decoded.shape[2] != 1:
            raise ValueError("decoded image must contain a single temporal frame")
        decoded = mx.squeeze(decoded, axis=2)
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError("decoded image must be shaped [B, 3, H, W] or [B, 3, 1, H, W]")
    image = mx.clip(decoded / 2.0 + 0.5, 0.0, 1.0)
    if output_type == "mlx":
        return image
    if output_type not in {"np", "pil"}:
        raise ValueError("output_type must be one of 'mlx', 'np', or 'pil'")

    import numpy as np

    image = image.transpose(0, 2, 3, 1)
    mx.eval(image)
    image_np = np.array(image, dtype=np.float32)
    if output_type == "np":
        return image_np

    from PIL import Image

    image_u8 = (image_np * 255.0).round().astype("uint8")
    return [Image.fromarray(item) for item in image_u8]


def _resolve_vae_dir(root: Path) -> Path:
    if (root / "vae").is_dir():
        return root / "vae"
    if (root / "config.json").is_file():
        return root
    raise Krea2TurboMlxError(
        f"VAE path must be an artifact/source root or vae directory: {root}"
    )


def _normalize_latents_shape(latents: Any, z_dim: int) -> Any:
    if latents.ndim == 4:
        latents = latents[:, :, None, :, :]
    if latents.ndim != 5:
        raise ValueError("latents must be shaped [B, C, H, W] or [B, C, 1, H, W]")
    if latents.shape[1] != z_dim:
        raise ValueError("latents channel dimension must match z_dim")
    if latents.shape[2] != 1:
        raise ValueError("Qwen Image VAE decode supports only single-frame latents")
    return latents


def _is_conv3d_shape_pair(
    official_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> bool:
    return (
        len(official_shape) == 5
        and len(local_shape) == 5
        and official_shape[0] == local_shape[0]
        and official_shape[1] == local_shape[4]
        and official_shape[2:] == local_shape[1:4]
    )


def _is_conv2d_shape_pair(
    official_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> bool:
    return (
        len(official_shape) == 4
        and len(local_shape) == 4
        and official_shape[0] == local_shape[0]
        and official_shape[1] == local_shape[3]
        and official_shape[2:] == local_shape[1:3]
    )


def _silu(hidden_states: Any) -> Any:
    return hidden_states * mx.sigmoid(hidden_states)


def _tuple_ints(value: Iterable[Any], name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} must be a sequence")
    items = tuple(int(item) for item in value)
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


def _tuple_floats(value: Iterable[Any], name: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"{name} must be a sequence")
    items = tuple(float(item) for item in value)
    return items


def _tuple_bools(value: Iterable[Any], name: str) -> tuple[bool, ...]:
    if value is None:
        raise ValueError(f"{name} must be a sequence")
    return tuple(bool(item) for item in value)


def _tuple3(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError("3D convolution parameters must contain three integers")
    return items


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _positive_or_existing(value: int | None, existing: int, name: str) -> int:
    if value is None:
        return existing
    parsed = int(value)
    _require_positive(parsed, name)
    return parsed


def _blend_vertical(above: Any, current: Any, extent: int) -> Any:
    if extent <= 0:
        return current
    extent = min(above.shape[3], current.shape[3], extent)
    if extent <= 0:
        return current
    weights = (
        mx.arange(extent, dtype=mx.float32).reshape(1, 1, 1, extent, 1)
        / float(extent)
    ).astype(current.dtype)
    blended = (
        above[:, :, :, above.shape[3] - extent :, :] * (1.0 - weights)
        + current[:, :, :, :extent, :] * weights
    )
    return mx.concatenate([blended, current[:, :, :, extent:, :]], axis=3)


def _blend_horizontal(left: Any, current: Any, extent: int) -> Any:
    if extent <= 0:
        return current
    extent = min(left.shape[4], current.shape[4], extent)
    if extent <= 0:
        return current
    weights = (
        mx.arange(extent, dtype=mx.float32).reshape(1, 1, 1, 1, extent)
        / float(extent)
    ).astype(current.dtype)
    blended = (
        left[:, :, :, :, left.shape[4] - extent :] * (1.0 - weights)
        + current[:, :, :, :, :extent] * weights
    )
    return mx.concatenate([blended, current[:, :, :, :, extent:]], axis=4)


def _clear_mlx_cache() -> None:
    if mx is not None:
        mx.clear_cache()


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise Krea2TurboMlxError(
            "Qwen Image VAE decode requires MLX. Install `krea-2-turbo-mlx[runtime]` "
            "on an MLX-supported machine."
        )


__all__ = [
    "AutoencoderKLQwenImage",
    "QwenImageAttentionBlock",
    "QwenImageCausalConv3d",
    "QwenImageDecoder3d",
    "QwenImageMidBlock",
    "QwenImageRMSNorm",
    "QwenImageResample",
    "QwenImageVAEDecodeOutput",
    "QwenImageVAEConfig",
    "decode_latents",
    "denormalize_latents",
    "flatten_parameter_shapes",
    "load_vae",
    "load_vae_weights",
    "map_vae_weight_key",
    "postprocess_decoded_image",
    "transform_vae_weight_value",
]
