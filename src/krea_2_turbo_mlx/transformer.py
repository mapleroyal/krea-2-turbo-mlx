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


@dataclass(frozen=True)
class Krea2TransformerOutput:
    sample: Any


@dataclass(frozen=True)
class Krea2TransformerConfig:
    in_channels: int = 64
    num_layers: int = 28
    attention_head_dim: int = 128
    num_attention_heads: int = 48
    num_key_value_heads: int = 12
    intermediate_size: int = 16384
    timestep_embed_dim: int = 256
    text_hidden_dim: int = 2560
    num_text_layers: int = 12
    text_num_attention_heads: int = 20
    text_num_key_value_heads: int = 20
    text_intermediate_size: int = 6912
    num_layerwise_text_blocks: int = 2
    num_refiner_text_blocks: int = 2
    axes_dims_rope: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 1000.0
    norm_eps: float = 1e-5

    @classmethod
    def from_model_config(cls, payload: Mapping[str, Any]) -> "Krea2TransformerConfig":
        if payload.get("_class_name") != "Krea2Transformer2DModel":
            raise ValueError("Krea2 transformer requires _class_name='Krea2Transformer2DModel'")
        config = cls(
            in_channels=int(payload.get("in_channels", cls.in_channels)),
            num_layers=int(payload.get("num_layers", cls.num_layers)),
            attention_head_dim=int(
                payload.get("attention_head_dim", cls.attention_head_dim)
            ),
            num_attention_heads=int(
                payload.get("num_attention_heads", cls.num_attention_heads)
            ),
            num_key_value_heads=int(
                payload.get("num_key_value_heads", cls.num_key_value_heads)
            ),
            intermediate_size=int(payload.get("intermediate_size", cls.intermediate_size)),
            timestep_embed_dim=int(
                payload.get("timestep_embed_dim", cls.timestep_embed_dim)
            ),
            text_hidden_dim=int(payload.get("text_hidden_dim", cls.text_hidden_dim)),
            num_text_layers=int(payload.get("num_text_layers", cls.num_text_layers)),
            text_num_attention_heads=int(
                payload.get("text_num_attention_heads", cls.text_num_attention_heads)
            ),
            text_num_key_value_heads=int(
                payload.get("text_num_key_value_heads", cls.text_num_key_value_heads)
            ),
            text_intermediate_size=int(
                payload.get("text_intermediate_size", cls.text_intermediate_size)
            ),
            num_layerwise_text_blocks=int(
                payload.get(
                    "num_layerwise_text_blocks",
                    cls.num_layerwise_text_blocks,
                )
            ),
            num_refiner_text_blocks=int(
                payload.get("num_refiner_text_blocks", cls.num_refiner_text_blocks)
            ),
            axes_dims_rope=_tuple_ints3(
                payload.get("axes_dims_rope", cls.axes_dims_rope),
                "axes_dims_rope",
            ),
            rope_theta=float(payload.get("rope_theta", cls.rope_theta)),
            norm_eps=float(payload.get("norm_eps", cls.norm_eps)),
        )
        if "hidden_size" in payload and int(payload["hidden_size"]) != config.hidden_size:
            raise ValueError("hidden_size must equal attention_head_dim * num_attention_heads")
        config.validate()
        return config

    @property
    def hidden_size(self) -> int:
        return self.attention_head_dim * self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate(self) -> None:
        _require_positive(self.in_channels, "in_channels")
        _require_positive(self.num_layers, "num_layers")
        _require_positive(self.attention_head_dim, "attention_head_dim")
        _require_positive(self.num_attention_heads, "num_attention_heads")
        _require_positive(self.num_key_value_heads, "num_key_value_heads")
        _require_positive(self.intermediate_size, "intermediate_size")
        _require_positive(self.timestep_embed_dim, "timestep_embed_dim")
        _require_positive(self.text_hidden_dim, "text_hidden_dim")
        _require_positive(self.num_text_layers, "num_text_layers")
        _require_positive(self.text_num_attention_heads, "text_num_attention_heads")
        _require_positive(self.text_num_key_value_heads, "text_num_key_value_heads")
        _require_positive(self.text_intermediate_size, "text_intermediate_size")
        if self.num_layerwise_text_blocks < 0:
            raise ValueError("num_layerwise_text_blocks must be non-negative")
        if self.num_refiner_text_blocks < 0:
            raise ValueError("num_refiner_text_blocks must be non-negative")
        if self.timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be even")
        if self.attention_head_dim % 2:
            raise ValueError("attention_head_dim must be even")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_size != self.attention_head_dim * self.num_attention_heads:
            raise ValueError("hidden_size must equal attention_head_dim * num_attention_heads")
        if self.text_hidden_dim % self.text_num_attention_heads:
            raise ValueError("text_hidden_dim must be divisible by text_num_attention_heads")
        if self.text_num_attention_heads % self.text_num_key_value_heads:
            raise ValueError(
                "text_num_attention_heads must be divisible by text_num_key_value_heads"
            )
        if any(axis <= 0 or axis % 2 for axis in self.axes_dims_rope):
            raise ValueError("axes_dims_rope entries must be positive even integers")
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError("sum(axes_dims_rope) must equal attention_head_dim")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")


class Krea2RMSNorm(_MlxModuleBase):
    def __init__(self, dim: int, *, eps: float) -> None:
        _require_mlx()
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, hidden_states: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_float), axis=-1, keepdims=True)
        hidden_float = hidden_float * mx.rsqrt(variance + self.eps)
        hidden_float = hidden_float * (1.0 + self.weight.astype(mx.float32))
        return hidden_float.astype(input_dtype)


class Krea2SwiGLU(_MlxModuleBase):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        _require_mlx()
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, hidden_states: Any) -> Any:
        gate = self.gate(hidden_states)
        up = self.up(hidden_states)
        return self.down(_silu(gate.astype(mx.float32)).astype(gate.dtype) * up)


class Krea2Attention(_MlxModuleBase):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        eps: float,
    ) -> None:
        _require_mlx()
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(hidden_size, hidden_size, bias=False)]

    def __call__(
        self,
        hidden_states: Any,
        *,
        attention_mask: Any | None = None,
        image_rotary_emb: tuple[Any, Any] | None = None,
    ) -> Any:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states).reshape(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim,
        )
        key = self.to_k(hidden_states).reshape(
            batch_size,
            -1,
            self.num_kv_heads,
            self.head_dim,
        )
        value = self.to_v(hidden_states).reshape(
            batch_size,
            -1,
            self.num_kv_heads,
            self.head_dim,
        )
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size,
            -1,
            self.hidden_size,
        )
        hidden_states = hidden_states * mx.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2TextFusionBlock(_MlxModuleBase):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def __call__(self, hidden_states: Any, attention_mask: Any | None = None) -> Any:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
        )
        return hidden_states + self.ff(self.norm2(hidden_states))


class Krea2TextFusion(_MlxModuleBase):
    def __init__(
        self,
        *,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.layerwise_blocks = [
            Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
            for _ in range(num_layerwise_blocks)
        ]
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = [
            Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
            for _ in range(num_refiner_blocks)
        ]

    def __call__(self, encoder_hidden_states: Any, attention_mask: Any | None = None) -> Any:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape
        hidden_states = encoder_hidden_states.reshape(
            batch_size * seq_len,
            num_text_layers,
            dim,
        )
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            seq_len,
            num_text_layers,
            dim,
        ).transpose(0, 1, 3, 2)
        hidden_states = mx.squeeze(self.projector(hidden_states), axis=-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


class Krea2TransformerBlock(_MlxModuleBase):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.scale_shift_table = mx.zeros((6, hidden_size), dtype=mx.float32)
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: Any,
        temb: Any,
        image_rotary_emb: tuple[Any, Any],
        attention_mask: Any | None = None,
    ) -> Any:
        modulation = temb.reshape(temb.shape[0], 1, 6, -1) + self.scale_shift_table
        prescale = modulation[:, :, 0, :]
        preshift = modulation[:, :, 1, :]
        pregate = modulation[:, :, 2, :]
        postscale = modulation[:, :, 3, :]
        postshift = modulation[:, :, 4, :]
        postgate = modulation[:, :, 5, :]

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        return hidden_states + postgate * ff_out


class Krea2TimestepEmbedding(_MlxModuleBase):
    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        _require_mlx()
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, timestep: Any, *, dtype: Any) -> Any:
        half = self.embed_dim // 2
        freqs = mx.exp(
            -math.log(10_000.0)
            * mx.arange(half, dtype=mx.float32)
            / half
        )
        args = (timestep.astype(mx.float32) * 1000.0)[:, None, None] * freqs
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)
        return self.linear_2(_gelu_tanh(self.linear_1(emb)))


class Krea2TextProjection(_MlxModuleBase):
    def __init__(self, text_dim: int, hidden_size: int, *, eps: float) -> None:
        _require_mlx()
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(_gelu_tanh(hidden_states))


class Krea2FinalLayer(_MlxModuleBase):
    def __init__(self, hidden_size: int, out_channels: int, *, eps: float) -> None:
        _require_mlx()
        super().__init__()
        self.scale_shift_table = mx.zeros((2, hidden_size), dtype=mx.float32)
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, hidden_states: Any, temb: Any) -> Any:
        modulation = temb + self.scale_shift_table
        scale = modulation[:, 0:1, :]
        shift = modulation[:, 1:2, :]
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2Transformer2DModel(_MlxModuleBase):
    def __init__(self, config: Krea2TransformerConfig | None = None) -> None:
        if nn is not None:
            super().__init__()
        self.config = config or Krea2TransformerConfig()
        self.config.validate()
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.in_channels
        self.hidden_size = self.config.hidden_size
        if nn is None:
            return

        self.img_in = nn.Linear(self.config.in_channels, self.hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(
            self.config.timestep_embed_dim,
            self.hidden_size,
        )
        self.time_mod_proj = nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=self.config.num_text_layers,
            dim=self.config.text_hidden_dim,
            num_heads=self.config.text_num_attention_heads,
            num_kv_heads=self.config.text_num_key_value_heads,
            intermediate_size=self.config.text_intermediate_size,
            num_layerwise_blocks=self.config.num_layerwise_text_blocks,
            num_refiner_blocks=self.config.num_refiner_text_blocks,
            eps=self.config.norm_eps,
        )
        self.txt_in = Krea2TextProjection(
            self.config.text_hidden_dim,
            self.hidden_size,
            eps=self.config.norm_eps,
        )
        self.transformer_blocks = [
            Krea2TransformerBlock(
                hidden_size=self.hidden_size,
                intermediate_size=self.config.intermediate_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                norm_eps=self.config.norm_eps,
            )
            for _ in range(self.config.num_layers)
        ]
        self.final_layer = Krea2FinalLayer(
            self.hidden_size,
            out_channels=self.config.in_channels,
            eps=self.config.norm_eps,
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        dtype: Any | None = None,
    ) -> "Krea2Transformer2DModel":
        return load_transformer(path, dtype=dtype)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        timestep: Any,
        position_ids: Any,
        encoder_attention_mask: Any | None = None,
        *,
        return_dict: bool = True,
    ) -> Krea2TransformerOutput | tuple[Any]:
        _require_mlx()
        self._validate_forward_inputs(
            hidden_states,
            encoder_hidden_states,
            timestep,
            position_ids,
            encoder_attention_mask,
        )
        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(_gelu_tanh(temb))
        text_attention_mask, attention_mask = build_attention_masks(
            encoder_attention_mask,
            image_seq_len=image_seq_len,
        )

        encoder_hidden_states = self.text_fusion(
            encoder_hidden_states,
            attention_mask=text_attention_mask,
        )
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        image_rotary_emb = build_krea2_rotary_embeddings(
            position_ids,
            axes_dims_rope=self.config.axes_dims_rope,
            theta=self.config.rope_theta,
            dtype=mx.float32,
        )

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                temb_mod,
                image_rotary_emb,
                attention_mask=attention_mask,
            )

        hidden_states = hidden_states[:, text_seq_len:]
        output = self.final_layer(hidden_states, temb)
        if not return_dict:
            return (output,)
        return Krea2TransformerOutput(sample=output)

    def _validate_forward_inputs(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        timestep: Any,
        position_ids: Any,
        encoder_attention_mask: Any | None,
    ) -> None:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.in_channels:
            raise ValueError("hidden_states must be shaped [B, image_seq_len, in_channels]")
        if encoder_hidden_states.ndim != 4:
            raise ValueError(
                "encoder_hidden_states must be shaped [B, text_seq_len, num_text_layers, text_hidden_dim]"
            )
        if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
            raise ValueError("encoder_hidden_states batch size must match hidden_states")
        if encoder_hidden_states.shape[2] != self.config.num_text_layers:
            raise ValueError("encoder_hidden_states num_text_layers does not match config")
        if encoder_hidden_states.shape[3] != self.config.text_hidden_dim:
            raise ValueError("encoder_hidden_states text_hidden_dim does not match config")
        if timestep.ndim != 1 or timestep.shape[0] != hidden_states.shape[0]:
            raise ValueError("timestep must be shaped [B]")
        expected_seq_len = encoder_hidden_states.shape[1] + hidden_states.shape[1]
        if position_ids.ndim != 2 or position_ids.shape != (expected_seq_len, 3):
            raise ValueError("position_ids must be shaped [text_seq_len + image_seq_len, 3]")
        if encoder_attention_mask is not None and (
            encoder_attention_mask.ndim != 2
            or encoder_attention_mask.shape
            != (hidden_states.shape[0], encoder_hidden_states.shape[1])
        ):
            raise ValueError("encoder_attention_mask must be shaped [B, text_seq_len]")


def load_transformer(
    path: str | Path,
    *,
    dtype: Any | None = None,
) -> Krea2Transformer2DModel:
    _require_mlx()
    root = Path(path).expanduser()
    transformer_dir = _resolve_transformer_dir(root)
    config = Krea2TransformerConfig.from_model_config(
        read_json_object(transformer_dir / "config.json")
    )
    model = Krea2Transformer2DModel(config)
    expected_shapes = flatten_parameter_shapes(model.parameters())
    weights = load_transformer_weights(
        transformer_dir,
        expected_shapes=expected_shapes,
        dtype=dtype,
    )
    model.load_weights(sorted(weights.items()), strict=True)
    mx.eval(*weights.values())
    return model


def load_transformer_weights(
    transformer_dir: str | Path,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
    dtype: Any | None = None,
) -> dict[str, Any]:
    _require_mlx()
    return load_mapped_safetensors_weights(
        transformer_dir,
        expected_shapes=expected_shapes,
        map_key=map_transformer_weight_key,
        index_filename="diffusion_pytorch_model.safetensors.index.json",
        single_filename="diffusion_pytorch_model.safetensors",
        label="Transformer",
        dtype=dtype,
    )


def map_transformer_weight_key(key: str) -> str:
    if not key:
        raise Krea2TurboMlxError("Unexpected transformer tensor with an empty key")
    return key


def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int) -> Any:
    _require_mlx()
    if text_seq_len < 0:
        raise ValueError("text_seq_len must be non-negative")
    _require_positive(grid_height, "grid_height")
    _require_positive(grid_width, "grid_width")
    text_ids = mx.zeros((text_seq_len, 3), dtype=mx.int32)
    row_ids = mx.broadcast_to(
        mx.arange(grid_height, dtype=mx.int32)[:, None],
        (grid_height, grid_width),
    )
    col_ids = mx.broadcast_to(
        mx.arange(grid_width, dtype=mx.int32)[None, :],
        (grid_height, grid_width),
    )
    image_ids = mx.stack(
        [
            mx.zeros((grid_height, grid_width), dtype=mx.int32),
            row_ids,
            col_ids,
        ],
        axis=-1,
    ).reshape(grid_height * grid_width, 3)
    return mx.concatenate([text_ids, image_ids], axis=0)


def build_krea2_rotary_embeddings(
    position_ids: Any,
    *,
    axes_dims_rope: Iterable[int] = (32, 48, 48),
    theta: float = 1000.0,
    dtype: Any | None = None,
) -> tuple[Any, Any]:
    _require_mlx()
    axes_dims = _tuple_ints3(axes_dims_rope, "axes_dims_rope")
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError("position_ids must be shaped [sequence_length, 3]")
    cos_out = []
    sin_out = []
    positions = position_ids.astype(mx.float32)
    for axis, axis_dim in enumerate(axes_dims):
        if axis_dim <= 0 or axis_dim % 2:
            raise ValueError("axes_dims_rope entries must be positive even integers")
        inv_freq = 1.0 / (
            theta
            ** (mx.arange(0, axis_dim, 2, dtype=mx.float32) / axis_dim)
        )
        freqs = positions[:, axis : axis + 1] * inv_freq[None, :]
        cos_out.append(mx.repeat(mx.cos(freqs), 2, axis=1))
        sin_out.append(mx.repeat(mx.sin(freqs), 2, axis=1))
    cos = mx.concatenate(cos_out, axis=-1)
    sin = mx.concatenate(sin_out, axis=-1)
    if dtype is not None:
        cos = cos.astype(dtype)
        sin = sin.astype(dtype)
    return cos, sin


def apply_rotary_emb(
    hidden_states: Any,
    rotary_embeddings: tuple[Any, Any],
    *,
    sequence_dim: int = 1,
) -> Any:
    _require_mlx()
    cos, sin = rotary_embeddings
    if sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        raise ValueError("sequence_dim must be 1 or 2")

    rotated = _rotate_adjacent_pairs(hidden_states)
    output = hidden_states.astype(mx.float32) * cos.astype(mx.float32)
    output = output + rotated.astype(mx.float32) * sin.astype(mx.float32)
    return output.astype(hidden_states.dtype)


def build_attention_masks(
    encoder_attention_mask: Any | None,
    *,
    image_seq_len: int,
) -> tuple[Any | None, Any | None]:
    _require_mlx()
    _require_positive(image_seq_len, "image_seq_len")
    if encoder_attention_mask is None:
        return None, None
    if encoder_attention_mask.ndim != 2:
        raise ValueError("encoder_attention_mask must be shaped [B, text_seq_len]")

    mask = encoder_attention_mask.astype(mx.bool_)
    if bool(mx.all(mask).item()):
        return None, None
    batch_size = mask.shape[0]
    text_mask = mask[:, None, None, :]
    image_mask = mx.ones((batch_size, image_seq_len), dtype=mx.bool_)
    combined_mask = mx.concatenate([mask, image_mask], axis=1)[:, None, None, :]
    return text_mask, combined_mask


def _resolve_transformer_dir(root: Path) -> Path:
    if (root / "transformer").is_dir():
        return root / "transformer"
    if (root / "config.json").is_file():
        return root
    raise Krea2TurboMlxError(
        f"Transformer path must be an artifact/source root or transformer directory: {root}"
    )


def _rotate_adjacent_pairs(hidden_states: Any) -> Any:
    paired = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2)
    real = paired[..., 0]
    imag = paired[..., 1]
    return mx.stack([-imag, real], axis=-1).reshape(hidden_states.shape)


def _gelu_tanh(x: Any) -> Any:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def _silu(x: Any) -> Any:
    return x * mx.sigmoid(x)


def _tuple_ints3(value: Iterable[Any], name: str) -> tuple[int, int, int]:
    items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three integers")
    return items


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise Krea2TurboMlxError(
            "Krea2 transformer requires MLX. Install `krea-2-turbo-mlx[runtime]` "
            "on an MLX-supported machine."
        )


__all__ = [
    "Krea2Attention",
    "Krea2FinalLayer",
    "Krea2RMSNorm",
    "Krea2SwiGLU",
    "Krea2TextFusion",
    "Krea2TextFusionBlock",
    "Krea2TextProjection",
    "Krea2TimestepEmbedding",
    "Krea2Transformer2DModel",
    "Krea2TransformerBlock",
    "Krea2TransformerConfig",
    "Krea2TransformerOutput",
    "apply_rotary_emb",
    "build_attention_masks",
    "build_krea2_rotary_embeddings",
    "load_transformer",
    "load_transformer_weights",
    "map_transformer_weight_key",
    "prepare_position_ids",
]
