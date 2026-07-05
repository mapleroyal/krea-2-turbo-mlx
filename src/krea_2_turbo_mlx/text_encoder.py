from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import TEXT_ENCODER_WEIGHT_BODY_PREFIX
from .errors import Krea2TurboMlxError
from .json_io import read_json_object
from ._weights import (
    flatten_parameter_shapes,
    flatten_parameters,
    load_mapped_safetensors_weights,
)

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None
    nn = None

_MlxModuleBase = object if nn is None else nn.Module


@dataclass(frozen=True)
class Qwen3VLTextModelOutput:
    last_hidden_state: Any
    hidden_states: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class Qwen3VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_type: str = "default"
    mrope_interleaved: bool = True
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    pad_token_id: int | None = None

    @classmethod
    def from_model_config(cls, payload: Mapping[str, Any]) -> "Qwen3VLTextConfig":
        if payload.get("model_type") != "qwen3_vl":
            raise ValueError("Qwen3-VL text encoder requires model_type='qwen3_vl'")
        text_config = payload.get("text_config")
        if not isinstance(text_config, Mapping):
            raise ValueError("Qwen3-VL config must contain a text_config mapping")
        return cls.from_text_config(text_config)

    @classmethod
    def from_text_config(cls, payload: Mapping[str, Any]) -> "Qwen3VLTextConfig":
        rope_payload = payload.get("rope_scaling") or payload.get("rope_parameters") or {}
        if not isinstance(rope_payload, Mapping):
            raise ValueError("Qwen3-VL rope settings must be a mapping")

        config = cls(
            vocab_size=int(payload.get("vocab_size", cls.vocab_size)),
            hidden_size=int(payload.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(payload.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(payload.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=int(payload.get("num_attention_heads", cls.num_attention_heads)),
            num_key_value_heads=int(payload.get("num_key_value_heads", cls.num_key_value_heads)),
            head_dim=int(payload.get("head_dim", cls.head_dim)),
            hidden_act=str(payload.get("hidden_act", cls.hidden_act)),
            rms_norm_eps=float(payload.get("rms_norm_eps", cls.rms_norm_eps)),
            rope_theta=float(
                rope_payload.get("rope_theta", payload.get("rope_theta", cls.rope_theta))
            ),
            max_position_embeddings=int(
                payload.get("max_position_embeddings", cls.max_position_embeddings)
            ),
            attention_bias=bool(payload.get("attention_bias", cls.attention_bias)),
            attention_dropout=float(payload.get("attention_dropout", cls.attention_dropout)),
            rope_type=str(rope_payload.get("rope_type", cls.rope_type)),
            mrope_interleaved=bool(
                rope_payload.get("mrope_interleaved", cls.mrope_interleaved)
            ),
            mrope_section=_tuple_ints3(
                rope_payload.get("mrope_section", cls.mrope_section),
                "mrope_section",
            ),
            pad_token_id=(
                None if payload.get("pad_token_id") is None else int(payload["pad_token_id"])
            ),
        )
        if payload.get("model_type") not in {None, "qwen3_vl_text"}:
            raise ValueError("Qwen3-VL text encoder supports only qwen3_vl_text")
        config.validate()
        return config

    @property
    def attention_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if self.hidden_act != "silu":
            raise ValueError("Qwen3-VL text encoder supports only hidden_act='silu'")
        if self.attention_bias:
            raise ValueError("Qwen3-VL text encoder supports only attention_bias=false")
        if self.attention_dropout != 0.0:
            raise ValueError("Qwen3-VL text encoder supports only attention_dropout=0")
        if self.rope_type != "default":
            raise ValueError("Qwen3-VL text encoder supports only default RoPE")
        if not self.mrope_interleaved:
            raise ValueError("Qwen3-VL text encoder requires interleaved M-RoPE")
        if any(section <= 0 for section in self.mrope_section):
            raise ValueError("mrope_section entries must be positive")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("sum(mrope_section) must equal head_dim // 2")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")


class Qwen3VLTextRMSNorm(_MlxModuleBase):
    def __init__(self, hidden_size: int, *, eps: float) -> None:
        _require_mlx()
        super().__init__()
        self.weight = mx.ones((hidden_size,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, hidden_states: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_float), axis=-1, keepdims=True)
        hidden_float = hidden_float * mx.rsqrt(variance + self.eps)
        return self.weight.astype(input_dtype) * hidden_float.astype(input_dtype)


class Qwen3VLTextMLP(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        _require_mlx()
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: Any) -> Any:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(_silu(gate.astype(mx.float32)).astype(gate.dtype) * up)


class Qwen3VLTextAttention(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        _require_mlx()
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.attention_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.attention_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3VLTextRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
    ) -> Any:
        batch_size = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(batch_size, -1, self.heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(batch_size, -1, self.kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(batch_size, -1, self.kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary_pos_emb(query, key, position_embeddings)

        query = query.transpose(0, 2, 1, 3)
        key = _repeat_kv(key.transpose(0, 2, 1, 3), self.kv_groups)
        value = _repeat_kv(value.transpose(0, 2, 1, 3), self.kv_groups)

        hidden_states = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size,
            -1,
            self.heads * self.head_dim,
        )
        return self.o_proj(hidden_states)


class Qwen3VLTextDecoderLayer(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        _require_mlx()
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
    ) -> Any:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3VLTextModel(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig | None = None) -> None:
        if nn is not None:
            super().__init__()
        self.config = config or Qwen3VLTextConfig()
        self.config.validate()
        if nn is None:
            return

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = [
            Qwen3VLTextDecoderLayer(self.config)
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = Qwen3VLTextRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        dtype: Any | None = None,
    ) -> "Qwen3VLTextModel":
        return load_text_encoder(path, dtype=dtype)

    def __call__(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> Qwen3VLTextModelOutput:
        _require_mlx()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shaped [B, T]")
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must be shaped like input_ids")

        position_ids = _normalize_position_ids(position_ids, batch_size, seq_len)
        hidden_states = self.embed_tokens(input_ids)
        captured: list[Any] | None = [hidden_states] if output_hidden_states else None
        position_embeddings = build_text_rotary_embeddings(
            self.config,
            position_ids,
            dtype=hidden_states.dtype,
        )
        mask = causal_padding_attention_mask(attention_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, mask)
            if captured is not None:
                captured.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if captured is not None:
            captured.append(hidden_states)
        return Qwen3VLTextModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None if captured is None else tuple(captured),
        )


def load_text_encoder(
    path: str | Path,
    *,
    dtype: Any | None = None,
) -> Qwen3VLTextModel:
    _require_mlx()
    root = Path(path).expanduser()
    text_dir = _resolve_text_encoder_dir(root)
    config = Qwen3VLTextConfig.from_model_config(read_json_object(text_dir / "config.json"))
    model = Qwen3VLTextModel(config)
    expected_shapes = flatten_parameter_shapes(model.parameters())
    weights = load_text_encoder_weights(
        text_dir,
        expected_shapes=expected_shapes,
        dtype=dtype,
    )
    model.load_weights(sorted(weights.items()), strict=True)
    mx.eval(*weights.values())
    return model


def load_text_encoder_weights(
    text_encoder_dir: str | Path,
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
    dtype: Any | None = None,
) -> dict[str, Any]:
    _require_mlx()
    return load_mapped_safetensors_weights(
        text_encoder_dir,
        expected_shapes=expected_shapes,
        map_key=map_text_encoder_weight_key,
        index_filename="model.safetensors.index.json",
        single_filename="model.safetensors",
        label="Text encoder",
        dtype=dtype,
    )


def map_text_encoder_weight_key(key: str) -> str | None:
    if _is_ignored_text_encoder_weight(key):
        return None
    if key.startswith(TEXT_ENCODER_WEIGHT_BODY_PREFIX):
        return key.removeprefix(TEXT_ENCODER_WEIGHT_BODY_PREFIX)
    raise Krea2TurboMlxError(f"Unexpected text encoder tensor {key!r}")


def build_text_position_ids(attention_mask: Any) -> Any:
    _require_mlx()
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be shaped [B, T]")
    positions = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
    positions = mx.maximum(positions, mx.zeros_like(positions))
    return mx.stack([positions, positions, positions], axis=0)


def build_text_rotary_embeddings(
    config: Qwen3VLTextConfig,
    position_ids: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    _require_mlx()
    if position_ids.ndim == 2:
        position_ids = mx.stack([position_ids, position_ids, position_ids], axis=0)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("position_ids must be shaped [3, B, T] or [B, T]")

    inv_freq = 1.0 / (
        config.rope_theta
        ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
    )
    freqs = position_ids[:, :, :, None].astype(mx.float32) * inv_freq[None, None, None, :]
    freqs = _apply_interleaved_mrope(freqs, config.mrope_section)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def causal_padding_attention_mask(attention_mask: Any) -> Any:
    _require_mlx()
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be shaped [B, T]")
    seq_len = attention_mask.shape[1]
    query_positions = mx.arange(seq_len)[:, None]
    key_positions = mx.arange(seq_len)[None, :]
    causal = query_positions >= key_positions
    key_padding = attention_mask.astype(mx.bool_)[:, None, None, :]
    return causal[None, None, :, :] & key_padding


def _resolve_text_encoder_dir(root: Path) -> Path:
    if (root / "text_encoder").is_dir():
        return root / "text_encoder"
    if (root / "config.json").is_file():
        return root
    raise Krea2TurboMlxError(
        f"Text encoder path must be an artifact/source root or text_encoder directory: {root}"
    )


def _normalize_position_ids(position_ids: Any | None, batch_size: int, seq_len: int) -> Any:
    if position_ids is None:
        base = mx.arange(seq_len, dtype=mx.int32)
        positions = mx.broadcast_to(base[None, :], (batch_size, seq_len))
        return mx.stack([positions, positions, positions], axis=0)
    if position_ids.ndim == 2:
        position_ids = mx.stack([position_ids, position_ids, position_ids], axis=0)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        position_ids = position_ids[1:]
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("position_ids must be shaped [3, B, T], [4, B, T], or [B, T]")
    if position_ids.shape[1] != batch_size or position_ids.shape[2] != seq_len:
        raise ValueError("position_ids must match input_ids batch and sequence dimensions")
    return position_ids.astype(mx.int32)


def _apply_interleaved_mrope(freqs: Any, mrope_section: tuple[int, int, int]) -> Any:
    half_dim = freqs.shape[-1]
    replacements = []
    for index in range(half_dim):
        axis = _mrope_axis_for_frequency_index(index, mrope_section)
        replacements.append(freqs[axis, ..., index : index + 1])
    return mx.concatenate(replacements, axis=-1)


def _mrope_axis_for_frequency_index(
    index: int,
    mrope_section: tuple[int, int, int],
) -> int:
    for axis in (1, 2):
        length = mrope_section[axis] * 3
        if index < length and index % 3 == axis:
            return axis
    return 0


def _apply_rotary_pos_emb(
    query: Any,
    key: Any,
    position_embeddings: tuple[Any, Any],
) -> tuple[Any, Any]:
    cos, sin = position_embeddings
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    query_embed = (query * cos) + (_rotate_half(query) * sin)
    key_embed = (key * cos) + (_rotate_half(key) * sin)
    return query_embed, key_embed


def _rotate_half(x: Any) -> Any:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _repeat_kv(hidden_states: Any, repeats: int) -> Any:
    if repeats == 1:
        return hidden_states
    return mx.repeat(hidden_states, repeats, axis=1)


def _silu(x: Any) -> Any:
    return x * mx.sigmoid(x)


def _is_ignored_text_encoder_weight(key: str) -> bool:
    return (
        key.startswith("visual.")
        or key == "lm_head"
        or key.startswith("lm_head.")
        or key == "language_model.lm_head"
        or key.startswith("language_model.lm_head.")
    )


def _tuple_ints3(value: Iterable[Any], name: str) -> tuple[int, int, int]:
    items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three integers")
    return items


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise Krea2TurboMlxError(
            "Text conditioning requires MLX. Install `krea-2-turbo-mlx[runtime]` "
            "on an MLX-supported machine."
        )


__all__ = [
    "Qwen3VLTextConfig",
    "Qwen3VLTextModel",
    "Qwen3VLTextModelOutput",
    "build_text_position_ids",
    "build_text_rotary_embeddings",
    "causal_padding_attention_mask",
    "flatten_parameter_shapes",
    "flatten_parameters",
    "load_text_encoder",
    "load_text_encoder_weights",
    "map_text_encoder_weight_key",
]
