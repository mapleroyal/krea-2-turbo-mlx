from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_TEXT_MAX_SEQUENCE_LENGTH,
    KREA_TEXT_PREFIX_DROP_INDEX,
    KREA_TEXT_PROMPT_PREFIX,
    KREA_TEXT_PROMPT_SUFFIX,
    KREA_TEXT_SUFFIX_TOKEN_COUNT,
    TEXT_ENCODER_SELECT_LAYERS,
)
from .errors import Krea2TurboMlxError
from .text_encoder import build_text_position_ids, load_text_encoder

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None


@dataclass(frozen=True)
class PromptTruncation:
    prompt_index: int
    token_count: int
    max_length: int
    truncated_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_index": self.prompt_index,
            "token_count": self.token_count,
            "max_length": self.max_length,
            "truncated_tokens": self.truncated_tokens,
        }


@dataclass(frozen=True)
class PromptTokenLayout:
    input_ids: list[list[int]]
    attention_mask: list[list[bool]]
    position_ids: list[list[list[int]]]
    prompt_token_max_length: int
    truncation_warnings: tuple[PromptTruncation, ...]


@dataclass(frozen=True)
class TextConditioning:
    hidden_states: Any
    attention_mask: Any
    input_ids: Any
    position_ids: Any
    truncation_warnings: tuple[PromptTruncation, ...]


@dataclass(frozen=True)
class KreaTextConditioner:
    tokenizer: Any
    encoder: Any

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        dtype: Any | None = None,
    ) -> "KreaTextConditioner":
        root = Path(path).expanduser()
        return cls(
            tokenizer=_load_tokenizer(root / "tokenizer"),
            encoder=load_text_encoder(root, dtype=dtype),
        )

    def encode(
        self,
        prompts: str | Sequence[str],
        *,
        max_sequence_length: int = DEFAULT_TEXT_MAX_SEQUENCE_LENGTH,
        num_images_per_prompt: int = 1,
    ) -> TextConditioning:
        return encode_prompt(
            prompts,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
        )


def load_text_conditioner(
    path: str | Path,
    *,
    dtype: Any | None = None,
) -> KreaTextConditioner:
    return KreaTextConditioner.from_artifact(path, dtype=dtype)


def encode_prompt(
    prompts: str | Sequence[str],
    *,
    tokenizer: Any,
    encoder: Any,
    max_sequence_length: int = DEFAULT_TEXT_MAX_SEQUENCE_LENGTH,
    num_images_per_prompt: int = 1,
    select_layers: Sequence[int] = TEXT_ENCODER_SELECT_LAYERS,
) -> TextConditioning:
    _require_mlx()
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")
    if not select_layers:
        raise ValueError("select_layers must not be empty")

    layout = build_prompt_token_layout(
        prompts,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
    )
    input_ids = mx.array(layout.input_ids, dtype=mx.int32)
    attention_mask = mx.array(layout.attention_mask, dtype=mx.bool_)
    position_ids = mx.array(layout.position_ids, dtype=mx.int32)

    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    hidden_tuple = _extract_hidden_states(outputs)
    max_layer = max(int(layer) for layer in select_layers)
    if max_layer >= len(hidden_tuple):
        raise Krea2TurboMlxError(
            f"Text encoder returned {len(hidden_tuple)} hidden-state tensors; "
            f"layer {max_layer} was requested"
        )

    hidden_states = mx.stack([hidden_tuple[int(layer)] for layer in select_layers], axis=2)
    hidden_states = hidden_states[:, KREA_TEXT_PREFIX_DROP_INDEX:]
    output_mask = attention_mask[:, KREA_TEXT_PREFIX_DROP_INDEX:]

    if num_images_per_prompt > 1:
        hidden_states = mx.repeat(hidden_states, num_images_per_prompt, axis=0)
        output_mask = mx.repeat(output_mask, num_images_per_prompt, axis=0)
        input_ids = mx.repeat(input_ids, num_images_per_prompt, axis=0)
        position_ids = mx.repeat(position_ids, num_images_per_prompt, axis=1)

    mx.eval(hidden_states, output_mask, input_ids, position_ids)
    return TextConditioning(
        hidden_states=hidden_states,
        attention_mask=output_mask,
        input_ids=input_ids,
        position_ids=position_ids,
        truncation_warnings=layout.truncation_warnings,
    )


def build_prompt_token_layout(
    prompts: str | Sequence[str],
    *,
    tokenizer: Any,
    max_sequence_length: int = DEFAULT_TEXT_MAX_SEQUENCE_LENGTH,
) -> PromptTokenLayout:
    prompt_list = _normalize_prompts(prompts)
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    prompt_token_max_length = (
        max_sequence_length
        + KREA_TEXT_PREFIX_DROP_INDEX
        - KREA_TEXT_SUFFIX_TOKEN_COUNT
    )
    if prompt_token_max_length <= 0:
        raise ValueError("max_sequence_length is too small for the Krea text template")

    prompt_texts = [KREA_TEXT_PROMPT_PREFIX + prompt for prompt in prompt_list]
    truncation_warnings = _detect_truncation(
        prompt_texts,
        tokenizer=tokenizer,
        prompt_token_max_length=prompt_token_max_length,
    )

    text_tokens = tokenizer(
        prompt_texts,
        truncation=True,
        padding="max_length",
        max_length=prompt_token_max_length,
        return_tensors=None,
    )
    suffix_tokens = tokenizer(
        [KREA_TEXT_PROMPT_SUFFIX] * len(prompt_list),
        return_tensors=None,
    )
    text_input_ids = _rows(text_tokens, "input_ids")
    text_attention_mask = _rows(text_tokens, "attention_mask")
    suffix_input_ids = _rows(suffix_tokens, "input_ids")
    suffix_attention_mask = _rows(suffix_tokens, "attention_mask")

    suffix_lengths = {len(row) for row in suffix_input_ids}
    if suffix_lengths != {KREA_TEXT_SUFFIX_TOKEN_COUNT}:
        raise Krea2TurboMlxError(
            "Krea text suffix tokenization must produce exactly "
            f"{KREA_TEXT_SUFFIX_TOKEN_COUNT} tokens; got {sorted(suffix_lengths)}"
        )

    input_ids = [
        [int(item) for item in text_row + suffix_row]
        for text_row, suffix_row in zip(text_input_ids, suffix_input_ids, strict=True)
    ]
    attention_mask = [
        [bool(item) for item in text_row + suffix_row]
        for text_row, suffix_row in zip(
            text_attention_mask,
            suffix_attention_mask,
            strict=True,
        )
    ]
    position_rows = [_position_row(mask_row) for mask_row in attention_mask]
    return PromptTokenLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=[position_rows, position_rows, position_rows],
        prompt_token_max_length=prompt_token_max_length,
        truncation_warnings=truncation_warnings,
    )


def _detect_truncation(
    prompt_texts: list[str],
    *,
    tokenizer: Any,
    prompt_token_max_length: int,
) -> tuple[PromptTruncation, ...]:
    tokens = tokenizer(
        prompt_texts,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    rows = _rows(tokens, "input_ids")
    warnings: list[PromptTruncation] = []
    for index, row in enumerate(rows):
        token_count = len(row)
        if token_count > prompt_token_max_length:
            warnings.append(
                PromptTruncation(
                    prompt_index=index,
                    token_count=token_count,
                    max_length=prompt_token_max_length,
                    truncated_tokens=token_count - prompt_token_max_length,
                )
            )
    return tuple(warnings)


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise Krea2TurboMlxError(
            "Text conditioning tokenization requires transformers. Install "
            "`krea-2-turbo-mlx[runtime]`."
        ) from exc
    if not path.is_dir():
        raise Krea2TurboMlxError(f"Missing tokenizer directory: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise Krea2TurboMlxError("Krea tokenizer must define a pad token")
    return tokenizer


def _extract_hidden_states(outputs: Any) -> tuple[Any, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        raise Krea2TurboMlxError("Text encoder did not return hidden states")
    return tuple(hidden_states)


def _rows(payload: Any, key: str) -> list[list[int]]:
    try:
        value = payload[key]
    except (KeyError, TypeError) as exc:
        raise Krea2TurboMlxError(f"Tokenizer output is missing {key!r}") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = [list(row) for row in value]
    if not rows:
        raise Krea2TurboMlxError(f"Tokenizer output {key!r} must contain a batch")
    return rows


def _position_row(mask_row: Sequence[bool]) -> list[int]:
    running = 0
    positions: list[int] = []
    for item in mask_row:
        if item:
            running += 1
        positions.append(max(running - 1, 0))
    return positions


def _normalize_prompts(prompts: str | Sequence[str]) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    if not isinstance(prompts, Sequence):
        raise ValueError("prompts must be a string or sequence of strings")
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("prompt sequence must not be empty")
    if not all(isinstance(prompt, str) for prompt in prompt_list):
        raise ValueError("prompt sequence entries must be strings")
    return prompt_list


def _require_mlx() -> None:
    if mx is None:
        raise Krea2TurboMlxError(
            "Text conditioning requires MLX. Install `krea-2-turbo-mlx[runtime]` "
            "on an MLX-supported machine."
        )


__all__ = [
    "KreaTextConditioner",
    "PromptTruncation",
    "PromptTokenLayout",
    "TextConditioning",
    "build_prompt_token_layout",
    "build_text_position_ids",
    "encode_prompt",
    "load_text_conditioner",
]
