from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest

from krea_2_turbo_mlx.constants import (
    DEFAULT_TEXT_MAX_SEQUENCE_LENGTH,
    KREA_TEXT_PREFIX_DROP_INDEX,
    KREA_TEXT_PROMPT_PREFIX,
    KREA_TEXT_PROMPT_SUFFIX,
    KREA_TEXT_SUFFIX_TOKEN_COUNT,
    TEXT_ENCODER_SELECT_LAYERS,
)
from krea_2_turbo_mlx.text_conditioning import encode_prompt
from krea_2_turbo_mlx.text_encoder import load_text_encoder

ORACLE_SOURCE_ENV = "KREA2_TURBO_MLX_TEXT_ORACLE_SOURCE"
ORACLE_PROMPTS = [
    "a fox in the snow",
    "",
    "macro photograph of translucent glass fruit on a steel table",
]


pytestmark = pytest.mark.skipif(
    not os.environ.get(ORACLE_SOURCE_ENV),
    reason=f"set {ORACLE_SOURCE_ENV} to run the Krea text-conditioning oracle",
)


def test_text_conditioning_matches_hf_reference_oracle() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    mx = pytest.importorskip("mlx.core")

    source = Path(os.environ[ORACLE_SOURCE_ENV]).expanduser()
    if not (source / "text_encoder" / "model.safetensors").is_file():
        pytest.skip("oracle source does not contain text_encoder/model.safetensors")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source / "tokenizer",
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    ref_ids, ref_mask, ref_positions = _reference_tokens(torch, tokenizer)

    actual_fp32 = encode_prompt(
        ORACLE_PROMPTS,
        tokenizer=tokenizer,
        encoder=load_text_encoder(source, dtype=mx.float32),
    )
    assert np.array_equal(np.array(actual_fp32.input_ids), ref_ids.numpy())
    assert np.array_equal(
        np.array(actual_fp32.attention_mask),
        ref_mask[:, KREA_TEXT_PREFIX_DROP_INDEX:].numpy(),
    )
    assert np.array_equal(np.array(actual_fp32.position_ids), ref_positions.numpy())

    ref_fp32 = _reference_hidden_states(
        torch,
        transformers,
        source,
        ref_ids,
        ref_mask,
        ref_positions,
        torch.float32,
    )
    actual_fp32_array = np.array(actual_fp32.hidden_states, dtype=np.float32)
    _assert_metrics(
        "fp32",
        actual_fp32_array,
        ref_fp32,
        mean_abs=0.01,
        p99_abs=0.05,
        cosine=0.9999,
    )
    del actual_fp32, actual_fp32_array
    gc.collect()

    ref_bf16 = _reference_hidden_states(
        torch,
        transformers,
        source,
        ref_ids,
        ref_mask,
        ref_positions,
        torch.bfloat16,
    )
    actual_bf16 = encode_prompt(
        ORACLE_PROMPTS,
        tokenizer=tokenizer,
        encoder=load_text_encoder(source),
    )
    actual_bf16_array = np.array(actual_bf16.hidden_states.astype(mx.float32), dtype=np.float32)
    floor = _metrics(ref_bf16, ref_fp32)
    bf16 = _metrics(actual_bf16_array, ref_fp32)
    bf16_message = json.dumps(
        {
            "actual_vs_hf_fp32": bf16,
            "actual_vs_hf_fp32_layers": [
                _metrics(actual_bf16_array[:, :, layer_index, :], ref_fp32[:, :, layer_index, :])
                for layer_index in range(actual_bf16_array.shape[2])
            ],
            "hf_bf16_floor": floor,
            "hf_bf16_floor_layers": [
                _metrics(ref_bf16[:, :, layer_index, :], ref_fp32[:, :, layer_index, :])
                for layer_index in range(ref_bf16.shape[2])
            ],
        },
        indent=2,
        sort_keys=True,
    )
    assert bf16["mean_abs"] <= 3 * floor["mean_abs"], bf16_message
    assert bf16["p99_abs"] <= 3 * floor["p99_abs"], bf16_message
    assert bf16["mean_abs"] <= 0.04, bf16_message
    assert bf16["p99_abs"] <= 0.35, bf16_message
    assert bf16["cosine"] >= 0.999, bf16_message


def _reference_tokens(torch: object, tokenizer: object) -> tuple[object, object, object]:
    prompt_max_length = (
        DEFAULT_TEXT_MAX_SEQUENCE_LENGTH
        + KREA_TEXT_PREFIX_DROP_INDEX
        - KREA_TEXT_SUFFIX_TOKEN_COUNT
    )
    text_tokens = tokenizer(
        [KREA_TEXT_PROMPT_PREFIX + prompt for prompt in ORACLE_PROMPTS],
        truncation=True,
        padding="max_length",
        max_length=prompt_max_length,
        return_tensors="pt",
    )
    suffix_tokens = tokenizer(
        [KREA_TEXT_PROMPT_SUFFIX] * len(ORACLE_PROMPTS),
        return_tensors="pt",
    )
    input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    attention_mask = torch.cat(
        [text_tokens.attention_mask, suffix_tokens.attention_mask],
        dim=1,
    ).bool()
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    return input_ids, attention_mask, position_ids


def _reference_hidden_states(
    torch: object,
    transformers: object,
    source: Path,
    input_ids: object,
    attention_mask: object,
    position_ids: object,
    dtype: object,
) -> object:
    np = pytest.importorskip("numpy")
    model = transformers.Qwen3VLModel.from_pretrained(
        source / "text_encoder",
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
    hidden = torch.stack(
        [outputs.hidden_states[layer] for layer in TEXT_ENCODER_SELECT_LAYERS],
        dim=2,
    )
    hidden = hidden[:, KREA_TEXT_PREFIX_DROP_INDEX:].float().cpu().numpy()
    del model, outputs
    gc.collect()
    return np.asarray(hidden, dtype=np.float32)


def _assert_metrics(
    label: str,
    actual: object,
    expected: object,
    *,
    mean_abs: float,
    p99_abs: float,
    cosine: float,
) -> None:
    payload = {
        "overall": _metrics(actual, expected),
        "layers": [
            _metrics(actual[:, :, layer_index, :], expected[:, :, layer_index, :])
            for layer_index in range(actual.shape[2])
        ],
    }
    message = json.dumps(payload, indent=2, sort_keys=True)
    assert payload["overall"]["mean_abs"] <= mean_abs, f"{label}\n{message}"
    assert payload["overall"]["p99_abs"] <= p99_abs, f"{label}\n{message}"
    assert payload["overall"]["cosine"] >= cosine, f"{label}\n{message}"


def _metrics(actual: object, expected: object) -> dict[str, float]:
    np = pytest.importorskip("numpy")
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    diff = np.abs(actual_array - expected_array)
    actual_flat = actual_array.reshape(-1)
    expected_flat = expected_array.reshape(-1)
    denom = np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)
    cosine = float(np.dot(actual_flat, expected_flat) / denom)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "cosine": cosine,
    }
