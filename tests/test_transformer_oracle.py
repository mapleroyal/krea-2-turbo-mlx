from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest

from krea_2_turbo_mlx.transformer import (
    build_attention_masks,
    build_krea2_rotary_embeddings,
    load_transformer,
    prepare_position_ids,
)

ORACLE_SOURCE_ENV = "KREA2_TURBO_MLX_TRANSFORMER_ORACLE_SOURCE"


pytestmark = pytest.mark.skipif(
    not os.environ.get(ORACLE_SOURCE_ENV),
    reason=f"set {ORACLE_SOURCE_ENV} to run the Krea transformer oracle",
)


def test_transformer_forward_matches_diffusers_reference_oracle() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    mx = pytest.importorskip("mlx.core")

    source = Path(os.environ[ORACLE_SOURCE_ENV]).expanduser()
    transformer_dir = source / "transformer"
    if not _has_transformer_weights(transformer_dir):
        pytest.skip("oracle source does not contain transformer weights")

    inputs = _deterministic_inputs(np, mx)

    ref_fp32, ref_fp32_probes = _reference_outputs(
        np,
        torch,
        diffusers,
        source,
        inputs,
        torch.float32,
        collect_probes=True,
    )
    _collect(torch)

    actual_fp32, actual_fp32_probes = _mlx_outputs(
        np,
        mx,
        source,
        inputs,
        dtype=mx.float32,
        input_dtype=mx.float32,
        collect_probes=True,
    )
    fp32_payload = _assert_metrics(
        "fp32",
        actual_fp32,
        ref_fp32,
        mean_abs=0.01,
        p99_abs=0.05,
        cosine=0.9999,
        actual_probes=actual_fp32_probes,
        expected_probes=ref_fp32_probes,
    )
    del actual_fp32, actual_fp32_probes
    _collect(torch)

    ref_bf16, _ = _reference_outputs(
        np,
        torch,
        diffusers,
        source,
        inputs,
        torch.bfloat16,
        collect_probes=False,
    )
    _collect(torch)

    actual_native, _ = _mlx_outputs(
        np,
        mx,
        source,
        inputs,
        dtype=None,
        input_dtype=mx.bfloat16,
        collect_probes=False,
    )
    floor = _metrics(ref_bf16, ref_fp32)
    native = _metrics(actual_native, ref_fp32)
    message = json.dumps(
        {
            "actual_native_vs_hf_fp32": native,
            "hf_bf16_floor": floor,
        },
        indent=2,
        sort_keys=True,
    )
    assert native["mean_abs"] <= 3 * floor["mean_abs"], message
    assert native["p99_abs"] <= 3 * floor["p99_abs"], message
    assert native["mean_abs"] <= 0.05, message
    assert native["p99_abs"] <= 0.50, message
    assert native["cosine"] >= 0.999, message
    print(
        "TRANSFORMER_ORACLE_METRICS "
        + json.dumps(
            {
                "fp32": fp32_payload["overall"],
                "actual_native_vs_hf_fp32": native,
                "hf_bf16_floor": floor,
            },
            sort_keys=True,
        )
    )


def _reference_outputs(
    np: object,
    torch: object,
    diffusers: object,
    source: Path,
    inputs: dict[str, object],
    dtype: object,
    *,
    collect_probes: bool,
) -> tuple[object, dict[str, object]]:
    import torch.nn.functional as F

    model = diffusers.Krea2Transformer2DModel.from_pretrained(
        source / "transformer",
        torch_dtype=dtype,
    )
    model.eval()
    hidden_states = torch.from_numpy(inputs["hidden_states"]).to(dtype=dtype)
    encoder_hidden_states = torch.from_numpy(inputs["encoder_hidden_states"]).to(dtype=dtype)
    timestep = torch.from_numpy(inputs["timestep"]).to(dtype=dtype)
    position_ids = torch.from_numpy(inputs["position_ids"])
    encoder_attention_mask = torch.from_numpy(inputs["encoder_attention_mask"]).bool()
    probes: dict[str, object] = {}

    with torch.inference_mode():
        text_seq_len = encoder_hidden_states.shape[1]
        image_seq_len = hidden_states.shape[1]
        temb = model.time_embed(timestep, dtype=hidden_states.dtype)
        if collect_probes:
            probes["timestep_embedding"] = temb.float().cpu().numpy()
        temb_mod = model.time_mod_proj(F.gelu(temb, approximate="tanh"))

        text_attention_mask = encoder_attention_mask[:, None, None, :]
        image_mask = encoder_attention_mask.new_ones((encoder_attention_mask.shape[0], image_seq_len))
        attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

        if collect_probes:
            text_fused = _reference_text_fusion_with_probes(
                model.text_fusion,
                encoder_hidden_states,
                text_attention_mask,
                probes,
            )
        else:
            text_fused = model.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        if collect_probes:
            probes["text_fusion"] = text_fused.float().cpu().numpy()
        text_projected = model.txt_in(text_fused)
        hidden_projected = model.img_in(hidden_states)
        combined = torch.cat([text_projected, hidden_projected], dim=1)
        rotary = model.rotary_emb(position_ids)
        for index, block in enumerate(model.transformer_blocks):
            combined = block(combined, temb_mod, rotary, attention_mask)
            if collect_probes and index == 0:
                probes["first_block"] = combined.float().cpu().numpy()
        if collect_probes:
            probes["final_block"] = combined.float().cpu().numpy()
        image_hidden = combined[:, text_seq_len:]
        output = model.final_layer(image_hidden, temb)
        if collect_probes:
            probes["final_projection"] = output.float().cpu().numpy()
        output_array = output.float().cpu().numpy()

    del model
    return np.asarray(output_array, dtype=np.float32), probes


def _mlx_outputs(
    np: object,
    mx: object,
    source: Path,
    inputs: dict[str, object],
    *,
    dtype: object | None,
    input_dtype: object,
    collect_probes: bool,
) -> tuple[object, dict[str, object]]:
    model = load_transformer(source, dtype=dtype)
    hidden_states = mx.array(inputs["hidden_states"], dtype=input_dtype)
    encoder_hidden_states = mx.array(inputs["encoder_hidden_states"], dtype=input_dtype)
    timestep = mx.array(inputs["timestep"], dtype=input_dtype)
    position_ids = mx.array(inputs["position_ids"], dtype=mx.int32)
    encoder_attention_mask = mx.array(inputs["encoder_attention_mask"], dtype=mx.bool_)
    probes: dict[str, object] = {}

    text_seq_len = encoder_hidden_states.shape[1]
    image_seq_len = hidden_states.shape[1]
    temb = model.time_embed(timestep, dtype=hidden_states.dtype)
    if collect_probes:
        probes["timestep_embedding"] = temb.astype(mx.float32)
    temb_mod = model.time_mod_proj(_gelu_tanh(mx, temb))

    text_attention_mask, attention_mask = build_attention_masks(
        encoder_attention_mask,
        image_seq_len=image_seq_len,
    )
    if collect_probes:
        text_fused = _mlx_text_fusion_with_probes(
            mx,
            model.text_fusion,
            encoder_hidden_states,
            text_attention_mask,
            probes,
        )
    else:
        text_fused = model.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
    if collect_probes:
        probes["text_fusion"] = text_fused.astype(mx.float32)
    text_projected = model.txt_in(text_fused)
    hidden_projected = model.img_in(hidden_states)
    combined = mx.concatenate([text_projected, hidden_projected], axis=1)
    rotary = build_krea2_rotary_embeddings(
        position_ids,
        axes_dims_rope=model.config.axes_dims_rope,
        theta=model.config.rope_theta,
        dtype=mx.float32,
    )
    for index, block in enumerate(model.transformer_blocks):
        combined = block(combined, temb_mod, rotary, attention_mask=attention_mask)
        if collect_probes and index == 0:
            probes["first_block"] = combined.astype(mx.float32)
    if collect_probes:
        probes["final_block"] = combined.astype(mx.float32)
    image_hidden = combined[:, text_seq_len:]
    output = model.final_layer(image_hidden, temb)
    if collect_probes:
        probes["final_projection"] = output.astype(mx.float32)

    arrays = [output, *probes.values()]
    mx.eval(*arrays)
    output_array = np.array(output.astype(mx.float32), dtype=np.float32)
    probe_arrays = {
        name: np.array(value, dtype=np.float32)
        for name, value in probes.items()
    }
    del model
    return output_array, probe_arrays


def _deterministic_inputs(np: object, mx: object) -> dict[str, object]:
    rng = np.random.default_rng(13)
    hidden_states = rng.standard_normal((1, 256, 64)).astype(np.float32)
    encoder_hidden_states = rng.standard_normal((1, 512, 12, 2560)).astype(np.float32)
    timestep = np.array([0.5], dtype=np.float32)
    encoder_attention_mask = np.ones((1, 512), dtype=bool)
    encoder_attention_mask[:, 448:] = False
    position_ids = prepare_position_ids(512, 16, 16)
    mx.eval(position_ids)
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "encoder_attention_mask": encoder_attention_mask,
        "position_ids": np.array(position_ids, dtype=np.int32),
    }


def _reference_text_fusion_with_probes(
    text_fusion: object,
    encoder_hidden_states: object,
    attention_mask: object,
    probes: dict[str, object],
) -> object:
    batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape
    hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
    for index, block in enumerate(text_fusion.layerwise_blocks):
        if index == 0:
            hidden_states = _reference_text_fusion_block_with_probes(
                block,
                hidden_states.contiguous(),
                None,
                probes,
                "text_fusion_layerwise_0",
            )
        else:
            hidden_states = block(hidden_states.contiguous())
        probes[f"text_fusion_layerwise_{index}"] = hidden_states.float().cpu().numpy()

    hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2)
    hidden_states = text_fusion.projector(hidden_states).squeeze(-1)
    probes["text_fusion_projector"] = hidden_states.float().cpu().numpy()

    for index, block in enumerate(text_fusion.refiner_blocks):
        hidden_states = block(hidden_states, attention_mask=attention_mask)
        probes[f"text_fusion_refiner_{index}"] = hidden_states.float().cpu().numpy()
    return hidden_states


def _mlx_text_fusion_with_probes(
    mx: object,
    text_fusion: object,
    encoder_hidden_states: object,
    attention_mask: object,
    probes: dict[str, object],
) -> object:
    batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape
    hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
    for index, block in enumerate(text_fusion.layerwise_blocks):
        if index == 0:
            hidden_states = _mlx_text_fusion_block_with_probes(
                mx,
                block,
                hidden_states,
                None,
                probes,
                "text_fusion_layerwise_0",
            )
        else:
            hidden_states = block(hidden_states)
        probes[f"text_fusion_layerwise_{index}"] = hidden_states.astype(mx.float32)

    hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).transpose(0, 1, 3, 2)
    hidden_states = mx.squeeze(text_fusion.projector(hidden_states), axis=-1)
    probes["text_fusion_projector"] = hidden_states.astype(mx.float32)

    for index, block in enumerate(text_fusion.refiner_blocks):
        hidden_states = block(hidden_states, attention_mask=attention_mask)
        probes[f"text_fusion_refiner_{index}"] = hidden_states.astype(mx.float32)
    return hidden_states


def _reference_text_fusion_block_with_probes(
    block: object,
    hidden_states: object,
    attention_mask: object,
    probes: dict[str, object],
    prefix: str,
) -> object:
    norm1 = block.norm1(hidden_states)
    probes[f"{prefix}_norm1"] = norm1.float().cpu().numpy()
    attn_out = block.attn(norm1, attention_mask=attention_mask)
    probes[f"{prefix}_attn"] = attn_out.float().cpu().numpy()
    hidden_states = hidden_states + attn_out
    probes[f"{prefix}_post_attn"] = hidden_states.float().cpu().numpy()
    norm2 = block.norm2(hidden_states)
    probes[f"{prefix}_norm2"] = norm2.float().cpu().numpy()
    ff_out = block.ff(norm2)
    probes[f"{prefix}_ff"] = ff_out.float().cpu().numpy()
    return hidden_states + ff_out


def _mlx_text_fusion_block_with_probes(
    mx: object,
    block: object,
    hidden_states: object,
    attention_mask: object,
    probes: dict[str, object],
    prefix: str,
) -> object:
    norm1 = block.norm1(hidden_states)
    probes[f"{prefix}_norm1"] = norm1.astype(mx.float32)
    attn_out = block.attn(norm1, attention_mask=attention_mask)
    probes[f"{prefix}_attn"] = attn_out.astype(mx.float32)
    hidden_states = hidden_states + attn_out
    probes[f"{prefix}_post_attn"] = hidden_states.astype(mx.float32)
    norm2 = block.norm2(hidden_states)
    probes[f"{prefix}_norm2"] = norm2.astype(mx.float32)
    ff_out = block.ff(norm2)
    probes[f"{prefix}_ff"] = ff_out.astype(mx.float32)
    return hidden_states + ff_out


def _assert_metrics(
    label: str,
    actual: object,
    expected: object,
    *,
    mean_abs: float,
    p99_abs: float,
    cosine: float,
    actual_probes: dict[str, object] | None = None,
    expected_probes: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {"overall": _metrics(actual, expected)}
    if actual_probes and expected_probes:
        payload["probes"] = {
            key: _metrics(actual_probes[key], expected_probes[key])
            for key in sorted(actual_probes.keys() & expected_probes.keys())
        }
    message = json.dumps(payload, indent=2, sort_keys=True)
    assert payload["overall"]["mean_abs"] <= mean_abs, f"{label}\n{message}"
    assert payload["overall"]["p99_abs"] <= p99_abs, f"{label}\n{message}"
    assert payload["overall"]["cosine"] >= cosine, f"{label}\n{message}"
    return payload


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


def _gelu_tanh(mx: object, x: object) -> object:
    return 0.5 * x * (1.0 + mx.tanh((2.0 / 3.141592653589793) ** 0.5 * (x + 0.044715 * x * x * x)))


def _has_transformer_weights(transformer_dir: Path) -> bool:
    return (
        (transformer_dir / "diffusion_pytorch_model.safetensors").is_file()
        or (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").is_file()
    )


def _collect(torch: object) -> None:
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
