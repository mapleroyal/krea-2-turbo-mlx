from __future__ import annotations

from typing import Any, Iterable

from .errors import Krea2TurboMlxError

SCHEMA_VERSION = 2
SELECTION_POLICY_VERSION = "m1-full-precision-v1"
QUANTIZED_DTYPES = frozenset({"U8", "I8", "F8_E4M3", "F8_E5M2"})


def select_manifest_tensors(manifest: dict[str, Any]) -> dict[str, Any]:
    decisions = [
        _select_tensor(entry)
        for entry in sorted(
            manifest.get("tensor_inventory", []),
            key=lambda item: (str(item.get("path", "")), str(item.get("key", ""))),
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_version": SELECTION_POLICY_VERSION,
        "decisions": decisions,
        "summary": summarize_decisions(decisions),
        "rule_matches": _rule_matches(decisions),
    }


def summarize_decisions(decisions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = _new_bucket()
    excluded = _new_bucket()
    quantized_dtypes = set()

    for decision in decisions:
        bucket = selected if decision["keep"] else excluded
        _add_to_bucket(bucket, decision)
        dtype = str(decision.get("dtype", "unknown"))
        if dtype in QUANTIZED_DTYPES:
            quantized_dtypes.add(dtype)

    return {
        "selected": selected,
        "excluded": excluded,
        "quantized_dtypes_present": bool(quantized_dtypes),
        "quantized_dtypes": sorted(quantized_dtypes),
    }


def _select_tensor(entry: dict[str, Any]) -> dict[str, Any]:
    component = str(entry.get("component", ""))
    key = str(entry.get("key", ""))
    path = str(entry.get("path", ""))

    keep: bool
    rule: str
    reason: str
    destination_path: str | None

    if component == "transformer":
        keep = True
        rule = "transformer.keep_all_runtime_tensors"
        reason = "Transformer runtime tensors are all required for denoising."
        destination_path = path
    elif component == "text_encoder":
        keep, rule, reason = _select_text_encoder_key(key)
        destination_path = "text_encoder/model.safetensors" if keep else None
    elif component == "vae":
        keep, rule, reason = _select_vae_key(key)
        destination_path = "vae/diffusion_pytorch_model.safetensors" if keep else None
    elif component == "root" and path == "turbo.safetensors":
        keep = False
        rule = "root.drop_duplicate_turbo_package"
        reason = "Root turbo.safetensors duplicates the canonical Diffusers component layout."
        destination_path = None
    else:
        raise Krea2TurboMlxError(
            f"No M1 tensor selection rule for component {component!r}, "
            f"path {path!r}, key {key!r}."
        )

    return {
        "key": key,
        "component": component,
        "source_path": path,
        "destination_path": destination_path,
        "dtype": str(entry.get("dtype", "")),
        "shape": list(entry.get("shape", [])),
        "byte_count": int(entry.get("byte_count", 0)),
        "keep": keep,
        "decision": "keep" if keep else "drop",
        "matched_rule": rule,
        "reason": reason,
        "preserve_source_dtype": True,
    }


def _select_text_encoder_key(key: str) -> tuple[bool, str, str]:
    if (
        key == "lm_head"
        or key.startswith("lm_head.")
        or key == "language_model.lm_head"
        or key.startswith("language_model.lm_head.")
    ):
        return (
            False,
            "text_encoder.drop_lm_head",
            "The language-model output head is not used for hidden-state conditioning.",
        )
    if key.startswith("language_model."):
        return (
            True,
            "text_encoder.keep_language_model",
            "Qwen language-model tensors are required for text conditioning.",
        )
    if key.startswith("visual."):
        return (
            False,
            "text_encoder.drop_visual_branch",
            "The visual encoder branch is unreachable in text-to-image conditioning.",
        )
    raise Krea2TurboMlxError(
        f"No M1 text_encoder tensor selection rule matched key {key!r}."
    )


def _select_vae_key(key: str) -> tuple[bool, str, str]:
    if key.startswith("decoder."):
        return (
            True,
            "vae.keep_decoder",
            "VAE decoder tensors are required for latent-to-image decoding.",
        )
    if key.startswith("post_quant_conv."):
        return (
            True,
            "vae.keep_post_quant_conv",
            "The post-quantization projection is required before VAE decode.",
        )
    if key.startswith("encoder."):
        return (
            False,
            "vae.drop_encoder",
            "The encode path is outside the text-to-image runtime path for M1.",
        )
    if key.startswith("quant_conv."):
        return (
            False,
            "vae.drop_quant_conv",
            "The encode-side quantization projection is outside the M1 decode path.",
        )
    raise Krea2TurboMlxError(f"No M1 vae tensor selection rule matched key {key!r}.")


def _new_bucket() -> dict[str, Any]:
    return {
        "tensor_count": 0,
        "total_tensor_bytes": 0,
        "dtypes": {},
        "components": {},
    }


def _add_to_bucket(bucket: dict[str, Any], decision: dict[str, Any]) -> None:
    bucket["tensor_count"] += 1
    bucket["total_tensor_bytes"] += int(decision.get("byte_count", 0))
    dtype = str(decision.get("dtype", "unknown"))
    bucket["dtypes"][dtype] = bucket["dtypes"].get(dtype, 0) + 1

    component = str(decision.get("component", "other"))
    component_bucket = bucket["components"].setdefault(
        component,
        {
            "tensor_count": 0,
            "total_tensor_bytes": 0,
            "dtypes": {},
        },
    )
    component_bucket["tensor_count"] += 1
    component_bucket["total_tensor_bytes"] += int(decision.get("byte_count", 0))
    component_bucket["dtypes"][dtype] = component_bucket["dtypes"].get(dtype, 0) + 1


def _rule_matches(decisions: Iterable[dict[str, Any]]) -> dict[str, int]:
    matches: dict[str, int] = {
        "text_encoder.drop_lm_head": 0,
    }
    for decision in decisions:
        rule = str(decision.get("matched_rule", "unknown"))
        matches[rule] = matches.get(rule, 0) + 1
    return matches
