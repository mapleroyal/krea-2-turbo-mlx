# Artifact Contract

This is the canonical human-readable contract for M1 full-precision artifacts.

## Source Scope

- Canonical source: `krea/Krea-2-Turbo`
- Audited revision: `1161245028ef398cd0a951101b2bbf486464f841`
- Canonical packaging: Diffusers component layout
- Duplicate packaging: root `turbo.safetensors` is inventoried and excluded from artifacts
- Precision posture: preserve official source tensor dtypes and bytes; do not quantize, upcast, or rename runtime keys

## Output Layout

```text
artifact.json
conversion_report.json
model_index.json
scheduler/scheduler_config.json
tokenizer/*
text_encoder/config.json
text_encoder/model.safetensors
transformer/config.json
transformer/diffusion_pytorch_model-00001-of-00003.safetensors
transformer/diffusion_pytorch_model-00002-of-00003.safetensors
transformer/diffusion_pytorch_model-00003-of-00003.safetensors
transformer/diffusion_pytorch_model.safetensors.index.json
vae/config.json
vae/diffusion_pytorch_model.safetensors
```

The exact transformer shard names follow the local source. Official artifacts use three transformer shards; smaller test or development sources may have a single transformer safetensors file plus the regenerated transformer index.

## `artifact.json`

`artifact.json` is the compact doctor-readable contract. It uses `schema_version: 2` and `format: "krea-2-turbo-mlx-artifact"`.

Required proof fields:

- `full_precision_only: true`
- `selection_policy_version`
- selected and excluded tensor counts, dtype histograms, and byte totals
- `precision.preserves_source_dtypes: true`
- `precision.dtype_equivalence_verified: true`
- `precision.quantized_dtypes_present: false`
- `provenance.source_revision`
- `provenance.selected_tensor_fingerprint`
- `provenance.selected_header_fingerprint`
- `provenance.copied_metadata_fingerprint`
- `provenance.source_layout`

`artifact.json` must not contain a top-level `quantization` key.

The provenance fingerprints are deterministic SHA-256 digests over selected tensor decisions, selected header-level tensor facts, and copied metadata file contents. They are intended for reproducibility checks and for generation metadata; they are not a replacement for the full `conversion_report.json` audit trail.

## `conversion_report.json`

`conversion_report.json` is the full audit trail. It uses `schema_version: 2` and `format: "krea-2-turbo-mlx-conversion-report"`.

Every tensor decision records:

- tensor key
- component
- source path
- destination path when kept
- dtype
- shape
- byte count
- keep/drop decision
- matched selection rule
- reason

## Tensor Selection

Selection policy version: `m1-full-precision-v1`.

- `transformer`: keep every tensor.
- `text_encoder`: keep `language_model.*`; drop `visual.*`; drop `lm_head*` if present.
- `vae`: keep `decoder.*` and `post_quant_conv.*`; drop `encoder.*` and `quant_conv.*`.
- `root/turbo.safetensors`: drop as duplicate standalone packaging.

Unknown components or tensor branches fail conversion. This is intentional: M1 excludes only verified unreachable branches.

## Safetensors Writes

- All-kept files are copied byte-for-byte.
- Selective files are rebuilt with preserved tensor bytes, dtype, shape, and `__metadata__`.
- Rebuilt headers use deterministic key order and recomputed offsets.
- Transformer indexes are regenerated to match written shards.
- Written headers are re-read and checked against source dtype, shape, and byte counts.
