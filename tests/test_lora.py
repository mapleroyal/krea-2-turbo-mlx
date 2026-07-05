from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.lora import (
    LOCAL_LORA_DEFAULT_SCALE,
    applied_lora_patches,
    map_lora_target_key,
    normalize_lora_payload,
    resolve_lora_patches,
    scan_lora_catalog,
)
from safetensors_fixtures import write_safetensors_fixture

PROJECTOR_SHAPE = (1, 12)


def test_weight_diff_lora_file_wraps_restores_and_does_not_drift(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    projector = _Linear(np.arange(12, dtype=np.float32).reshape(1, 12))
    transformer = _transformer(projector=projector)
    lora_dir = tmp_path / "loras"
    path = lora_dir / "filter.safetensors"
    diff = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.75, -0.25, 0]],
        dtype=np.float32,
    )
    _write_arrays(path, {"diffusion_model.txtfusion.projector.diff": diff})
    x = np.ones((2, 12), dtype=np.float32)
    base_out = projector(x).copy()

    catalog = scan_lora_catalog(lora_dir).to_mapping()
    assert [item["id"] for item in catalog["items"]] == ["filter.safetensors"]
    assert catalog["items"][0]["adapter_type"] == "weight-diff"
    assert catalog["items"][0]["target_count"] == 1

    patch = resolve_lora_patches(
        ["filter.safetensors:1.5"],
        transformer=transformer,
        lora_dir=lora_dir,
    )[0]
    assert patch.scale == 1.5
    assert patch.source_type == "catalog"
    assert patch.adapter_type == "weight-diff"
    with applied_lora_patches(transformer, (patch,)):
        np.testing.assert_allclose(
            transformer.text_fusion.projector(x),
            base_out + x @ diff.T * 1.5,
        )

    assert transformer.text_fusion.projector is projector
    np.testing.assert_allclose(transformer.text_fusion.projector(x), base_out)

    repeated_patch = resolve_lora_patches(
        ["filter.safetensors:4"],
        transformer=transformer,
        lora_dir=lora_dir,
    )[0]
    for _ in range(3):
        with applied_lora_patches(transformer, (repeated_patch,)):
            np.testing.assert_allclose(
                transformer.text_fusion.projector(x),
                base_out + x @ diff.T * 4.0,
            )
        assert transformer.text_fusion.projector is projector


def test_key_mapping_supports_krea_and_prefixed_topologies() -> None:
    assert (
        map_lora_target_key("diffusion_model.blocks.2.attn.wq")
        == "transformer_blocks.2.attn.to_q"
    )
    assert (
        map_lora_target_key("diffusion_model.blocks.2.attn.wo")
        == "transformer_blocks.2.attn.to_out.0"
    )
    assert (
        map_lora_target_key("diffusion_model.blocks.2.mlp.down")
        == "transformer_blocks.2.ff.down"
    )
    assert (
        map_lora_target_key(
            "base_model.model.transformer.diffusion_model.txtfusion.refiner_blocks.1.attn.wv"
        )
        == "text_fusion.refiner_blocks.1.attn.to_v"
    )
    assert (
        map_lora_target_key("diffusion_model.txtfusion.projector")
        == "text_fusion.projector"
    )
    assert (
        map_lora_target_key("transformer.transformer_blocks.0.attn.to_k")
        == "transformer_blocks.0.attn.to_k"
    )


def test_standard_lora_matches_dense_low_rank_delta_and_catalog_resolution(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    lora_dir = tmp_path / "loras"
    path = lora_dir / "styles" / "glass.safetensors"
    down = np.array([[1, 2, 0], [0, -1, 3]], dtype=np.float32)
    up = np.array([[1, 0], [0, 2], [-1, 1], [3, -2]], dtype=np.float32)
    alpha = np.array([4], dtype=np.float32)
    _write_arrays(
        path,
        {
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": down,
            "diffusion_model.blocks.0.attn.wq.lora_up.weight": up,
            "diffusion_model.blocks.0.attn.wq.alpha": alpha,
        },
    )
    transformer = _transformer(q_weight=np.zeros((4, 3), dtype=np.float32))
    x = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)

    catalog = scan_lora_catalog(lora_dir).to_mapping()
    assert [item["id"] for item in catalog["items"]] == ["styles/glass.safetensors"]
    assert catalog["items"][0]["adapter_type"] == "standard"
    assert catalog["items"][0]["target_count"] == 1

    patch = resolve_lora_patches(
        ["styles/glass.safetensors:0.5"],
        transformer=transformer,
        lora_dir=lora_dir,
    )[0]
    assert patch.id == "styles/glass.safetensors"
    assert patch.source_type == "catalog"
    assert patch.scale == 0.5
    assert patch.metadata()["target_count"] == 1
    assert len(patch.metadata()["sha256"]) == 64

    with applied_lora_patches(transformer, (patch,)):
        dense_delta = (up @ down) * (4 / 2) * 0.5
        np.testing.assert_allclose(
            transformer.transformer_blocks[0].attn.to_q(x),
            x @ dense_delta.T,
        )


def test_catalog_resolution_allows_symlinked_loras_inside_lora_dir(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    source_dir = tmp_path / "source"
    source_path = source_dir / "filter.safetensors"
    diff = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.75, -0.25, 0]],
        dtype=np.float32,
    )
    _write_arrays(source_path, {"diffusion_model.txtfusion.projector.diff": diff})

    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    catalog_path = lora_dir / "filter.safetensors"
    try:
        catalog_path.symlink_to(source_path)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    catalog = scan_lora_catalog(lora_dir).to_mapping()
    assert [item["id"] for item in catalog["items"]] == ["filter.safetensors"]

    patch = resolve_lora_patches(
        ["filter.safetensors"],
        transformer=_transformer(),
        lora_dir=lora_dir,
    )[0]

    assert patch.source_type == "catalog"
    assert patch.adapter_type == "weight-diff"
    assert Path(str(patch.path)).is_symlink()


def test_catalog_skips_incomplete_adapter_groups_with_loader_warnings(
    tmp_path: Path,
) -> None:
    lora_dir = tmp_path / "loras"
    standard_path = lora_dir / "standard-down-only.safetensors"
    lokr_path = lora_dir / "lokr-w1-only.safetensors"
    write_safetensors_fixture(
        standard_path,
        {
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": ("F32", [1, 3]),
        },
    )
    write_safetensors_fixture(
        lokr_path,
        {
            "diffusion_model.blocks.0.attn.wk.lokr_w1": ("F32", [1, 1]),
        },
    )

    catalog = scan_lora_catalog(lora_dir).to_mapping()

    assert [item["id"] for item in catalog["items"]] == []
    assert any(
        "standard LoRA requires down and up tensors" in warning
        for warning in catalog["warnings"]
    )
    assert any(
        "LoKr requires w2 or w2_a/w2_b tensors" in warning
        for warning in catalog["warnings"]
    )
    with pytest.raises(Krea2TurboMlxError, match="standard LoRA requires down and up"):
        resolve_lora_patches(
            ["standard-down-only.safetensors"],
            transformer=_transformer(),
            lora_dir=lora_dir,
        )
    with pytest.raises(Krea2TurboMlxError, match="LoKr requires w2"):
        resolve_lora_patches(
            ["lokr-w1-only.safetensors"],
            transformer=_transformer(),
            lora_dir=lora_dir,
        )


def test_lokr_direct_ignores_alpha_and_decomposed_uses_alpha_over_rank(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    direct_path = tmp_path / "direct.safetensors"
    w1 = np.array([[1], [2]], dtype=np.float32)
    w2 = np.array([[1, 0, -1], [2, 1, 0]], dtype=np.float32)
    _write_arrays(
        direct_path,
        {
            "diffusion_model.blocks.0.attn.wk.lokr_w1": w1,
            "diffusion_model.blocks.0.attn.wk.lokr_w2": w2,
            "diffusion_model.blocks.0.attn.wk.alpha": np.array([999], dtype=np.float32),
        },
    )

    transformer = _transformer(k_weight=np.zeros((4, 3), dtype=np.float32))
    x = np.array([[[1, 2, 3]]], dtype=np.float32)
    direct = resolve_lora_patches([f"{direct_path}:2"], transformer=transformer)[0]

    with applied_lora_patches(transformer, (direct,)):
        dense_delta = np.kron(w1, w2) * 2
        np.testing.assert_allclose(
            transformer.transformer_blocks[0].attn.to_k(x),
            x @ dense_delta.T,
        )

    decomp_path = tmp_path / "decomp.safetensors"
    w1_a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    w1_b = np.array([[2], [3]], dtype=np.float32)
    w2_a = np.array([[1, -1], [2, 0]], dtype=np.float32)
    w2_b = np.array([[1, 2, 0], [0, 1, 3]], dtype=np.float32)
    _write_arrays(
        decomp_path,
        {
            "diffusion_model.blocks.0.attn.wv.lokr_w1_a": w1_a,
            "diffusion_model.blocks.0.attn.wv.lokr_w1_b": w1_b,
            "diffusion_model.blocks.0.attn.wv.lokr_w2_a": w2_a,
            "diffusion_model.blocks.0.attn.wv.lokr_w2_b": w2_b,
            "diffusion_model.blocks.0.attn.wv.alpha": np.array([4], dtype=np.float32),
        },
    )
    transformer = _transformer(v_weight=np.zeros((4, 3), dtype=np.float32))
    decomp = resolve_lora_patches([str(decomp_path)], transformer=transformer)[0]

    with applied_lora_patches(transformer, (decomp,)):
        dense_delta = np.kron(w1_a @ w1_b, w2_a @ w2_b) * (4 / 2)
        np.testing.assert_allclose(
            transformer.transformer_blocks[0].attn.to_v(x),
            x @ dense_delta.T,
        )


def test_multiple_adapters_stack_on_same_target_and_restore(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write_arrays(
        first,
        {
            "diffusion_model.blocks.0.mlp.down.lora_A.weight": np.array(
                [[1, 0]], dtype=np.float32
            ),
            "diffusion_model.blocks.0.mlp.down.lora_B.weight": np.array(
                [[2], [0], [0]], dtype=np.float32
            ),
        },
    )
    _write_arrays(
        second,
        {
            "diffusion_model.blocks.0.mlp.down.lora_A.weight": np.array(
                [[0, 1]], dtype=np.float32
            ),
            "diffusion_model.blocks.0.mlp.down.lora_B.weight": np.array(
                [[0], [3], [0]], dtype=np.float32
            ),
        },
    )
    down = _Linear(np.zeros((3, 2), dtype=np.float32))
    transformer = _transformer(down=down)
    patches = resolve_lora_patches(
        [str(first), str(second)],
        transformer=transformer,
    )
    x = np.array([[5, 7]], dtype=np.float32)

    with applied_lora_patches(transformer, patches):
        np.testing.assert_allclose(
            transformer.transformer_blocks[0].ff.down(x),
            np.array([[10, 21, 0]], dtype=np.float32),
        )

    assert transformer.transformer_blocks[0].ff.down is down
    np.testing.assert_allclose(down(x), np.zeros((1, 3), dtype=np.float32))


def test_local_lora_rejects_unsupported_variants_and_shape_mismatches(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    dora = tmp_path / "dora.safetensors"
    _write_arrays(
        dora,
        {
            "diffusion_model.blocks.0.attn.wq.lokr_w1": np.ones((1, 1), dtype=np.float32),
            "diffusion_model.blocks.0.attn.wq.lokr_w2": np.ones((1, 1), dtype=np.float32),
            "diffusion_model.blocks.0.attn.wq.dora_scale": np.ones((1,), dtype=np.float32),
        },
    )
    with pytest.raises(Krea2TurboMlxError, match="DoRA"):
        resolve_lora_patches([str(dora)], transformer=_transformer())

    wrong_shape = tmp_path / "wrong-shape.safetensors"
    _write_arrays(
        wrong_shape,
        {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": np.ones((1, 5), dtype=np.float32),
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": np.ones((4, 1), dtype=np.float32),
        },
    )
    with pytest.raises(Krea2TurboMlxError, match="shape"):
        resolve_lora_patches([str(wrong_shape)], transformer=_transformer())


def test_lora_scales_are_validated_by_source_type() -> None:
    with pytest.raises(Krea2TurboMlxError, match="from 0 to 4"):
        normalize_lora_payload([{"id": "local.safetensors", "scale": 5}])

    reference = normalize_lora_payload(
        [{"id": "SomeLocalAdapter"}],
        clamp_scale=True,
    )[0]
    assert reference.id == "SomeLocalAdapter"
    assert reference.scale == LOCAL_LORA_DEFAULT_SCALE
    assert normalize_lora_payload(
        [{"id": "style.safetensors"}],
        clamp_scale=True,
    )[0].scale == LOCAL_LORA_DEFAULT_SCALE


class _Linear:
    def __init__(self, weight: object) -> None:
        self.weight = weight
        self.dtype = getattr(weight, "dtype", None)

    def __call__(self, x: object) -> object:
        return x @ self.weight.T


def _transformer(
    *,
    projector: object | None = None,
    q_weight: object | None = None,
    k_weight: object | None = None,
    v_weight: object | None = None,
    down: object | None = None,
) -> SimpleNamespace:
    np = pytest.importorskip("numpy")
    return SimpleNamespace(
        text_fusion=SimpleNamespace(
            projector=projector
            or _Linear(np.zeros(PROJECTOR_SHAPE, dtype=np.float32)),
        ),
        transformer_blocks=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    to_q=_Linear(
                        q_weight
                        if q_weight is not None
                        else np.zeros((4, 3), dtype=np.float32)
                    ),
                    to_k=_Linear(
                        k_weight
                        if k_weight is not None
                        else np.zeros((4, 3), dtype=np.float32)
                    ),
                    to_v=_Linear(
                        v_weight
                        if v_weight is not None
                        else np.zeros((4, 3), dtype=np.float32)
                    ),
                    to_out=[_Linear(np.zeros((4, 4), dtype=np.float32))],
                    to_gate=_Linear(np.zeros((4, 4), dtype=np.float32)),
                ),
                ff=SimpleNamespace(
                    gate=_Linear(np.zeros((3, 2), dtype=np.float32)),
                    up=_Linear(np.zeros((3, 2), dtype=np.float32)),
                    down=down or _Linear(np.zeros((3, 2), dtype=np.float32)),
                ),
            )
        ],
    )


def _write_arrays(path: Path, arrays: dict[str, object]) -> None:
    tensors = {}
    payloads = {}
    for key, array in arrays.items():
        dtype = str(array.dtype)
        if dtype == "float32":
            tensor_dtype = "F32"
        elif dtype == "float16":
            tensor_dtype = "F16"
        else:
            raise AssertionError(f"unsupported fixture dtype: {array.dtype}")
        tensors[key] = (tensor_dtype, list(array.shape))
        payloads[key] = array.tobytes()
    write_safetensors_fixture(path, tensors, payloads=payloads)
