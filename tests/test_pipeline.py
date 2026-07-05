from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from krea_2_turbo_mlx.constants import ARTIFACT_FORMAT, EXPECTED_COMPONENT_CLASSES
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.pipeline import (
    KreaTurboPipeline,
    PipelinePreviewFrame,
    latent_preview_rgb,
    pack_latents,
    unpack_latents,
)
from krea_2_turbo_mlx.scheduler import FlowMatchEulerDiscreteScheduler
from krea_2_turbo_mlx.tensor_selection import SELECTION_POLICY_VERSION


def test_pack_latents_matches_diffusers_patch_layout_for_non_square_grid() -> None:
    np = pytest.importorskip("numpy")
    latents = np.arange(1 * 2 * 4 * 6, dtype=np.float32).reshape(1, 2, 4, 6)

    packed = pack_latents(latents, patch_size=2)

    expected = (
        latents.reshape(1, 2, 2, 2, 3, 2)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(1, 6, 8)
    )
    np.testing.assert_array_equal(packed, expected)


def test_unpack_latents_reverses_pack_layout_and_adds_single_frame_axis() -> None:
    np = pytest.importorskip("numpy")
    original = np.arange(1 * 2 * 4 * 6, dtype=np.float32).reshape(1, 2, 4, 6)
    packed = pack_latents(original, patch_size=2)

    unpacked = unpack_latents(
        packed,
        height=32,
        width=48,
        vae_scale_factor=8,
        patch_size=2,
    )

    np.testing.assert_array_equal(unpacked, original[:, :, None, :, :])


def test_unpack_latents_rejects_grid_size_mismatch() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="sequence length"):
        unpack_latents(
            np.zeros((1, 3, 4), dtype=np.float32),
            height=16,
            width=16,
        )


def test_from_artifact_requires_converted_artifact_metadata(tmp_path: Path) -> None:
    _write_model_index(tmp_path)

    with pytest.raises(Krea2TurboMlxError, match="Missing converted artifact metadata"):
        KreaTurboPipeline.from_artifact(tmp_path)


def test_from_artifact_accepts_full_precision_artifact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import krea_2_turbo_mlx.pipeline as pipeline_module

    _write_artifact_metadata(tmp_path)
    _write_model_index(tmp_path)
    scheduler = object()
    text_conditioner = object()
    transformer = object()
    vae = _FakeLoadVAE()

    monkeypatch.setattr(pipeline_module, "load_scheduler", lambda root: scheduler)
    monkeypatch.setattr(
        pipeline_module,
        "load_text_conditioner",
        lambda root, *, dtype=None: text_conditioner,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_transformer",
        lambda root, *, dtype=None: transformer,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_vae",
        lambda root, *, dtype=None: vae,
    )

    pipe = KreaTurboPipeline.from_artifact(tmp_path)

    assert pipe.scheduler is scheduler
    assert pipe.text_conditioner is text_conditioner
    assert pipe.transformer is transformer
    assert pipe.vae is vae
    assert vae.tiling_enabled is True


def test_pipeline_runs_tiny_injected_component_path_with_normalized_timesteps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    fake_mx = _FakeMx(np)
    monkeypatch.setattr(pipeline_module, "mx", fake_mx)
    monkeypatch.setattr(pipeline_module, "prepare_position_ids", _numpy_position_ids)
    monkeypatch.setattr(
        pipeline_module,
        "postprocess_decoded_image",
        lambda decoded, *, output_type: ["image"],
    )

    transformer = _FakeTransformer(np)
    vae = _FakeVAE(np)
    pipe = KreaTurboPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_conditioner=_FakeTextConditioner(np),
        transformer=transformer,
        vae=vae,
        patch_size=2,
        vae_scale_factor=8,
        is_distilled=True,
    )

    result = pipe(
        "a glass observatory",
        width=32,
        height=16,
        steps=2,
        seed=123,
    )

    assert result.images == ["image"]
    assert result.seed == 123
    assert result.latents is None
    assert fake_mx.clear_cache_calls == 1
    assert fake_mx.random.seed_value == 123
    assert transformer.timesteps == pytest.approx(pipe.scheduler.sigmas[:2])
    np.testing.assert_array_equal(
        transformer.position_ids,
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        ),
    )
    assert transformer.hidden_shapes == [(1, 2, 4), (1, 2, 4)]
    assert vae.latents.shape == (1, 1, 1, 2, 4)


def test_pipeline_preview_callback_fires_on_completed_interval_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    fake_mx = _FakeMx(np)
    monkeypatch.setattr(pipeline_module, "mx", fake_mx)
    monkeypatch.setattr(pipeline_module, "prepare_position_ids", _numpy_position_ids)
    monkeypatch.setattr(
        pipeline_module,
        "postprocess_decoded_image",
        lambda decoded, *, output_type: ["image"],
    )
    frames: list[PipelinePreviewFrame] = []

    pipe = KreaTurboPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_conditioner=_FakeTextConditioner(np),
        transformer=_FakeTransformer(np, in_channels=64),
        vae=_FakeVAE(np),
    )

    pipe(
        "a glass observatory",
        width=32,
        height=32,
        steps=4,
        seed=123,
        live_preview="latent",
        preview_interval_steps=2,
        preview_callback=frames.append,
    )

    assert [frame.step_index for frame in frames] == [1]
    assert [frame.step_count for frame in frames] == [4]
    assert all(frame.mode == "latent" for frame in frames)
    assert all(frame.width == 32 and frame.height == 32 for frame in frames)
    assert all(frame.image.shape == (4, 4, 3) for frame in frames)


def test_pipeline_latent_preview_keeps_final_step_when_output_is_latent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    fake_mx = _FakeMx(np)
    monkeypatch.setattr(pipeline_module, "mx", fake_mx)
    monkeypatch.setattr(pipeline_module, "prepare_position_ids", _numpy_position_ids)
    frames: list[PipelinePreviewFrame] = []

    pipe = KreaTurboPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_conditioner=_FakeTextConditioner(np),
        transformer=_FakeTransformer(np, in_channels=64),
        vae=_FakeVAE(np),
    )

    pipe(
        "a glass observatory",
        width=32,
        height=32,
        steps=2,
        seed=123,
        output_type="latent",
        live_preview="latent",
        preview_interval_steps=1,
        preview_callback=frames.append,
    )

    assert [frame.step_index for frame in frames] == [0, 1]


def test_pipeline_vae_preview_skips_final_step_when_final_decode_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "mx", _FakeMx(np))
    monkeypatch.setattr(pipeline_module, "prepare_position_ids", _numpy_position_ids)
    monkeypatch.setattr(
        pipeline_module,
        "postprocess_decoded_image",
        lambda decoded, *, output_type: np.transpose(
            np.clip(decoded[:, :, 0] / 2.0 + 0.5, 0.0, 1.0),
            (0, 2, 3, 1),
        ),
    )

    frames: list[PipelinePreviewFrame] = []
    vae = _FakeVAE(np)
    pipe = KreaTurboPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_conditioner=_FakeTextConditioner(np),
        transformer=_FakeTransformer(np, in_channels=64),
        vae=vae,
    )

    pipe(
        "a glass observatory",
        width=32,
        height=32,
        steps=2,
        seed=123,
        live_preview="vae",
        preview_interval_steps=1,
        preview_callback=frames.append,
    )

    assert [frame.step_index for frame in frames] == [0]
    assert frames[0].image.shape == (16, 32, 3)
    assert vae.decode_count == 2


def test_latent_preview_rgb_handles_packed_krea_latents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "mx", _FakeMx(np))
    packed = pack_latents(np.zeros((1, 16, 4, 4), dtype=np.float32))

    preview = latent_preview_rgb(packed, height=32, width=32)

    assert preview.shape == (4, 4, 3)
    assert preview.dtype == np.float32
    np.testing.assert_allclose(
        preview[0, 0],
        np.array([0.40825, 0.4566, 0.332], dtype=np.float32),
        atol=1e-6,
    )


def test_pipeline_private_latent_hook_validates_packed_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "mx", _FakeMx(np))
    pipe = KreaTurboPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_conditioner=_FakeTextConditioner(np),
        transformer=_FakeTransformer(np),
        vae=_FakeVAE(np),
    )

    with pytest.raises(ValueError, match="packed latents"):
        pipe.prepare_latents(
            batch_size=1,
            height=16,
            width=16,
            dtype=np.float32,
            latents=np.zeros((1, 2, 4), dtype=np.float32),
        )


def test_pipeline_rejects_non_turbo_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    import krea_2_turbo_mlx.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "mx", _FakeMx(np))
    pipe = KreaTurboPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_conditioner=_FakeTextConditioner(np),
        transformer=_FakeTransformer(np),
        vae=_FakeVAE(np),
    )

    with pytest.raises(ValueError, match="guidance_scale=0.0"):
        pipe("prompt", guidance_scale=1.0)


class _FakeRandom:
    def __init__(self, np: object) -> None:
        self.np = np
        self.seed_value: int | None = None
        self.rng = np.random.default_rng(0)

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)
        self.rng = self.np.random.default_rng(seed)

    def normal(self, shape: tuple[int, ...], dtype: object | None = None) -> object:
        return self.rng.standard_normal(shape).astype(dtype or self.np.float32)


class _FakeMx:
    def __init__(self, np: object) -> None:
        self.np = np
        self.float32 = np.float32
        self.int32 = np.int32
        self.bool_ = np.bool_
        self.random = _FakeRandom(np)
        self.clear_cache_calls = 0

    def array(self, value: object, dtype: object | None = None) -> object:
        return self.np.array(value, dtype=dtype)

    def eval(self, *values: object) -> None:
        return None

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1


@dataclass
class _FakeTextConditioner:
    np: object

    def encode(self, prompt: str) -> object:
        return SimpleNamespace(
            hidden_states=self.np.ones((1, 3, 2, 4), dtype=self.np.float32),
            attention_mask=self.np.array([[True, False, True]], dtype=self.np.bool_),
            truncation_warnings=(),
        )


@dataclass
class _FakeTransformer:
    np: object
    in_channels: int = 4

    def __post_init__(self) -> None:
        self.config = SimpleNamespace(in_channels=self.in_channels)
        self.timesteps: list[float] = []
        self.hidden_shapes: list[tuple[int, ...]] = []
        self.position_ids = None

    def __call__(
        self,
        *,
        hidden_states: object,
        encoder_hidden_states: object,
        timestep: object,
        position_ids: object,
        encoder_attention_mask: object,
        return_dict: bool,
    ) -> tuple[object]:
        del encoder_hidden_states, encoder_attention_mask, return_dict
        self.timesteps.append(float(timestep[0]))
        self.hidden_shapes.append(tuple(hidden_states.shape))
        self.position_ids = self.np.array(position_ids)
        return (self.np.ones_like(hidden_states, dtype=self.np.float32),)


@dataclass
class _FakeVAE:
    np: object

    def __post_init__(self) -> None:
        self.spatial_compression_ratio = 8
        self.post_quant_conv = SimpleNamespace(
            weight=self.np.zeros((1,), dtype=self.np.float32)
        )
        self.latents = None
        self.decode_count = 0

    def decode(self, latents: object, *, return_dict: bool) -> tuple[object]:
        del return_dict
        self.decode_count += 1
        self.latents = latents
        return (self.np.zeros((1, 3, 1, 16, 32), dtype=self.np.float32),)


class _FakeLoadVAE:
    spatial_compression_ratio = 8

    def __init__(self) -> None:
        self.tiling_enabled = False

    def enable_tiling(self) -> None:
        self.tiling_enabled = True


def _numpy_position_ids(text_seq_len: int, grid_height: int, grid_width: int) -> object:
    np = pytest.importorskip("numpy")
    text_ids = np.zeros((text_seq_len, 3), dtype=np.int32)
    rows = np.broadcast_to(np.arange(grid_height, dtype=np.int32)[:, None], (grid_height, grid_width))
    cols = np.broadcast_to(np.arange(grid_width, dtype=np.int32)[None, :], (grid_height, grid_width))
    image_ids = np.stack(
        [np.zeros((grid_height, grid_width), dtype=np.int32), rows, cols],
        axis=-1,
    ).reshape(grid_height * grid_width, 3)
    return np.concatenate([text_ids, image_ids], axis=0)


def _write_artifact_metadata(root: Path) -> None:
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": ARTIFACT_FORMAT,
                "full_precision_only": True,
                "selection_policy_version": SELECTION_POLICY_VERSION,
                "precision": {
                    "preserves_source_dtypes": True,
                    "dtype_equivalence_verified": True,
                    "quantized_dtypes_present": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_model_index(root: Path) -> None:
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Krea2Pipeline",
                **EXPECTED_COMPONENT_CLASSES,
                "is_distilled": True,
                "patch_size": 2,
            }
        ),
        encoding="utf-8",
    )
