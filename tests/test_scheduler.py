from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.scheduler import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchSchedulerConfig,
    load_scheduler,
)


def test_scheduler_config_accepts_pinned_krea_turbo_values() -> None:
    config = FlowMatchSchedulerConfig.from_scheduler_config(_pinned_config())

    assert config.num_train_timesteps == 1000
    assert config.use_dynamic_shifting is True
    assert config.time_shift_type == "exponential"
    assert config.max_image_seq_len == 6400


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"_class_name": "Other"}, "FlowMatchEulerDiscreteScheduler"),
        ({"num_train_timesteps": 999}, "num_train_timesteps"),
        ({"shift": 1.1}, "shift=1.0"),
        ({"use_dynamic_shifting": False}, "dynamic shifting"),
        ({"time_shift_type": "linear"}, "exponential time shifting"),
        ({"invert_sigmas": True}, "inverted sigmas"),
        ({"shift_terminal": 0.1}, "shift_terminal"),
        ({"stochastic_sampling": True}, "stochastic"),
        ({"use_karras_sigmas": True}, "Karras"),
        ({"use_exponential_sigmas": True}, "exponential sigmas"),
        ({"use_beta_sigmas": True}, "beta sigmas"),
        ({"base_image_seq_len": 128}, "base_image_seq_len"),
        ({"max_image_seq_len": 4096}, "max_image_seq_len"),
    ],
)
def test_scheduler_config_rejects_unsupported_variants(
    override: dict[str, object],
    match: str,
) -> None:
    payload = _pinned_config()
    payload.update(override)

    with pytest.raises(ValueError, match=match):
        FlowMatchSchedulerConfig.from_scheduler_config(payload)


def test_set_timesteps_matches_shifted_diffusers_schedule() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        FlowMatchSchedulerConfig.from_scheduler_config(_pinned_config())
    )

    timesteps = scheduler.set_timesteps(4, mu=1.15)

    base = [1.0, 0.75, 0.5, 0.25]
    expected_sigmas = [
        math.exp(1.15) / (math.exp(1.15) + (1.0 / sigma - 1.0))
        for sigma in base
    ]
    assert scheduler.sigmas == pytest.approx([*expected_sigmas, 0.0])
    assert timesteps == pytest.approx([sigma * 1000.0 for sigma in expected_sigmas])
    assert scheduler.timesteps == pytest.approx(timesteps)


def test_set_timesteps_handles_single_step_with_terminal_sigma() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler()

    timesteps = scheduler.set_timesteps(1, mu=1.15)

    assert scheduler.sigmas == pytest.approx([1.0, 0.0])
    assert timesteps == pytest.approx([1000.0])


def test_euler_step_uses_next_minus_current_sigma() -> None:
    np = pytest.importorskip("numpy")
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.sigmas = [1.0, 0.25, 0.0]
    latents = np.array([[2.0, 4.0]], dtype=np.float16)
    noise = np.array([[0.5, -1.0]], dtype=np.float32)

    actual = scheduler.step(noise, 0, latents)

    expected = latents.astype(np.float32) + (0.25 - 1.0) * noise
    assert actual.dtype == noise.dtype
    np.testing.assert_allclose(actual, expected, atol=0)


def test_euler_step_requires_configured_schedule() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler()

    with pytest.raises(Krea2TurboMlxError, match="timesteps"):
        scheduler.step(1.0, 0, 1.0)


def test_load_scheduler_reads_artifact_config(tmp_path: Path) -> None:
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(_pinned_config()),
        encoding="utf-8",
    )

    scheduler = load_scheduler(tmp_path)

    assert scheduler.config == FlowMatchSchedulerConfig.from_scheduler_config(
        _pinned_config()
    )


def _pinned_config() -> dict[str, object]:
    return {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.39.0.dev0",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 6400,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
