from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DEFAULT_DISTILLED_SHIFT
from .errors import Krea2TurboMlxError
from .json_io import read_json_object

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised on non-MLX test runners.
    mx = None


@dataclass(frozen=True)
class FlowMatchSchedulerConfig:
    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = True
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 6400
    invert_sigmas: bool = False
    shift_terminal: float | None = None
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_beta_sigmas: bool = False
    time_shift_type: str = "exponential"
    stochastic_sampling: bool = False

    @classmethod
    def from_scheduler_config(cls, payload: Mapping[str, Any]) -> "FlowMatchSchedulerConfig":
        if payload.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
            raise ValueError(
                "Krea 2 Turbo requires _class_name='FlowMatchEulerDiscreteScheduler'"
            )
        config = cls(
            num_train_timesteps=int(
                payload.get("num_train_timesteps", cls.num_train_timesteps)
            ),
            shift=float(payload.get("shift", cls.shift)),
            use_dynamic_shifting=bool(
                payload.get("use_dynamic_shifting", cls.use_dynamic_shifting)
            ),
            base_shift=float(payload.get("base_shift", cls.base_shift)),
            max_shift=float(payload.get("max_shift", cls.max_shift)),
            base_image_seq_len=int(
                payload.get("base_image_seq_len", cls.base_image_seq_len)
            ),
            max_image_seq_len=int(
                payload.get("max_image_seq_len", cls.max_image_seq_len)
            ),
            invert_sigmas=bool(payload.get("invert_sigmas", cls.invert_sigmas)),
            shift_terminal=(
                None
                if payload.get("shift_terminal", None) is None
                else float(payload["shift_terminal"])
            ),
            use_karras_sigmas=bool(
                payload.get("use_karras_sigmas", cls.use_karras_sigmas)
            ),
            use_exponential_sigmas=bool(
                payload.get("use_exponential_sigmas", cls.use_exponential_sigmas)
            ),
            use_beta_sigmas=bool(payload.get("use_beta_sigmas", cls.use_beta_sigmas)),
            time_shift_type=str(payload.get("time_shift_type", cls.time_shift_type)),
            stochastic_sampling=bool(
                payload.get("stochastic_sampling", cls.stochastic_sampling)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_train_timesteps != 1000:
            raise ValueError("Krea 2 Turbo scheduler requires num_train_timesteps=1000")
        if self.shift != 1.0:
            raise ValueError("Krea 2 Turbo scheduler requires shift=1.0")
        if not self.use_dynamic_shifting:
            raise ValueError("Krea 2 Turbo scheduler requires dynamic shifting")
        if self.time_shift_type != "exponential":
            raise ValueError("Krea 2 Turbo scheduler requires exponential time shifting")
        if self.invert_sigmas:
            raise ValueError("Krea 2 Turbo scheduler does not support inverted sigmas")
        if self.shift_terminal is not None:
            raise ValueError("Krea 2 Turbo scheduler does not support shift_terminal")
        if self.stochastic_sampling:
            raise ValueError("Krea 2 Turbo scheduler does not support stochastic sampling")
        if self.use_karras_sigmas:
            raise ValueError("Krea 2 Turbo scheduler does not support Karras sigmas")
        if self.use_exponential_sigmas:
            raise ValueError("Krea 2 Turbo scheduler does not support exponential sigmas")
        if self.use_beta_sigmas:
            raise ValueError("Krea 2 Turbo scheduler does not support beta sigmas")
        if self.base_shift != 0.5:
            raise ValueError("Krea 2 Turbo scheduler requires base_shift=0.5")
        if self.max_shift != 1.15:
            raise ValueError("Krea 2 Turbo scheduler requires max_shift=1.15")
        if self.base_image_seq_len != 256:
            raise ValueError("Krea 2 Turbo scheduler requires base_image_seq_len=256")
        if self.max_image_seq_len != 6400:
            raise ValueError("Krea 2 Turbo scheduler requires max_image_seq_len=6400")


class FlowMatchEulerDiscreteScheduler:
    order = 1

    def __init__(self, config: FlowMatchSchedulerConfig | None = None) -> None:
        self.config = config or FlowMatchSchedulerConfig()
        self.config.validate()
        self.sigmas: list[float] = []
        self.timesteps: list[float] = []

    @classmethod
    def from_artifact(cls, path: str | Path) -> "FlowMatchEulerDiscreteScheduler":
        return load_scheduler(path)

    def set_timesteps(
        self,
        steps: int,
        *,
        mu: float = DEFAULT_DISTILLED_SHIFT,
    ) -> list[float]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if steps == 1:
            base_sigmas = [1.0]
        else:
            delta = (1.0 - (1.0 / steps)) / (steps - 1)
            base_sigmas = [1.0 - index * delta for index in range(steps)]
        shifted = [_time_shift_exponential(mu, 1.0, sigma) for sigma in base_sigmas]
        self.timesteps = [
            sigma * float(self.config.num_train_timesteps)
            for sigma in shifted
        ]
        self.sigmas = [*shifted, 0.0]
        return list(self.timesteps)

    def step(self, noise_pred: Any, step_index: int, latents: Any) -> Any:
        if not self.sigmas:
            raise Krea2TurboMlxError("Scheduler timesteps must be set before stepping")
        if step_index < 0 or step_index >= len(self.sigmas) - 1:
            raise ValueError("step_index is outside the configured sigma schedule")
        dt = self.sigmas[step_index + 1] - self.sigmas[step_index]
        latents_f32 = _astype_float32(latents)
        prev_sample = latents_f32 + dt * noise_pred
        dtype = getattr(noise_pred, "dtype", None)
        if dtype is not None and hasattr(prev_sample, "astype"):
            prev_sample = prev_sample.astype(dtype)
        return prev_sample


def load_scheduler(path: str | Path) -> FlowMatchEulerDiscreteScheduler:
    root = Path(path).expanduser()
    scheduler_dir = root / "scheduler" if (root / "scheduler").is_dir() else root
    config_path = scheduler_dir / "scheduler_config.json"
    if not config_path.is_file():
        raise Krea2TurboMlxError(f"Missing scheduler config: {config_path}")
    config = FlowMatchSchedulerConfig.from_scheduler_config(read_json_object(config_path))
    return FlowMatchEulerDiscreteScheduler(config)


def _time_shift_exponential(mu: float, sigma: float, timestep: float) -> float:
    return math.exp(mu) / (math.exp(mu) + (1.0 / timestep - 1.0) ** sigma)


def _astype_float32(value: Any) -> Any:
    if not hasattr(value, "astype"):
        return value
    if mx is not None:
        try:
            return value.astype(mx.float32)
        except TypeError:
            pass
    return value.astype("float32")


__all__ = [
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchSchedulerConfig",
    "load_scheduler",
]
