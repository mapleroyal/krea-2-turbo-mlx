from __future__ import annotations

from operator import index
from typing import Any

from .constants import MAX_GENERATION_SIZE, OUTPUT_ALIGNMENT
from .errors import ValidationError


def validate_generation_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer")
    try:
        parsed = index(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    if parsed > MAX_GENERATION_SIZE:
        raise ValidationError(f"{name} must be {MAX_GENERATION_SIZE} or smaller")
    if parsed % OUTPUT_ALIGNMENT != 0:
        raise ValidationError(f"{name} must be a multiple of {OUTPUT_ALIGNMENT}")
    return int(parsed)


def validate_generation_dimensions(width: Any, height: Any) -> tuple[int, int]:
    return (
        validate_generation_dimension(width, "width"),
        validate_generation_dimension(height, "height"),
    )


__all__ = ["validate_generation_dimension", "validate_generation_dimensions"]
