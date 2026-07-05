from __future__ import annotations


class Krea2TurboMlxError(Exception):
    """Base exception for project-level failures."""


class ValidationError(Krea2TurboMlxError, ValueError):
    """Raised when project-specific validation fails."""

