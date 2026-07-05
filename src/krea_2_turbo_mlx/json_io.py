from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import Krea2TurboMlxError


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Krea2TurboMlxError(f"Unable to read JSON from {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Krea2TurboMlxError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise Krea2TurboMlxError(f"JSON file must contain an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write formatted JSON via a sibling temporary file and atomic replace."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except OSError as exc:
        raise Krea2TurboMlxError(f"Unable to write JSON to {path}: {exc}") from exc
