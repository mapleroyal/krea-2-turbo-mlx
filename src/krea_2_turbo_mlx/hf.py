from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_SOURCE_DIR,
    DIFFUSERS_COMPONENTS,
    EXPECTED_SOURCE_METADATA_PATHS,
    OFFICIAL_HF_REPO_ID,
    OFFICIAL_HF_REVISION,
    PROJECT_NAME,
)
from .errors import Krea2TurboMlxError

DOWNLOAD_ALLOW_PATTERNS = (
    "model_index.json",
    "README*",
    "LICENSE*",
    "*.pdf",
    ".gitattributes",
    *(f"{component}/**" for component in DIFFUSERS_COMPONENTS),
)
DOWNLOAD_IGNORE_PATTERNS = ("turbo.safetensors",)


def resolve_source_revision(source: str | Path, revision: str | None = None) -> str | None:
    source_text = str(source)
    if Path(source_text).expanduser().exists():
        return revision
    if revision:
        return revision
    if source_text == OFFICIAL_HF_REPO_ID:
        return OFFICIAL_HF_REVISION
    raise Krea2TurboMlxError(
        f"Remote source {source_text!r} requires --revision so runs are reproducible."
    )


def build_download_plan(
    source: str | Path = OFFICIAL_HF_REPO_ID,
    *,
    revision: str | None = None,
    dest: str | Path | None = None,
) -> dict[str, Any]:
    source_text = str(source)
    source_path = Path(source_text).expanduser()
    if source_path.exists():
        return {
            "schema_version": 1,
            "status": "local_source_already_available",
            "action": "download",
            "source": str(source_path),
            "source_kind": "local_directory" if source_path.is_dir() else "local_file",
            "revision": revision,
            "downloads_model_weights": False,
            "notes": ["The download command does not copy local sources."],
        }

    effective_revision = resolve_source_revision(source_text, revision)
    destination = Path(dest).expanduser() if dest is not None else DEFAULT_SOURCE_DIR
    if destination.name != _repo_leaf(source_text):
        destination = destination / _repo_leaf(source_text)
    return {
        "schema_version": 1,
        "status": "download_planned",
        "action": "download",
        "source": source_text,
        "source_kind": "huggingface_model",
        "revision": effective_revision,
        "destination": str(destination),
        "downloads_model_weights": True,
        "allow_patterns": list(DOWNLOAD_ALLOW_PATTERNS),
        "ignore_patterns": list(DOWNLOAD_IGNORE_PATTERNS),
        "notes": [
            "Downloads the Diffusers component layout and source metadata.",
            "Does not download the duplicate root turbo.safetensors file.",
        ],
    }


def download_source(
    source: str | Path = OFFICIAL_HF_REPO_ID,
    *,
    revision: str | None = None,
    dest_root: str | Path = DEFAULT_SOURCE_DIR.parent,
    local_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Download a Hugging Face source snapshot, or return an existing local path."""

    source_path = Path(str(source)).expanduser()
    if source_path.exists():
        return source_path

    repo_id = str(source)
    effective_revision = resolve_source_revision(repo_id, revision)
    destination = (
        Path(local_dir).expanduser()
        if local_dir is not None
        else Path(dest_root).expanduser() / _repo_leaf(repo_id)
    )
    if _looks_like_source_dir(destination):
        _emit(progress_callback, f"reusing source at {destination}")
        return destination

    snapshot_download = _load_snapshot_download()
    _emit(
        progress_callback,
        f"downloading {repo_id}@{effective_revision} into {destination}",
    )
    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            revision=effective_revision,
            local_dir=destination,
            allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
            ignore_patterns=list(DOWNLOAD_IGNORE_PATTERNS),
            library_name=PROJECT_NAME,
        )
    except Exception as exc:  # pragma: no cover - exercised through the real HF client.
        raise Krea2TurboMlxError(
            f"Failed to download Hugging Face source {repo_id!r}: {exc}"
        ) from exc
    return Path(downloaded).expanduser()


def _load_hf_tools() -> tuple[type[Any], Callable[..., str]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise Krea2TurboMlxError(
            "Hugging Face metadata inspection requires huggingface-hub. "
            "Install the project dependency set before using remote manifests."
        ) from exc
    return HfApi, hf_hub_download


def _load_snapshot_download() -> Callable[..., Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise Krea2TurboMlxError(
            "Downloading Hugging Face sources requires huggingface-hub. "
            "Run `./setup.sh` or install `krea-2-turbo-mlx[runtime]`."
        ) from exc
    return snapshot_download


def _repo_leaf(repo_id: str) -> str:
    return repo_id.rstrip("/").rsplit("/", 1)[-1]


def _looks_like_source_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / rel_path).is_file() for rel_path in EXPECTED_SOURCE_METADATA_PATHS)


def _emit(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
