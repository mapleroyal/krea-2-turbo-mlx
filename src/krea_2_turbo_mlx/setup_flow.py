from __future__ import annotations

import html
import platform
import shlex
import shutil
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qs, urlencode

from .constants import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_GENERATION_HEIGHT,
    DEFAULT_GENERATION_WIDTH,
    DEFAULT_GUI_LAUNCHER,
    DEFAULT_LORA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    OFFICIAL_HF_REPO_ID,
    OFFICIAL_HF_REVISION,
    PROJECT_NAME,
)
from .convert import run_conversion
from .doctor import format_doctor_report, run_doctor
from .errors import Krea2TurboMlxError
from .hf import download_source
from .json_io import read_json_object, write_json
from .local_server_security import (
    SESSION_TOKEN_FIELD,
    new_session_token,
    validate_local_request,
    validate_session_token,
)


@dataclass(frozen=True)
class SetupConfig:
    source: str = OFFICIAL_HF_REPO_ID
    revision: str | None = OFFICIAL_HF_REVISION
    source_dir: Path = DEFAULT_SOURCE_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    lora_dir: Path = DEFAULT_LORA_DIR
    cleanup_source: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SetupConfig":
        config = cls(
            source=_non_empty_str(payload.get("source"), OFFICIAL_HF_REPO_ID),
            revision=_optional_str(payload.get("revision")),
            source_dir=_path(payload.get("source_dir"), DEFAULT_SOURCE_DIR),
            artifact_dir=_path(payload.get("artifact_dir"), DEFAULT_ARTIFACT_PATH),
            output_dir=_path(payload.get("output_dir"), DEFAULT_OUTPUT_DIR),
            lora_dir=_path(payload.get("lora_dir"), DEFAULT_LORA_DIR),
            cleanup_source=_bool(payload.get("cleanup_source"), default=False),
        )
        return config.validate()

    @classmethod
    def from_form(cls, fields: dict[str, list[str]]) -> "SetupConfig":
        return cls.from_mapping(
            {
                "source": _field(fields, "source"),
                "revision": _field(fields, "revision"),
                "source_dir": _field(fields, "source_dir"),
                "artifact_dir": _field(fields, "artifact_dir"),
                "output_dir": _field(fields, "output_dir"),
                "lora_dir": _field(fields, "lora_dir"),
                "cleanup_source": "cleanup_source" in fields,
            }
        )

    def validate(self) -> "SetupConfig":
        if not self.source.strip():
            raise Krea2TurboMlxError("Setup source must not be empty.")
        for label, path in (
            ("source_dir", self.source_dir),
            ("artifact_dir", self.artifact_dir),
            ("output_dir", self.output_dir),
            ("lora_dir", self.lora_dir),
        ):
            if str(path).strip() in {"", "."}:
                raise Krea2TurboMlxError(f"Setup {label} must be a real path.")
        return self

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "revision": self.revision,
            "source_dir": str(self.source_dir),
            "artifact_dir": str(self.artifact_dir),
            "output_dir": str(self.output_dir),
            "lora_dir": str(self.lora_dir),
            "cleanup_source": self.cleanup_source,
        }


def run_setup_cli(args: Any, *, stream: TextIO = sys.stderr) -> int:
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path) if config_path.exists() else SetupConfig()
    config = _apply_cli_overrides(config, args)

    if not args.accept_defaults and not args.dry_run:
        config = collect_setup_config(
            config,
            open_browser=not args.no_browser,
            stream=stream,
        )

    if not args.dry_run:
        _save_config(config, config_path)
    return run_setup(config, config_path=config_path, dry_run=args.dry_run, stream=stream)


def run_setup(
    config: SetupConfig,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dry_run: bool = False,
    stream: TextIO = sys.stderr,
) -> int:
    errors = _environment_errors()
    if errors:
        raise Krea2TurboMlxError("; ".join(errors))

    config = config.validate()
    _print_setup_summary(config, config_path, stream=stream)
    if dry_run:
        _print(stream, "[dry-run] setup would download, convert, validate, and clean up as needed")
        return 0

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.lora_dir.mkdir(parents=True, exist_ok=True)
    artifact, source_to_cleanup = _ensure_artifact(config, stream=stream)
    _doctor_or_raise(artifact, stream=stream)
    if source_to_cleanup is not None and config.cleanup_source:
        _cleanup_source(source_to_cleanup, stream=stream)

    _print(stream, "")
    _print(stream, "Setup complete.")
    _print(stream, f"Try: {_format_try_command(artifact, config)}")
    launcher = _write_gui_launcher(config_path)
    _print_launcher_notice(stream, launcher)
    return 0


def collect_setup_config(
    default: SetupConfig,
    *,
    open_browser: bool,
    stream: TextIO,
) -> SetupConfig:
    done = threading.Event()
    result: dict[str, Any] = {}
    session_token = new_session_token()
    handler = _setup_handler(default, done, result, session_token=session_token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/?{urlencode({'token': session_token})}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _print(stream, f"Setup choices: {url}")
    if open_browser:
        webbrowser.open(url)
    _print(stream, "Choose setup options in the browser, then return here.")
    try:
        done.wait()
    except KeyboardInterrupt:
        raise Krea2TurboMlxError("Setup cancelled before choices were submitted.") from None
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    if "error" in result:
        raise result["error"]
    config = result.get("config")
    if not isinstance(config, SetupConfig):
        raise Krea2TurboMlxError("Setup browser closed before options were submitted.")
    _print(stream, "Setup choices saved; continuing in Terminal.")
    return config


def _ensure_source(config: SetupConfig, *, stream: TextIO) -> Path:
    if config.source_dir.exists():
        if not config.source_dir.is_dir():
            raise Krea2TurboMlxError(
                f"Source path must be a directory: {config.source_dir}"
            )
        _print(stream, f"[download] reusing source at {config.source_dir}")
        return config.source_dir

    _print(stream, f"[download] fetching {config.source} into {config.source_dir}")
    source = download_source(
        config.source,
        revision=config.revision,
        local_dir=config.source_dir,
        progress_callback=lambda message: _print(stream, f"[download] {message}"),
    )
    _print(stream, f"[download] ready at {source}")
    return source


def _ensure_artifact(config: SetupConfig, *, stream: TextIO) -> tuple[Path, Path | None]:
    status = _artifact_status(config.artifact_dir)
    if status == "ready":
        _print(stream, f"[convert] reusing artifact at {config.artifact_dir}")
        return config.artifact_dir, None
    if status == "blocked":
        raise Krea2TurboMlxError(
            f"Cannot write artifact because {config.artifact_dir} is not empty and "
            "does not look like a krea-2-turbo-mlx artifact."
        )

    source = _ensure_source(config, stream=stream)
    _print(stream, f"[convert] writing full-precision artifact to {config.artifact_dir}")
    run_conversion(
        source,
        revision=config.revision,
        output=config.artifact_dir,
        progress_callback=lambda message: _print(stream, f"[convert] {message}"),
    )
    _print(stream, f"[convert] ready at {config.artifact_dir}")
    return config.artifact_dir, source


def _doctor_or_raise(model: Path, *, stream: TextIO) -> None:
    _print(stream, f"[doctor] validating {model}")
    report = run_doctor(model=model, runtime_required=True)
    for line in format_doctor_report(report).splitlines():
        _print(stream, f"[doctor] {line}")
    if int(report.get("error_count", 0)):
        raise Krea2TurboMlxError(f"doctor failed for {model}")


def _cleanup_source(path: Path, *, stream: TextIO) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    project = Path.cwd().resolve()
    if resolved == project:
        _print(stream, f"[cleanup] keeping source at {path}; refusing to remove project root")
        return
    try:
        resolved.relative_to(project)
    except ValueError:
        _print(stream, f"[cleanup] keeping source at {path}; it is outside this project")
        return
    if not path.is_dir() or not (path / "model_index.json").is_file():
        _print(
            stream,
            f"[cleanup] keeping source at {path}; it does not look like a model source",
        )
        return
    shutil.rmtree(path)
    _print(stream, f"[cleanup] removed source at {path}")


def _artifact_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "blocked"
    if (path / "artifact.json").is_file():
        return "ready"
    return "empty" if not any(path.iterdir()) else "blocked"


def _environment_errors() -> list[str]:
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10 or newer is required; found {platform.python_version()}"
        )
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin":
        errors.append(f"macOS is required for MLX setup; found {system}")
    if machine not in {"arm64", "aarch64"}:
        errors.append(f"Apple Silicon is required for MLX setup; found {machine}")
    return errors


def _print_setup_summary(config: SetupConfig, config_path: Path, *, stream: TextIO) -> None:
    _print(stream, f"{PROJECT_NAME} setup")
    _print(stream, f"Config: {config_path}")
    source = config.source
    if config.revision:
        source = f"{source}@{config.revision}"
    _print(stream, f"Source: {source} -> {config.source_dir}")
    _print(stream, f"Artifact: full-precision -> {config.artifact_dir}")
    _print(
        stream,
        "Source cleanup: "
        + (
            "remove project-local source after validation"
            if config.cleanup_source
            else "keep source after validation"
        ),
    )
    _print(stream, f"Outputs: {config.output_dir}")
    _print(stream, f"LoRAs: {config.lora_dir}")


def _format_try_command(model: Path, config: SetupConfig) -> str:
    output = config.output_dir / "glass-observatory.png"
    parts = [
        _recommended_cli_command(),
        "generate",
        "--model",
        str(model),
        "--prompt",
        "a glass observatory at sunrise",
        "--width",
        str(DEFAULT_GENERATION_WIDTH),
        "--height",
        str(DEFAULT_GENERATION_HEIGHT),
        "--seed",
        "42",
        "--output",
        str(output),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _recommended_cli_command() -> str:
    script = Path(sys.executable).with_name(PROJECT_NAME)
    if not script.exists():
        return PROJECT_NAME
    try:
        return str(script.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(script)


def _write_gui_launcher(config_path: Path) -> Path:
    launcher = Path.cwd() / DEFAULT_GUI_LAUNCHER
    config_arg = shlex.quote(str(config_path))
    script_template = r"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec ./scripts/launch.sh --config __GENERATED_CONFIG_PATH__ "$@"
"""
    script = script_template.replace("__GENERATED_CONFIG_PATH__", config_arg)
    launcher.write_text(script, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def _print_launcher_notice(stream: TextIO, launcher: Path) -> None:
    lines = [
        "GUI launcher ready",
        "",
        f"Double-click: {launcher.name}",
        f"Location:     {launcher}",
        "",
        "The launcher opens the local React GUI for generation.",
    ]
    width = max(len(line) for line in lines) + 4
    border = "+" + "-" * (width - 2) + "+"
    _print(stream, "")
    _print(stream, border)
    for line in lines:
        _print(stream, "| " + line.ljust(width - 4) + " |")
    _print(stream, border)


def _apply_cli_overrides(config: SetupConfig, args: Any) -> SetupConfig:
    updates: dict[str, Any] = {}
    for field_name, arg_name in (
        ("source", "source"),
        ("revision", "revision"),
        ("source_dir", "source_dir"),
        ("artifact_dir", "artifact_dir"),
        ("output_dir", "output_dir"),
        ("lora_dir", "lora_dir"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = value
    cleanup_source = getattr(args, "cleanup_source", None)
    if cleanup_source is not None:
        updates["cleanup_source"] = bool(cleanup_source)
    if updates:
        config = replace(config, **updates)
    return config.validate()


def _load_config(path: Path) -> SetupConfig:
    return SetupConfig.from_mapping(read_json_object(path))


def _save_config(config: SetupConfig, path: Path) -> None:
    write_json(path, config.to_mapping())


def _setup_handler(
    default: SetupConfig,
    done: threading.Event,
    result: dict[str, Any],
    *,
    session_token: str,
) -> type[BaseHTTPRequestHandler]:
    class SetupHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                validate_local_request(
                    headers=self.headers,
                    path=self.path,
                    expected_token=session_token,
                    allow_unsafe_host=False,
                    require_same_origin=False,
                )
            except Krea2TurboMlxError as exc:
                self.send_error(403, str(exc))
                return
            self._write_html(_form_html(default, session_token))

        def do_POST(self) -> None:
            try:
                validate_local_request(
                    headers=self.headers,
                    path=self.path,
                    expected_token=session_token,
                    allow_unsafe_host=False,
                    require_same_origin=True,
                )
            except Krea2TurboMlxError as exc:
                self.send_error(403, str(exc))
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100_000:
                    raise Krea2TurboMlxError("Setup form submission is too large.")
                body = self.rfile.read(length).decode("utf-8")
                fields = parse_qs(body, keep_blank_values=True)
                validate_session_token(_field(fields, SESSION_TOKEN_FIELD), session_token)
                result["config"] = SetupConfig.from_form(fields)
                self._write_html(_done_html())
            except Krea2TurboMlxError as exc:
                result["error"] = exc
                self._write_html(_done_html(error=str(exc)), status=400)
            except Exception as exc:
                _log_unexpected_exception(sys.stderr, exc)
                result["error"] = Krea2TurboMlxError(
                    "An unexpected setup error occurred. See the Terminal for details."
                )
                self._write_html(
                    _done_html(
                        error="An unexpected setup error occurred. See the Terminal for details."
                    ),
                    status=500,
                )
            finally:
                done.set()

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _write_html(self, text: str, *, status: int = 200) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return SetupHandler


def _form_html(config: SetupConfig, session_token: str) -> str:
    cleanup_source_checked = " checked" if config.cleanup_source else ""
    revision = "" if config.revision is None else config.revision
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PROJECT_NAME} setup</title>
  <style>
    body {{ font: 16px/1.45 -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; max-width: 760px; }}
    fieldset {{ border: 1px solid #ccc; margin: 0 0 20px; padding: 16px; }}
    label {{ display: block; margin: 10px 0; }}
    p.note {{ color: #555; margin: 8px 0 14px; }}
    input[type=text] {{ box-sizing: border-box; font: inherit; padding: 8px; width: 100%; }}
    button {{ font: inherit; padding: 10px 14px; }}
  </style>
</head>
<body>
  <h1>{PROJECT_NAME} setup</h1>
  <form method="post" action="?{urlencode({'token': session_token})}">
    <input type="hidden" name="{SESSION_TOKEN_FIELD}" value="{_escape_attr(session_token)}">
    <fieldset>
      <legend>Folders</legend>
      <p class="note">If the source folder exists, setup reuses it. If it is missing, setup downloads the pinned Diffusers source snapshot to that path.</p>
      {_text_input("source", "Hugging Face source", config.source)}
      {_text_input("revision", "Hugging Face revision", revision)}
      {_text_input("source_dir", "Downloaded source folder", config.source_dir)}
      {_text_input("artifact_dir", "Converted artifact folder", config.artifact_dir)}
      {_text_input("output_dir", "Generated image folder", config.output_dir)}
      {_text_input("lora_dir", "Local LoRA folder", config.lora_dir)}
    </fieldset>
    <fieldset>
      <legend>Cleanup</legend>
      <label><input type="checkbox" name="cleanup_source"{cleanup_source_checked}> Remove project-local source folder after artifact validation</label>
      <p class="note">Removing the source saves disk space. Preparing a fresh artifact later will require downloading the source again or choosing an existing source folder.</p>
    </fieldset>
    <button type="submit">Start setup</button>
  </form>
</body>
</html>
"""


def _done_html(error: str | None = None) -> str:
    if error:
        heading = "Setup could not start"
        body = html.escape(error)
    else:
        heading = "Setup is ready to continue"
        body = "Return to Terminal to watch download, conversion, validation, and cleanup."
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{PROJECT_NAME} setup</title></head>
<body>
  <h1>{heading}</h1>
  <p>{body}</p>
</body>
</html>
"""


def _text_input(name: str, label: str, value: str | Path) -> str:
    return (
        f'<label>{html.escape(label)}'
        f'<input type="text" name="{html.escape(name)}" '
        f'value="{_escape_attr(value)}"></label>'
    )


def _escape_attr(value: str | Path) -> str:
    return html.escape(str(value), quote=True)


def _field(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name, [""])
    return values[0] if values else ""


def _path(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default
    return Path(str(value)).expanduser()


def _non_empty_str(value: Any, default: str) -> str:
    text = default if value is None else str(value).strip()
    return text or default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _print(stream: TextIO, message: str) -> None:
    print(message, file=stream, flush=True)


def _log_unexpected_exception(stream: TextIO, exc: Exception) -> None:
    _print(stream, f"Unexpected {type(exc).__name__}: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
