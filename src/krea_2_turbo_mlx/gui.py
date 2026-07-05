from __future__ import annotations

import base64
import binascii
import json
import math
import mimetypes
import secrets
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, TextIO
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from .constants import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DISTILLED_SHIFT,
    DEFAULT_GENERATION_HEIGHT,
    DEFAULT_GENERATION_STEPS,
    DEFAULT_GENERATION_WIDTH,
    DEFAULT_GUI_PORT,
    DEFAULT_GUI_SETTINGS_FILENAME,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LORA_DIR,
    DEFAULT_OUTPUT_DIR,
    MAX_GENERATION_SIZE,
    MAX_GENERATION_SEED,
    OUTPUT_ALIGNMENT,
    PROJECT_NAME,
)
from .errors import Krea2TurboMlxError
from .generation_validation import (
    validate_generation_dimension,
    validate_generation_dimensions,
)
from .json_io import json_safe, write_json
from .local_server_security import (
    new_session_token,
    redact_session_tokens,
    request_has_valid_session_token,
    validate_local_request,
    validate_loopback_request_host,
    validate_loopback_bind_host,
)
from .lora import (
    LOCAL_LORA_DEFAULT_SCALE,
    LOCAL_LORA_SCALE_MAX,
    LOCAL_LORA_SCALE_MIN,
    LoraReference,
    lora_metadata,
    normalize_lora_payload,
    resolve_lora_patches,
    scan_lora_catalog,
)
from .png import (
    PNG_METADATA_KEY,
    PNG_PARAMETERS_KEY,
    generation_metadata_payload,
    model_precision_label,
    save_generation_png,
)
from .pipeline import (
    DEFAULT_PREVIEW_INTERVAL_STEPS,
    MAX_PREVIEW_INTERVAL_STEPS,
    PipelinePreviewFrame,
    validate_live_preview_mode,
    validate_preview_interval_steps,
)
from .setup_flow import SetupConfig

MAX_GUI_SIZE = MAX_GENERATION_SIZE
MAX_GUI_BATCH_JOBS = 100
MIN_SIMPLE_BATCH_COUNT = 2
MAX_REQUEST_BYTES = 64_000
MAX_GUI_BATCH_BYTES = 256_000
MAX_SOURCE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_IMAGE_JSON_BYTES = MAX_SOURCE_IMAGE_BYTES * 2
MAX_EVENT_HISTORY = 200
MAX_INITIAL_RECENT_GENERATIONS = 500
_BATCH_JOB_KEYS = frozenset(
    {"prompt", "width", "height", "steps", "seed", "loras"}
)
_SOURCE_IMAGE_SETTING_KEYS = ("prompt", "width", "height", "steps", "seed", "loras")
_PIPELINE_LOADER = Callable[[Path, Any], Any]
_GUI_THEME_MODES = frozenset(("system", "light", "dark"))
_GUI_SETTINGS_KEYS = frozenset(
    ("theme", "generation", "live_preview", "loras", "simple_batch")
)
_GUI_GENERATION_KEYS = frozenset(
    ("width", "height", "steps", "randomization_locked")
)
_GUI_LIVE_PREVIEW_KEYS = frozenset(("mode", "interval_steps"))
_GUI_SIMPLE_BATCH_KEYS = frozenset(("enabled", "count"))
_INITIAL_STATUS_GLOBAL = "__KREA_2_TURBO_MLX_INITIAL_STATUS__"


class Krea2TurboGenerationCancelled(Krea2TurboMlxError):
    """Raised inside the generation callback when the user requests cancellation."""


@dataclass(frozen=True)
class _GenerationRequest:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    loras: tuple[LoraReference, ...] = ()
    live_preview: str = "off"
    preview_interval_steps: int = DEFAULT_PREVIEW_INTERVAL_STEPS


@dataclass(frozen=True)
class _GenerationRecord:
    id: int
    path: Path
    prompt: str
    seed: int
    width: int
    height: int
    steps: int
    model_path: str
    model_precision: str
    created_at: str
    loras: tuple[dict[str, Any], ...] = ()

    @property
    def url(self) -> str:
        return f"/api/image/{self.id}"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "prompt": self.prompt,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "model_path": self.model_path,
            "model_precision": self.model_precision,
            "created_at": self.created_at,
            "filename": self.path.name,
            "loras": [dict(item) for item in self.loras],
        }


@dataclass(frozen=True)
class _PreviewImage:
    revision: int
    mode: str
    step: int
    step_count: int
    width: int
    height: int
    data: bytes
    content_type: str = "image/jpeg"

    @property
    def url(self) -> str:
        return f"/api/preview/current?{urlencode({'rev': self.revision})}"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "revision": self.revision,
            "step": self.step,
            "step_count": self.step_count,
            "width": self.width,
            "height": self.height,
            "url": self.url,
        }


def run_gui(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    host: str = "127.0.0.1",
    port: int = DEFAULT_GUI_PORT,
    allow_unsafe_host: bool = False,
    open_browser: bool = True,
    preload: bool = True,
    stream: TextIO = sys.stdout,
) -> int:
    validate_loopback_bind_host(
        host,
        allow_unsafe_host=allow_unsafe_host,
        server_name="GUI",
    )
    config_path = Path(config_path).expanduser()
    config = _load_config(config_path)
    config.lora_dir.mkdir(parents=True, exist_ok=True)
    settings_path = _gui_settings_path_for_config(config_path)
    static_root = _frontend_static_root()
    _validate_static_root(static_root)

    state = _GuiState(config=config, stream=stream, settings_path=settings_path)
    session_token = new_session_token()
    handler = _gui_handler(
        state,
        static_root=static_root,
        session_token=session_token,
        allow_unsafe_host=allow_unsafe_host,
    )
    try:
        server = ThreadingHTTPServer((host, int(port)), handler)
    except OSError as exc:
        raise Krea2TurboMlxError(
            f"Could not start GUI server on {_url_host(host)}:{port}: {exc}. "
            "Pass --port PORT to choose a different fixed port, or --port 0 "
            "to pick a random free port."
        ) from exc
    resolved_port = int(server.server_address[1])
    url = f"http://{_url_host(host)}:{resolved_port}/?{urlencode({'token': session_token})}"

    _print_gui_banner(
        stream,
        url=url,
        model=config.artifact_dir,
        config_path=config_path,
        output_dir=config.output_dir,
    )
    state.record_system_event(
        "server",
        "GUI server started",
        progress=1.0,
        details={"url": url, "host": host, "port": resolved_port},
    )

    if preload:
        threading.Timer(0.1, state.start_load).start()
    if open_browser:
        threading.Timer(0.2, _open_browser, args=(url, stream)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log(stream, "Shutdown requested from Terminal.")
    finally:
        state.request_shutdown()
        server.server_close()
        state.eject_model()
        _log(stream, "GUI server stopped.")
    return 0


@dataclass
class _GuiState:
    config: SetupConfig
    stream: TextIO
    settings_path: Path | None = None
    pipeline_loader: _PIPELINE_LOADER = field(default_factory=lambda: _load_pipeline)
    lock: threading.RLock = field(default_factory=threading.RLock)
    pipeline_lock: threading.Lock = field(default_factory=threading.Lock)
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1
    pipeline: Any | None = None
    phase: str = "starting"
    message: str = "Starting"
    progress: float = 0.0
    error: str | None = None
    load_running: bool = False
    generation_running: bool = False
    cancel_requested: bool = False
    batch_clear_queue_requested: bool = False
    batch_active: bool = False
    batch_index: int = 0
    batch_total: int = 0
    batch_current: _GenerationRequest | None = None
    batch_jobs: list[_GenerationRequest] = field(default_factory=list)
    batch_job_statuses: list[str] = field(default_factory=list)
    batch_started_ms: int | None = None
    batch_current_started_ms: int | None = None
    recent_generations: list[_GenerationRecord] = field(default_factory=list)
    current_image: _GenerationRecord | None = None
    preview: _PreviewImage | None = None
    next_preview_revision: int = 1
    next_generation_id: int = 1
    generation_ids_by_path: dict[Path, int] = field(default_factory=dict)
    output_dir_signature: tuple[tuple[str, int, int], ...] = field(default_factory=tuple)
    task_name: str | None = None
    task_started_ms: int | None = None
    task_completed_ms: int | None = None
    model_precision: str = field(init=False)
    ui_settings: dict[str, Any] = field(init=False)
    lora_catalog: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.config.lora_dir.mkdir(parents=True, exist_ok=True)
        self.model_precision = model_precision_label(self.config.artifact_dir)
        self.ui_settings = _read_gui_settings(self.settings_path)
        self.lora_catalog = scan_lora_catalog(self.config.lora_dir).to_mapping()
        self._load_existing_generations()

    def snapshot(self) -> dict[str, Any]:
        self._sync_output_dir()
        with self.lock:
            image = self.current_image.to_mapping() if self.current_image else None
            preview = self.preview.to_mapping() if self.preview else None
            busy = self.load_running or self.generation_running
            return {
                "server": {"connected": True, "status": "running"},
                "model": {
                    "loaded": self.pipeline is not None,
                    "name": PROJECT_NAME,
                    "status": "in memory" if self.pipeline is not None else "not loaded",
                    "path": str(self.config.artifact_dir),
                    "precision": self.model_precision,
                },
                "busy": busy,
                "load_running": self.load_running,
                "generation_running": self.generation_running,
                "cancel_requested": self.cancel_requested,
                "batch": self._batch_snapshot_locked(),
                "phase": self.phase,
                "message": self.message,
                "progress": self.progress,
                "error": self.error,
                "image": image,
                "preview": preview,
                "recent": [item.to_mapping() for item in self.recent_generations],
                "output_dir": {"path": _output_dir_status_path(self.config.output_dir)},
                "loras": _copy_lora_catalog(self.lora_catalog),
                "ui_settings": _copy_gui_settings(self.ui_settings),
                "constraints": {
                    "alignment": OUTPUT_ALIGNMENT,
                    "max_size": MAX_GUI_SIZE,
                    "max_seed": MAX_GENERATION_SEED,
                    "default_width": DEFAULT_GENERATION_WIDTH,
                    "default_height": DEFAULT_GENERATION_HEIGHT,
                    "default_steps": DEFAULT_GENERATION_STEPS,
                    "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                    "shift": DEFAULT_DISTILLED_SHIFT,
                    "max_batch_jobs": MAX_GUI_BATCH_JOBS,
                    "local_lora_scale_min": LOCAL_LORA_SCALE_MIN,
                    "local_lora_scale_max": LOCAL_LORA_SCALE_MAX,
                    "default_local_lora_scale": LOCAL_LORA_DEFAULT_SCALE,
                },
                "events": redact_session_tokens(list(self.events[-40:])),
                "task": {
                    "name": self.task_name,
                    "started_ms": self.task_started_ms,
                    "completed_ms": self.task_completed_ms,
                },
            }

    def start_load(self) -> tuple[bool, str]:
        with self.lock:
            if self.pipeline is not None:
                self.record_system_event_locked(
                    "model_load",
                    "Model is already in memory",
                    progress=1.0,
                )
                return False, "Model is already in memory."
            if self.load_running or self.generation_running:
                return False, "The model is busy right now."
            self.load_running = True
            self.phase = "model_load"
            self.message = "Loading model into memory"
            self.progress = 0.0
            self.error = None
            self._begin_task_locked("Load model")
        thread = threading.Thread(target=self._load_worker, daemon=True)
        thread.start()
        return True, "Model load started."

    def start_eject(self) -> tuple[bool, str]:
        with self.lock:
            if self.load_running or self.generation_running:
                return False, "Wait for the current task to finish before ejecting."
        self.eject_model()
        return True, "Model ejected."

    def eject_model(self) -> None:
        with self.pipeline_lock:
            with self.lock:
                self.pipeline = None
                self.phase = "model_load"
                self.message = "Model ejected"
                self.progress = 0.0
                self.error = None
                self._clear_preview_locked()
                self._clear_task_locked()
            _clear_runtime_caches()
        self.record_system_event(
            "model_load",
            "Model ejected from memory",
            progress=0.0,
        )

    def start_generation(self, request: _GenerationRequest) -> tuple[bool, str]:
        with self.lock:
            if self.load_running or self.generation_running:
                return False, "A task is already running."
            self.generation_running = True
            self.cancel_requested = False
            self._clear_batch_locked()
            self._clear_preview_locked()
            self.phase = "generate"
            self.message = "Generation queued"
            self.progress = 0.0
            self.error = None
            self._begin_task_locked("Generate image")
        thread = threading.Thread(
            target=self._generation_worker,
            args=(request,),
            daemon=True,
        )
        thread.start()
        return True, "Generation started."

    def start_batch_generation(
        self,
        requests: list[_GenerationRequest],
        *,
        output_dir: Path | None = None,
    ) -> tuple[bool, str]:
        if not requests:
            return False, "Batch must include at least one job."
        resolved_output_dir = (
            _prepare_output_dir(output_dir) if output_dir is not None else None
        )
        output_dir_changed = False
        with self.lock:
            if self.load_running or self.generation_running:
                return False, "A task is already running."
            if resolved_output_dir is not None:
                output_dir_changed = resolved_output_dir != self.config.output_dir
                self._set_output_dir_locked(resolved_output_dir)
            self.generation_running = True
            self.cancel_requested = False
            self.batch_clear_queue_requested = False
            self.batch_active = True
            self._clear_preview_locked()
            self.batch_index = 0
            self.batch_total = len(requests)
            self.batch_current = None
            self.batch_jobs = list(requests)
            self.batch_job_statuses = ["queued"] * len(requests)
            self.phase = "generate"
            self.message = "Batch queued"
            self.progress = 0.0
            self.error = None
            self._begin_task_locked(_batch_task_name(len(requests)))
            self.batch_started_ms = self.task_started_ms
            self.batch_current_started_ms = None
        if output_dir_changed:
            self._sync_output_dir()
        thread = threading.Thread(
            target=self._batch_worker,
            args=(requests,),
            daemon=True,
        )
        thread.start()
        return True, "Batch generation started."

    def start_cancel_current_generation(self) -> tuple[bool, str]:
        with self.lock:
            if not self.generation_running:
                return False, "No generation is running."
            if self.batch_active and not self._batch_has_running_job_locked():
                return False, "No batch job is running."
            if self.cancel_requested:
                return True, "Cancellation is already requested."
            self.cancel_requested = True
            if self.batch_active:
                self._set_batch_job_status_locked(self.batch_index, "cancelling")
                self.message = "Cancelling current job"
                event_message = "Cancel current job requested"
            else:
                self.message = "Cancelling generation"
                event_message = "Cancel requested"
            self._clear_preview_locked()
            self.record_system_event_locked(
                "generate",
                event_message,
                progress=self.progress,
            )
        return True, "Cancellation requested."

    def start_clear_batch_queue(self) -> tuple[bool, str]:
        with self.lock:
            if not self.generation_running or not self.batch_active:
                return False, "No batch is running."
            if self.batch_clear_queue_requested:
                return True, "The batch queue is already cleared."
            self.batch_clear_queue_requested = True
            cleared_jobs = self._clear_queued_batch_jobs_locked()
            if cleared_jobs == 0:
                self.record_system_event_locked(
                    "generate",
                    "No queued batch jobs remain",
                    progress=self.progress,
                )
                return True, "No queued batch jobs remain."
            self.message = "Finishing current job; queue cleared"
            self.record_system_event_locked(
                "generate",
                "Clear queue requested",
                progress=self.progress,
                details={"cleared_jobs": cleared_jobs},
            )
        suffix = "job" if cleared_jobs == 1 else "jobs"
        return True, f"Cleared {cleared_jobs} queued {suffix}."

    def request_shutdown(self) -> bool:
        with self.lock:
            if not self.generation_running:
                return False

            self.cancel_requested = True
            self._clear_preview_locked()
            details: dict[str, Any] = {}

            if self.batch_active:
                if self._batch_has_running_job_locked():
                    self._set_batch_job_status_locked(self.batch_index, "cancelling")
                self.batch_clear_queue_requested = True
                cleared_jobs = self._clear_queued_batch_jobs_locked()
                details["cleared_jobs"] = cleared_jobs
                self.message = "Stopping; cancelling current job"
                event_message = (
                    "Shutdown requested; cancelling current job and clearing queue"
                )
            else:
                self.message = "Stopping; cancelling generation"
                event_message = "Shutdown requested; cancelling generation"

            self.record_system_event_locked(
                "generate",
                event_message,
                progress=self.progress,
                details=details,
            )
            return True

    def record_pipeline_event(self, event: Any) -> None:
        payload = _event_payload(event)
        with self.lock:
            self._append_event_locked(payload)
            self.phase = str(getattr(event, "stage", payload["stage"]))
            self.message = str(getattr(event, "message", payload["message"]))
            if getattr(event, "progress", None) is not None:
                self.progress = _clamp_progress(float(event.progress))
        _log(self.stream, _format_event_for_terminal(payload))

    def record_generation_event(self, event: Any) -> None:
        self.record_pipeline_event(event)
        self._raise_if_generation_cancelled()

    def record_preview_frame(self, frame: PipelinePreviewFrame) -> None:
        data, width, height = _preview_jpeg_from_frame(frame)
        with self.lock:
            if self.cancel_requested:
                return
            revision = self.next_preview_revision
            self.next_preview_revision += 1
            self.preview = _PreviewImage(
                revision=revision,
                mode=frame.mode,
                step=int(frame.step_index) + 1,
                step_count=int(frame.step_count),
                width=width,
                height=height,
                data=data,
            )

    def record_system_event(
        self,
        stage: str,
        message: str,
        *,
        progress: float | None = None,
        details: dict[str, Any] | None = None,
        kind: str = "system",
        time_ms: int | None = None,
        completed_ms: int | None = None,
    ) -> None:
        with self.lock:
            self.record_system_event_locked(
                stage,
                message,
                progress=progress,
                details=details,
                kind=kind,
                time_ms=time_ms,
                completed_ms=completed_ms,
            )

    def record_system_event_locked(
        self,
        stage: str,
        message: str,
        *,
        progress: float | None = None,
        details: dict[str, Any] | None = None,
        kind: str = "system",
        time_ms: int | None = None,
        completed_ms: int | None = None,
    ) -> None:
        payload = {
            **_event_time_payload(time_ms),
            "kind": kind,
            "stage": stage,
            "message": message,
            "progress": progress,
            "details": _safe_event_details(details or {}),
        }
        if completed_ms is not None:
            payload["completed_ms"] = max(int(payload["time_ms"]), int(completed_ms))
        self._append_event_locked(payload)
        self.phase = stage
        self.message = message
        if progress is not None:
            self.progress = _clamp_progress(progress)
        _log(self.stream, _format_event_for_terminal(payload))

    def _load_worker(self) -> None:
        try:
            with self.pipeline_lock:
                pipeline = self.pipeline_loader(
                    self.config.artifact_dir,
                    self.record_pipeline_event,
                )
                with self.lock:
                    self.pipeline = pipeline
        except Exception as exc:
            self._finish_with_error(exc, phase="model_load")
        finally:
            with self.lock:
                self.load_running = False
                if self.error is None and self.pipeline is not None:
                    self.phase = "model_load"
                    self.message = "Model ready in memory"
                    self.progress = 1.0
                self._complete_task_locked()

    def _generation_worker(self, request: _GenerationRequest) -> None:
        try:
            with self.pipeline_lock:
                self._run_single_job_locked(request)
        except Krea2TurboGenerationCancelled:
            self._finish_cancelled()
        except Exception as exc:
            self._finish_with_error(exc, phase="generate")
        finally:
            with self.lock:
                self.generation_running = False
                self.cancel_requested = False
                self._clear_batch_locked()
                self._complete_task_locked()

    def _batch_worker(self, requests: list[_GenerationRequest]) -> None:
        try:
            total = len(requests)
            for index, request in enumerate(requests, start=1):
                with self.lock:
                    if self.batch_clear_queue_requested:
                        self._clear_queued_batch_jobs_locked()
                        break
                    self.batch_current_started_ms = _timestamp_ms()
                    self.batch_active = True
                    self._clear_preview_locked()
                    self.batch_index = index
                    self.batch_total = total
                    self.batch_current = request
                    self._set_batch_job_status_locked(index, "running")
                    self.phase = "generate"
                    self.message = f"Job {index} of {total}"
                    self.progress = 0.0
                    self.record_system_event_locked(
                        "generate",
                        self.message,
                        progress=0.0,
                        details={
                            "job": index,
                            "total": total,
                            "seed": request.seed,
                            "height": request.height,
                            "width": request.width,
                            "steps": request.steps,
                        },
                    )
                try:
                    with self.pipeline_lock:
                        self._run_single_job_locked(request)
                except Krea2TurboGenerationCancelled:
                    with self.lock:
                        self.cancel_requested = False
                        self._clear_preview_locked()
                        self._set_batch_job_status_locked(index, "cancelled")
                        message = f"Job {index} of {total} cancelled"
                        self.phase = (
                            "cancelled"
                            if self.batch_clear_queue_requested or index >= total
                            else "generate"
                        )
                        self.message = message
                        self.record_system_event_locked(
                            self.phase,
                            message,
                            progress=self.progress,
                            details={"job": index, "total": total},
                        )
                else:
                    with self.lock:
                        self._set_batch_job_status_locked(index, "done")
                with self.lock:
                    if self.batch_clear_queue_requested:
                        cleared_jobs = self._clear_queued_batch_jobs_locked()
                        if cleared_jobs > 0:
                            self.record_system_event_locked(
                                "generate",
                                "Batch queue cleared",
                                progress=_batch_overall_progress(
                                    self.batch_index,
                                    self.batch_total,
                                    self.progress,
                                ),
                                details={"cleared_jobs": cleared_jobs},
                            )
                        break
        except Exception as exc:
            with self.lock:
                index = self.batch_index
                total = self.batch_total
            self._finish_with_error(
                Krea2TurboMlxError(f"Job {index} of {total} failed: {exc}"),
                phase="generate",
            )
        finally:
            with self.lock:
                self.generation_running = False
                self.cancel_requested = False
                self._clear_batch_locked()
                self._complete_task_locked()

    def _run_single_job_locked(self, request: _GenerationRequest) -> None:
        self._raise_if_generation_cancelled()
        pipeline = self.pipeline
        if pipeline is None:
            self.record_system_event(
                "model_load",
                "Loading model into memory",
                progress=0.0,
            )
            pipeline = self.pipeline_loader(
                self.config.artifact_dir,
                self.record_pipeline_event,
            )
            with self.lock:
                self.pipeline = pipeline
        self._raise_if_generation_cancelled()

        started = time.perf_counter()
        lora_patches = _resolve_request_loras(
            pipeline,
            request.loras,
            lora_dir=self.config.lora_dir,
        )
        active_lora_metadata = lora_metadata(lora_patches)
        generate_kwargs = {
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "seed": request.seed,
            "progress_callback": self.record_generation_event,
            "live_preview": request.live_preview,
            "preview_interval_steps": request.preview_interval_steps,
            "preview_callback": self.record_preview_frame,
        }
        if lora_patches:
            generate_kwargs["loras"] = lora_patches
        result = pipeline(request.prompt, **generate_kwargs)
        elapsed = (
            float(result.elapsed_seconds)
            if getattr(result, "elapsed_seconds", None) is not None
            else time.perf_counter() - started
        )
        effective_seed = int(getattr(result, "seed", request.seed))
        output_path = self._next_output_path(request, seed=effective_seed)
        metadata = generation_metadata_payload(
            prompt=request.prompt,
            seed=effective_seed,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            shift=DEFAULT_DISTILLED_SHIFT,
            model_path=self.config.artifact_dir,
            elapsed_seconds=elapsed,
            truncation_warnings=tuple(
                getattr(result, "truncation_warnings", ()) or ()
            ),
            loras=active_lora_metadata,
        )
        self._raise_if_generation_cancelled()
        self.record_system_event(
            "output",
            "Saving image",
            progress=0.98,
            details={"path": str(output_path)},
        )
        save_generation_png(
            getattr(result, "images", result),
            output_path,
            metadata=metadata,
            overwrite=False,
        )
        self._raise_if_generation_cancelled()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            record = _GenerationRecord(
                id=self.next_generation_id,
                path=output_path,
                prompt=request.prompt,
                seed=effective_seed,
                width=request.width,
                height=request.height,
                steps=request.steps,
                model_path=str(self.config.artifact_dir),
                model_precision=self.model_precision,
                created_at=created_at,
                loras=_generation_record_loras(active_lora_metadata),
            )
            self.next_generation_id += 1
            self.generation_ids_by_path[_generation_path_key(output_path)] = record.id
            self.recent_generations.insert(0, record)
            self.current_image = record
            self._clear_preview_locked()
            self.phase = "complete"
            self.message = f"Saved {output_path.name}"
            self.progress = 1.0
            self.error = None
        self.record_system_event(
            "output",
            "Image saved",
            progress=1.0,
            details={
                "path": str(output_path),
                "seed": effective_seed,
                "height": request.height,
                "width": request.width,
                "steps": request.steps,
            },
        )

    def _raise_if_generation_cancelled(self) -> None:
        with self.lock:
            if self.cancel_requested:
                raise Krea2TurboGenerationCancelled("Generation cancelled by user")

    def _finish_cancelled(self) -> None:
        with self.lock:
            self.phase = "cancelled"
            self.message = "Generation cancelled"
            self.error = None
            self._clear_preview_locked()
            self.record_system_event_locked(
                "cancelled",
                "Generation cancelled",
                progress=self.progress,
            )

    def _finish_with_error(self, exc: Exception, *, phase: str) -> None:
        expected = isinstance(exc, Krea2TurboMlxError)
        if expected:
            message = str(exc)
        else:
            _log_unexpected_exception(self.stream, exc)
            message = "An unexpected error occurred. See the Terminal for details."
        with self.lock:
            self.phase = phase
            self.message = message
            self.error = message
            self._clear_preview_locked()
        self.record_system_event(
            phase,
            message,
            details={"error_type": type(exc).__name__},
            kind="error",
        )

    def _next_output_path(self, request: _GenerationRequest, *, seed: int) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = self.config.output_dir
        filename = (
            f"krea-gui-{stamp}-{request.width}x{request.height}-"
            f"steps{request.steps}-seed{seed}.png"
        )
        path = output_dir / filename
        counter = 2
        while path.exists():
            path = output_dir / (
                f"krea-gui-{stamp}-{counter}-{request.width}x{request.height}-"
                f"steps{request.steps}-seed{seed}.png"
            )
            counter += 1
        return path

    def _batch_snapshot_locked(self) -> dict[str, Any] | None:
        if not self.batch_active:
            return None
        current = self.batch_current
        statuses = list(self.batch_job_statuses)
        payload: dict[str, Any] = {
            "index": self.batch_index,
            "total": self.batch_total,
            "started_ms": self.batch_started_ms,
            "current_started_ms": self.batch_current_started_ms,
            "progress": _clamp_progress(self.progress),
            "cancel_current_requested": self.cancel_requested,
            "clear_queue_requested": self.batch_clear_queue_requested,
            "queue_remaining": sum(1 for status in statuses if status == "queued"),
            "overall_progress": _batch_overall_progress(
                self.batch_index,
                self.batch_total,
                self.progress,
            ),
            "jobs": [
                {
                    "index": index,
                    "status": self._batch_job_status_locked(index),
                    **_generation_request_snapshot(request),
                }
                for index, request in enumerate(self.batch_jobs, start=1)
            ],
        }
        if current is None:
            payload.update(
                {
                    "prompt": "",
                    "width": None,
                    "height": None,
                    "steps": None,
                    "seed": None,
                    "loras": [],
                }
            )
        else:
            payload.update(_generation_request_snapshot(current))
            payload["status"] = self._batch_job_status_locked(self.batch_index)
        return payload

    def _batch_has_running_job_locked(self) -> bool:
        return self._batch_job_status_locked(self.batch_index) in {
            "running",
            "cancelling",
        }

    def _batch_job_status_locked(self, index: int) -> str:
        if index <= 0:
            return "queued"
        offset = index - 1
        if 0 <= offset < len(self.batch_job_statuses):
            return self.batch_job_statuses[offset]
        if self.batch_active and index == self.batch_index:
            return "running" if self.cancel_requested else "queued"
        if self.batch_active and index < self.batch_index:
            return "done"
        return "queued"

    def _set_batch_job_status_locked(self, index: int, status: str) -> None:
        if index <= 0:
            return
        while len(self.batch_job_statuses) < index:
            self.batch_job_statuses.append("queued")
        self.batch_job_statuses[index - 1] = status

    def _clear_queued_batch_jobs_locked(self) -> int:
        cleared = 0
        for index, status in enumerate(self.batch_job_statuses):
            if status == "queued":
                self.batch_job_statuses[index] = "cleared"
                cleared += 1
        return cleared

    def _clear_batch_locked(self) -> None:
        self.batch_active = False
        self.batch_clear_queue_requested = False
        self.batch_index = 0
        self.batch_total = 0
        self.batch_current = None
        self.batch_jobs = []
        self.batch_job_statuses = []
        self.batch_started_ms = None
        self.batch_current_started_ms = None

    def _clear_preview_locked(self) -> None:
        self.preview = None

    def current_preview(self, revision: int | None = None) -> _PreviewImage | None:
        with self.lock:
            preview = self.preview
            if preview is None:
                return None
            if revision is not None and preview.revision != revision:
                return None
            return preview

    def image_path_for_id(self, image_id: int) -> Path | None:
        self._sync_output_dir()
        with self.lock:
            for record in self.recent_generations:
                if record.id == image_id:
                    return record.path
        return None

    def latest_image_path(self) -> Path | None:
        self._sync_output_dir()
        with self.lock:
            if self.current_image is not None:
                return self.current_image.path
            if self.recent_generations:
                return self.recent_generations[0].path
        return None

    def open_output_dir(self) -> None:
        path = self.config.output_dir
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(path)], check=False)
        self.record_system_event(
            "output",
            "Opened output folder",
            details={"path": str(path)},
        )

    def choose_output_dir(self) -> Path | None:
        return _choose_output_dir(self.current_output_dir())

    def current_output_dir(self) -> Path:
        with self.lock:
            return self.config.output_dir

    def update_ui_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            settings = _merge_gui_settings_patch(self.ui_settings, payload)
            _write_gui_settings(self.settings_path, settings)
            self.ui_settings = settings
            return _copy_gui_settings(settings)

    def refresh_lora_catalog(self) -> dict[str, Any]:
        catalog = scan_lora_catalog(self.config.lora_dir).to_mapping()
        with self.lock:
            self.lora_catalog = catalog
            return _copy_lora_catalog(catalog)

    def read_batch_clipboard(self) -> tuple[str, str]:
        return _read_batch_clipboard()

    def source_image_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_id = payload.get("image_id")
        if image_id is not None:
            path = self.image_path_for_id(_parse_gui_int(image_id, "image id"))
            if path is None:
                raise Krea2TurboMlxError("Image not found.")
            return _source_image_metadata_from_path(path)

        return _source_image_metadata_from_payload(payload)

    def read_source_image_clipboard(self) -> dict[str, Any]:
        return _read_source_image_clipboard()

    def delete_image(self, image_id: int) -> tuple[bool, str]:
        self._sync_output_dir()
        with self.lock:
            record = next(
                (item for item in self.recent_generations if item.id == image_id),
                None,
            )
            if record is None:
                return False, "Image not found."
            path = record.path
        started = time.perf_counter()
        started_ms = _timestamp_ms()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise Krea2TurboMlxError(f"Could not delete image: {path}") from exc
        self._sync_output_dir()
        with self.lock:
            completed_ms = max(started_ms, _timestamp_ms())
            self.record_system_event_locked(
                "output",
                f"Deleted {path.name}",
                details={
                    "elapsed_seconds": time.perf_counter() - started,
                    "path": str(path),
                },
                time_ms=started_ms,
                completed_ms=completed_ms,
            )
        return True, "Image deleted."

    def _begin_task_locked(self, name: str) -> None:
        self._complete_current_event_locked()
        self.task_name = name
        self.task_started_ms = _timestamp_ms()
        self.task_completed_ms = None

    def _complete_task_locked(self) -> None:
        completed_ms = _timestamp_ms()
        self.task_completed_ms = completed_ms
        self._complete_current_event_locked(completed_ms=completed_ms)
        if self._should_record_task_summary_locked():
            self._append_task_summary_event_locked(completed_ms=completed_ms)
            self.progress = 0.0

    def _clear_task_locked(self) -> None:
        self.task_name = None
        self.task_started_ms = None
        self.task_completed_ms = None

    def _append_event_locked(self, payload: dict[str, Any]) -> None:
        time_ms = int(payload["time_ms"])
        self._complete_current_event_locked(completed_ms=time_ms)
        payload["id"] = self.next_event_id
        self.next_event_id += 1
        self.events.append(payload)
        del self.events[:-MAX_EVENT_HISTORY]

    def _complete_current_event_locked(self, *, completed_ms: int | None = None) -> None:
        if not self.events:
            return

        current = self.events[-1]
        if current.get("completed_ms") is not None:
            return

        current["completed_ms"] = completed_ms if completed_ms is not None else _timestamp_ms()

    def _should_record_task_summary_locked(self) -> bool:
        return (
            bool(self.task_name)
            and self.task_started_ms is not None
            and self.task_completed_ms is not None
            and self.error is None
            and self.phase != "cancelled"
        )

    def _append_task_summary_event_locked(self, *, completed_ms: int) -> None:
        if self.task_started_ms is None or not self.task_name:
            return

        duration_ms = max(0, completed_ms - self.task_started_ms)
        payload = {
            **_event_time_payload(completed_ms),
            "kind": "task",
            "stage": "task",
            "message": self.task_name,
            "progress": 1.0,
            "completed_ms": completed_ms,
            "details": {"elapsed_seconds": duration_ms / 1000.0},
        }
        self._append_event_locked(payload)
        _log(self.stream, _format_event_for_terminal(payload))

    def _load_existing_generations(self) -> None:
        self._sync_output_dir()

    def _set_output_dir_locked(self, output_dir: Path) -> None:
        if output_dir == self.config.output_dir:
            return

        self.config = replace(self.config, output_dir=output_dir)
        self.output_dir_signature = ()

    def _sync_output_dir(self) -> None:
        while True:
            with self.lock:
                output_dir = self.config.output_dir

            entries = _generation_file_entries(output_dir)
            signature = _generation_signature(entries)

            with self.lock:
                if output_dir != self.config.output_dir:
                    continue
                self._apply_output_dir_entries_locked(entries, signature)
                return

    def _apply_output_dir_entries_locked(
        self,
        entries: list[tuple[Path, int, int]],
        signature: tuple[tuple[str, int, int], ...],
    ) -> None:
        if signature == self.output_dir_signature:
            return

        records: list[_GenerationRecord] = []
        live_paths: set[Path] = set()
        for path, _, _ in entries[:MAX_INITIAL_RECENT_GENERATIONS]:
            path_key = _generation_path_key(path)
            record_id = self.generation_ids_by_path.get(path_key)
            if record_id is None:
                record_id = self.next_generation_id
                self.generation_ids_by_path[path_key] = record_id
                self.next_generation_id += 1
            record = _record_from_png(
                path,
                record_id=record_id,
                default_model=str(self.config.artifact_dir),
            )
            if record is not None:
                records.append(record)
                live_paths.add(path_key)
        self.recent_generations = records
        self.generation_ids_by_path = {
            path: record_id
            for path, record_id in self.generation_ids_by_path.items()
            if path in live_paths
        }
        self.output_dir_signature = signature
        if records:
            current_id = self.current_image.id if self.current_image else None
            self.current_image = next(
                (record for record in records if record.id == current_id),
                records[0],
            )
            if self.phase == "starting":
                self.message = f"Loaded {len(records)} image(s) from outputs"
        else:
            self.current_image = None


def _gui_handler(
    state: _GuiState,
    *,
    static_root: Path,
    session_token: str,
    allow_unsafe_host: bool,
) -> type[BaseHTTPRequestHandler]:
    class GuiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                try:
                    self._validate_api_request(require_same_origin=False)
                except Krea2TurboMlxError as exc:
                    self._write_json({"ok": False, "message": str(exc)}, status=403)
                    return
                try:
                    if parsed.path == "/api/status":
                        self._write_json(state.snapshot())
                    elif parsed.path == "/api/preview/current":
                        self._write_preview(parsed.query)
                    elif parsed.path == "/api/image/latest":
                        self._write_image_path(state.latest_image_path())
                    elif parsed.path.startswith("/api/image/"):
                        self._write_image_by_id(parsed.path)
                    else:
                        self.send_error(404)
                except Krea2TurboMlxError as exc:
                    self._write_expected_request_error(exc)
                except Exception as exc:
                    self._write_unexpected_request_error(exc)
                return

            try:
                self._validate_static_request()
                path = _static_path_from_url(static_root, parsed.path)
            except Krea2TurboMlxError as exc:
                self.send_error(403, str(exc))
                return
            try:
                self._write_static(path)
            except Exception as exc:
                self._write_unexpected_request_error(exc)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._validate_api_request(require_same_origin=True)
            except Krea2TurboMlxError as exc:
                state.record_system_event(
                    "request",
                    str(exc),
                    details={"error_type": type(exc).__name__},
                )
                self._write_json({"ok": False, "message": str(exc)}, status=403)
                return

            try:
                if parsed.path == "/api/load":
                    accepted, message = state.start_load()
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=202 if accepted else 409,
                    )
                elif parsed.path == "/api/eject":
                    accepted, message = state.start_eject()
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=200 if accepted else 409,
                    )
                elif parsed.path == "/api/open-output-dir":
                    state.open_output_dir()
                    self._write_json({"ok": True, "message": "Output folder opened."})
                elif parsed.path == "/api/read-batch-clipboard":
                    text, source = state.read_batch_clipboard()
                    self._write_json({"ok": True, "source": source, "text": text})
                elif parsed.path == "/api/read-source-image-clipboard":
                    self._write_json(
                        {"ok": True, **state.read_source_image_clipboard()}
                    )
                elif parsed.path == "/api/select-output-dir":
                    path = state.choose_output_dir()
                    if path is None:
                        self._write_json(
                            {
                                "ok": False,
                                "message": "No output folder selected.",
                                "path": _output_dir_status_path(
                                    state.current_output_dir()
                                ),
                            }
                        )
                    else:
                        self._write_json(
                            {"ok": True, "path": _output_dir_status_path(path)}
                        )
                elif parsed.path == "/api/ui-settings":
                    payload = self._read_json()
                    settings = state.update_ui_settings(payload)
                    self._write_json({"ok": True, "settings": settings})
                elif parsed.path == "/api/loras/refresh":
                    catalog = state.refresh_lora_catalog()
                    self._write_json({"ok": True, "loras": catalog})
                elif parsed.path == "/api/generate":
                    payload = self._read_json()
                    request = _generation_request_from_payload(payload)
                    accepted, message = state.start_generation(request)
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=202 if accepted else 409,
                    )
                elif parsed.path == "/api/validate-batch":
                    payload = self._read_json(max_bytes=MAX_GUI_BATCH_BYTES)
                    requests = _batch_jobs_from_payload(payload)
                    self._write_json({"ok": True, "count": len(requests)})
                elif parsed.path == "/api/validate-source-image":
                    payload = self._read_json(max_bytes=MAX_SOURCE_IMAGE_JSON_BYTES)
                    self._write_json({"ok": True, **state.source_image_metadata(payload)})
                elif parsed.path == "/api/generate-batch":
                    payload = self._read_json(max_bytes=MAX_GUI_BATCH_BYTES)
                    requests = _batch_jobs_from_payload(payload)
                    output_dir = _batch_output_dir_from_payload(payload)
                    accepted, message = state.start_batch_generation(
                        requests,
                        output_dir=output_dir,
                    )
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=202 if accepted else 409,
                    )
                elif parsed.path == "/api/cancel-current":
                    accepted, message = state.start_cancel_current_generation()
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=202 if accepted else 409,
                    )
                elif parsed.path == "/api/clear-queue":
                    accepted, message = state.start_clear_batch_queue()
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=202 if accepted else 409,
                    )
                else:
                    self.send_error(404)
            except Krea2TurboMlxError as exc:
                self._write_expected_request_error(exc)
            except Exception as exc:
                self._write_unexpected_request_error(exc)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._validate_api_request(require_same_origin=True)
            except Krea2TurboMlxError as exc:
                state.record_system_event(
                    "request",
                    str(exc),
                    details={"error_type": type(exc).__name__},
                )
                self._write_json({"ok": False, "message": str(exc)}, status=403)
                return

            try:
                if parsed.path.startswith("/api/image/"):
                    image_id = _image_id_from_path(parsed.path)
                    accepted, message = state.delete_image(image_id)
                    self._write_json(
                        {"ok": accepted, "message": message},
                        status=200 if accepted else 404,
                    )
                else:
                    self.send_error(404)
            except Krea2TurboMlxError as exc:
                self._write_expected_request_error(exc)
            except Exception as exc:
                self._write_unexpected_request_error(exc)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self, *, max_bytes: int = MAX_REQUEST_BYTES) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise Krea2TurboMlxError("Invalid Content-Length header.") from exc
            if length > max_bytes:
                raise Krea2TurboMlxError("Request is too large.")
            if length <= 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Krea2TurboMlxError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise Krea2TurboMlxError("Request body must be a JSON object.")
            return payload

        def _validate_api_request(self, *, require_same_origin: bool) -> None:
            validate_local_request(
                headers=self.headers,
                path=self.path,
                expected_token=session_token,
                allow_unsafe_host=allow_unsafe_host,
                require_same_origin=require_same_origin,
            )

        def _validate_static_request(self) -> None:
            validate_loopback_request_host(
                self.headers,
                allow_unsafe_host=allow_unsafe_host,
            )

        def _request_has_valid_token(self) -> bool:
            return request_has_valid_session_token(
                headers=self.headers,
                path=self.path,
                expected_token=session_token,
            )

        def _write_expected_request_error(self, exc: Krea2TurboMlxError) -> None:
            state.record_system_event(
                "request",
                str(exc),
                details={"error_type": type(exc).__name__},
                kind="error",
            )
            self._write_json({"ok": False, "message": str(exc)}, status=400)

        def _write_unexpected_request_error(self, exc: Exception) -> None:
            _log_unexpected_exception(state.stream, exc)
            state.record_system_event(
                "request",
                "Unexpected request error",
                details={"error_type": type(exc).__name__},
                kind="error",
            )
            self._write_json(
                {
                    "ok": False,
                    "message": "An unexpected error occurred. See the Terminal for details.",
                },
                status=500,
            )

        def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_static(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.name == "index.html":
                content_type = "text/html; charset=utf-8"
                status = (
                    state.snapshot()
                    if self._request_has_valid_token()
                    else _preauth_initial_status()
                )
                data = _index_html_with_initial_status(data, status)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            if path.name == "index.html":
                self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_image_by_id(self, path_text: str) -> None:
            try:
                image_id = _image_id_from_path(path_text)
            except ValueError:
                self.send_error(404)
                return
            self._write_image_path(state.image_path_for_id(image_id))

        def _write_image_path(self, path: Path | None) -> None:
            if path is None or not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_preview(self, query: str) -> None:
            try:
                revision = _preview_revision_from_query(query)
            except ValueError:
                self.send_error(404)
                return
            preview = state.current_preview(revision)
            if preview is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", preview.content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(preview.data)))
            self.end_headers()
            self.wfile.write(preview.data)

    return GuiHandler


def _load_config(path: Path) -> SetupConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Krea2TurboMlxError(f"Setup config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Krea2TurboMlxError(f"Invalid setup config JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Krea2TurboMlxError(f"Invalid setup config JSON: {path}")
    return SetupConfig.from_mapping(payload)


def _gui_settings_path_for_config(config_path: Path) -> Path:
    return config_path.expanduser().parent / DEFAULT_GUI_SETTINGS_FILENAME


def _default_gui_settings() -> dict[str, Any]:
    return {
        "theme": "system",
        "generation": {
            "width": DEFAULT_GENERATION_WIDTH,
            "height": DEFAULT_GENERATION_HEIGHT,
            "steps": DEFAULT_GENERATION_STEPS,
            "randomization_locked": False,
        },
        "live_preview": {
            "mode": "off",
            "interval_steps": DEFAULT_PREVIEW_INTERVAL_STEPS,
        },
        "loras": [],
        "simple_batch": {
            "enabled": False,
            "count": MIN_SIMPLE_BATCH_COUNT,
        },
    }


def _copy_gui_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    generation = settings.get("generation")
    live_preview = settings.get("live_preview")
    simple_batch = settings.get("simple_batch")
    return {
        "theme": str(settings.get("theme", "system")),
        "generation": dict(generation) if isinstance(generation, Mapping) else {},
        "live_preview": dict(live_preview) if isinstance(live_preview, Mapping) else {},
        "loras": _lora_settings_to_mapping_list(settings.get("loras")),
        "simple_batch": dict(simple_batch) if isinstance(simple_batch, Mapping) else {},
    }


def _read_gui_settings(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _default_gui_settings()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _default_gui_settings()
    except json.JSONDecodeError as exc:
        raise Krea2TurboMlxError(f"Invalid GUI settings JSON: {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise Krea2TurboMlxError(f"Invalid GUI settings JSON: {path}")

    return _normalize_gui_settings(payload)


def _write_gui_settings(path: Path | None, settings: dict[str, Any]) -> None:
    if path is None:
        return

    write_json(path, settings)


def _normalize_gui_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    defaults = _default_gui_settings()
    generation = payload.get("generation")
    live_preview = payload.get("live_preview")
    simple_batch = payload.get("simple_batch")
    loras = payload.get("loras")

    return {
        "theme": _normalize_gui_theme(payload.get("theme"), defaults["theme"]),
        "generation": {
            "width": _normalize_gui_dimension(
                generation.get("width") if isinstance(generation, Mapping) else None,
                defaults["generation"]["width"],
                "width",
            ),
            "height": _normalize_gui_dimension(
                generation.get("height") if isinstance(generation, Mapping) else None,
                defaults["generation"]["height"],
                "height",
            ),
            "steps": _normalize_gui_steps(
                generation.get("steps") if isinstance(generation, Mapping) else None,
                defaults["generation"]["steps"],
            ),
            "randomization_locked": _normalize_bool(
                (
                    generation.get("randomization_locked")
                    if isinstance(generation, Mapping)
                    else None
                ),
                defaults["generation"]["randomization_locked"],
            ),
        },
        "live_preview": {
            "mode": _normalize_live_preview_mode(
                live_preview.get("mode") if isinstance(live_preview, Mapping) else None,
                defaults["live_preview"]["mode"],
            ),
            "interval_steps": _normalize_preview_interval_steps(
                (
                    live_preview.get("interval_steps")
                    if isinstance(live_preview, Mapping)
                    else None
                ),
                defaults["live_preview"]["interval_steps"],
            ),
        },
        "loras": _normalize_gui_lora_settings(loras, defaults["loras"]),
        "simple_batch": {
            "enabled": _normalize_bool(
                (
                    simple_batch.get("enabled")
                    if isinstance(simple_batch, Mapping)
                    else None
                ),
                defaults["simple_batch"]["enabled"],
            ),
            "count": _normalize_simple_batch_count(
                (
                    simple_batch.get("count")
                    if isinstance(simple_batch, Mapping)
                    else None
                ),
                defaults["simple_batch"]["count"],
            ),
        },
    }


def _merge_gui_settings_patch(
    current: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(patch, Mapping):
        raise Krea2TurboMlxError("GUI settings payload must be a JSON object.")
    _reject_unsupported_keys(patch, _GUI_SETTINGS_KEYS, "GUI settings")

    settings = _normalize_gui_settings(current)

    if "theme" in patch:
        settings["theme"] = _strict_gui_theme(patch["theme"])

    if "generation" in patch:
        generation = patch["generation"]
        if not isinstance(generation, Mapping):
            raise Krea2TurboMlxError("Generation settings must be a JSON object.")
        _reject_unsupported_keys(
            generation,
            _GUI_GENERATION_KEYS,
            "Generation settings",
        )
        if "width" in generation:
            settings["generation"]["width"] = _strict_gui_dimension(
                generation["width"],
                "width",
            )
        if "height" in generation:
            settings["generation"]["height"] = _strict_gui_dimension(
                generation["height"],
                "height",
            )
        if "steps" in generation:
            settings["generation"]["steps"] = _strict_gui_steps(generation["steps"])
        if "randomization_locked" in generation:
            settings["generation"]["randomization_locked"] = _strict_bool(
                generation["randomization_locked"],
                "Randomization lock",
            )

    if "live_preview" in patch:
        live_preview = patch["live_preview"]
        if not isinstance(live_preview, Mapping):
            raise Krea2TurboMlxError("Live preview settings must be a JSON object.")
        _reject_unsupported_keys(
            live_preview,
            _GUI_LIVE_PREVIEW_KEYS,
            "Live preview settings",
        )
        if "mode" in live_preview:
            settings["live_preview"]["mode"] = _strict_live_preview_mode(
                live_preview["mode"]
            )
        if "interval_steps" in live_preview:
            settings["live_preview"]["interval_steps"] = _strict_preview_interval_steps(
                live_preview["interval_steps"]
            )

    if "loras" in patch:
        settings["loras"] = _strict_gui_lora_settings(patch["loras"])

    if "simple_batch" in patch:
        simple_batch = patch["simple_batch"]
        if not isinstance(simple_batch, Mapping):
            raise Krea2TurboMlxError("Simple batch settings must be a JSON object.")
        _reject_unsupported_keys(
            simple_batch,
            _GUI_SIMPLE_BATCH_KEYS,
            "Simple batch settings",
        )
        if "enabled" in simple_batch:
            settings["simple_batch"]["enabled"] = _strict_bool(
                simple_batch["enabled"],
                "Simple batch enabled",
            )
        if "count" in simple_batch:
            settings["simple_batch"]["count"] = _strict_simple_batch_count(
                simple_batch["count"]
            )

    return settings


def _reject_unsupported_keys(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        noun = "field" if len(unsupported) == 1 else "fields"
        raise Krea2TurboMlxError(
            f"{label} has unsupported {noun}: {', '.join(unsupported)}."
        )


def _normalize_gui_theme(value: Any, default: str) -> str:
    theme = str(value if value is not None else "").strip().lower()
    return theme if theme in _GUI_THEME_MODES else default


def _strict_gui_theme(value: Any) -> str:
    theme = _normalize_gui_theme(value, "")
    if not theme:
        raise Krea2TurboMlxError("Theme must be one of system, light, or dark.")
    return theme


def _normalize_gui_dimension(value: Any, default: int, label: str) -> int:
    try:
        parsed = validate_generation_dimension(default if value is None else value, label)
    except Krea2TurboMlxError:
        return default
    return parsed


def _strict_gui_dimension(value: Any, label: str) -> int:
    return validate_generation_dimension(value, label)


def _normalize_gui_steps(value: Any, default: int) -> int:
    try:
        parsed = _parse_gui_int(default if value is None else value, "steps")
    except Krea2TurboMlxError:
        return default
    return parsed if parsed > 0 else default


def _strict_gui_steps(value: Any) -> int:
    parsed = _parse_gui_int(value, "steps")
    if parsed <= 0:
        raise Krea2TurboMlxError("Steps must be a positive integer.")
    return parsed


def _normalize_live_preview_mode(value: Any, default: str) -> str:
    try:
        return validate_live_preview_mode(
            str(value if value is not None else default).strip().lower()
        )
    except ValueError:
        return default


def _strict_live_preview_mode(value: Any) -> str:
    try:
        return validate_live_preview_mode(str(value).strip().lower())
    except ValueError as exc:
        raise Krea2TurboMlxError(
            "Live preview must be one of off, latent, or vae."
        ) from exc


def _normalize_preview_interval_steps(value: Any, default: int) -> int:
    try:
        return validate_preview_interval_steps(default if value is None else value)
    except ValueError:
        return default


def _strict_preview_interval_steps(value: Any) -> int:
    try:
        return validate_preview_interval_steps(value)
    except ValueError as exc:
        raise Krea2TurboMlxError(
            f"Preview interval must be from 1 to {MAX_PREVIEW_INTERVAL_STEPS}."
        ) from exc


def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise Krea2TurboMlxError(f"{label} must be true or false.")


def _normalize_gui_lora_settings(value: Any, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        return _lora_settings_to_mapping_list(
            [reference.to_mapping() for reference in normalize_lora_payload(value, clamp_scale=True)]
        )
    except Krea2TurboMlxError:
        return [dict(item) for item in default]


def _strict_gui_lora_settings(value: Any) -> list[dict[str, Any]]:
    return _lora_settings_to_mapping_list(
        [reference.to_mapping() for reference in normalize_lora_payload(value, clamp_scale=False)]
    )


def _lora_settings_to_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    loras: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        lora_id = str(item.get("id") or "").strip()
        if not lora_id:
            continue
        try:
            scale = float(item.get("scale", LOCAL_LORA_DEFAULT_SCALE))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(scale):
            continue
        loras.append({"id": lora_id, "scale": scale})
    return loras


def _normalize_simple_batch_count(value: Any, default: int) -> int:
    try:
        return _strict_simple_batch_count(default if value is None else value)
    except Krea2TurboMlxError:
        return default


def _strict_simple_batch_count(value: Any) -> int:
    parsed = _parse_gui_int(value, "simple batch count")
    if parsed < MIN_SIMPLE_BATCH_COUNT or parsed > MAX_GUI_BATCH_JOBS:
        raise Krea2TurboMlxError(
            f"Simple batch count must be from {MIN_SIMPLE_BATCH_COUNT} to "
            f"{MAX_GUI_BATCH_JOBS}."
        )
    return parsed


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _load_pipeline(model: Path, progress_callback: Any) -> Any:
    from .pipeline import KreaTurboPipeline

    return KreaTurboPipeline.from_artifact(model, progress_callback=progress_callback)


def _generation_request_from_payload(payload: dict[str, Any]) -> _GenerationRequest:
    request = _generation_request_from_job(payload, allow_random_seed=True)
    return replace(
        request,
        loras=_lora_references_from_payload(payload),
        live_preview=_live_preview_from_payload(payload),
        preview_interval_steps=_preview_interval_from_payload(payload),
    )


def _batch_jobs_from_payload(payload: dict[str, Any]) -> list[_GenerationRequest]:
    if not isinstance(payload, dict):
        raise Krea2TurboMlxError("Batch payload must be a JSON object.")
    if "loras" in payload:
        raise Krea2TurboMlxError(
            "Batch LoRAs must be specified inside each job, not at the top level."
        )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise Krea2TurboMlxError("Batch payload must include a jobs array.")
    if not jobs:
        raise Krea2TurboMlxError("Batch must include at least one job.")
    if len(jobs) > MAX_GUI_BATCH_JOBS:
        raise Krea2TurboMlxError(
            f"Batch cannot include more than {MAX_GUI_BATCH_JOBS} jobs."
        )

    live_preview = _live_preview_from_payload(payload)
    preview_interval_steps = _preview_interval_from_payload(payload)
    requests: list[_GenerationRequest] = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise Krea2TurboMlxError(f"Job {index} must be a JSON object.")
        unsupported = sorted(set(job) - _BATCH_JOB_KEYS)
        if unsupported:
            label = "field" if len(unsupported) == 1 else "fields"
            raise Krea2TurboMlxError(
                f"Job {index} has unsupported {label}: {', '.join(unsupported)}."
            )
        try:
            requests.append(
                replace(
                    _generation_request_from_job(job, allow_random_seed=True),
                    loras=_lora_references_from_payload(job),
                    live_preview=live_preview,
                    preview_interval_steps=preview_interval_steps,
                )
            )
        except Krea2TurboMlxError as exc:
            raise Krea2TurboMlxError(f"Job {index}: {exc}") from exc
    return requests


def _batch_output_dir_from_payload(payload: dict[str, Any]) -> Path | None:
    output_dir = payload.get("output_dir")
    if output_dir is None:
        return None

    output_dir_text = str(output_dir).strip()
    if output_dir_text in {"", "."}:
        raise Krea2TurboMlxError("Batch output directory must be a real path.")
    return Path(output_dir_text).expanduser()


def _lora_references_from_payload(payload: dict[str, Any]) -> tuple[LoraReference, ...]:
    return normalize_lora_payload(payload.get("loras"), clamp_scale=True)


def _live_preview_from_payload(payload: dict[str, Any]) -> str:
    mode = str(payload.get("live_preview", "off")).strip().lower()
    try:
        return validate_live_preview_mode(mode)
    except ValueError as exc:
        raise Krea2TurboMlxError(
            "Live preview must be one of off, latent, or vae."
        ) from exc


def _preview_interval_from_payload(payload: dict[str, Any]) -> int:
    value = payload.get("preview_interval_steps", DEFAULT_PREVIEW_INTERVAL_STEPS)
    try:
        return validate_preview_interval_steps(value)
    except ValueError as exc:
        raise Krea2TurboMlxError(
            f"Preview interval must be an integer from 1 to {MAX_PREVIEW_INTERVAL_STEPS}."
        ) from exc


def _resolve_request_loras(
    pipeline: Any,
    loras: tuple[LoraReference, ...],
    *,
    lora_dir: Path = DEFAULT_LORA_DIR,
) -> tuple[Any, ...]:
    if not loras:
        return ()
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise Krea2TurboMlxError("LoRA generation requires a loaded transformer.")
    return resolve_lora_patches(
        loras,
        transformer=transformer,
        lora_dir=lora_dir,
    )


def _prepare_output_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Krea2TurboMlxError(f"Could not create output directory: {path}") from exc
    if not path.is_dir():
        raise Krea2TurboMlxError(f"Output path must be a directory: {path}")
    return path


def _output_dir_status_path(path: Path) -> str:
    try:
        expanded = path.expanduser()
        if expanded.is_absolute():
            return str(expanded.resolve())
        return str((Path.cwd() / expanded).resolve())
    except OSError:
        return str(path)


def _copy_lora_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    items = catalog.get("items")
    warnings = catalog.get("warnings")
    return {
        "dir": str(catalog.get("dir", DEFAULT_LORA_DIR)),
        "items": [dict(item) for item in items] if isinstance(items, list) else [],
        "warnings": list(warnings) if isinstance(warnings, list) else [],
        "scanned_at_ms": catalog.get("scanned_at_ms"),
    }


def _source_image_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    path_text = str(payload.get("path") or "").strip()
    image_base64 = str(payload.get("image_base64") or "").strip()

    if path_text:
        return _source_image_metadata_from_path(_source_path_from_text(path_text))
    if image_base64:
        filename = str(payload.get("filename") or "Pasted image").strip()
        data = _decode_source_image_base64(image_base64)
        return _source_image_metadata_from_bytes(data, source=filename or "Pasted image")

    raise Krea2TurboMlxError("Choose an image file or enter an image path.")


def _source_image_metadata_from_path(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    try:
        stat = expanded.stat()
    except OSError as exc:
        raise Krea2TurboMlxError(f"Source image is not readable: {expanded}") from exc
    if not expanded.is_file():
        raise Krea2TurboMlxError(f"Source image path must be a file: {expanded}")
    if stat.st_size > MAX_SOURCE_IMAGE_BYTES:
        raise Krea2TurboMlxError(
            f"Source image must be {_format_mb(MAX_SOURCE_IMAGE_BYTES)} or smaller."
        )
    return _source_image_metadata_from_image(expanded, source=str(expanded))


def _source_image_metadata_from_bytes(data: bytes, *, source: str) -> dict[str, Any]:
    if not data:
        raise Krea2TurboMlxError("Source image is empty.")
    if len(data) > MAX_SOURCE_IMAGE_BYTES:
        raise Krea2TurboMlxError(
            f"Source image must be {_format_mb(MAX_SOURCE_IMAGE_BYTES)} or smaller."
        )
    return _source_image_metadata_from_image(BytesIO(data), source=source)


def _source_image_metadata_from_image(image_source: Any, *, source: str) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise Krea2TurboMlxError(
            "Source image metadata import requires Pillow. Run `./setup.sh`."
        ) from exc

    try:
        with Image.open(image_source) as image:
            image_format = str(image.format or "").lower()
            image_size = tuple(int(item) for item in image.size)
            image_info = dict(image.info)
            exif = dict(image.getexif())
    except Exception as exc:
        raise Krea2TurboMlxError("Source image must be a readable image file.") from exc

    project_metadata = _project_metadata_from_image_info(image_info)
    settings, supported = _source_settings_from_metadata(project_metadata)
    other = _other_source_metadata_entries(
        project_metadata,
        image_info=image_info,
        exif=exif,
    )

    return {
        "source": source,
        "format": image_format,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "settings": settings,
        "supported": supported,
        "other": other,
    }


def _project_metadata_from_image_info(image_info: dict[str, Any]) -> dict[str, Any]:
    metadata_text = image_info.get(PNG_METADATA_KEY)
    if not isinstance(metadata_text, str) or not metadata_text.strip():
        raise Krea2TurboMlxError(
            f"Source image is missing {PNG_METADATA_KEY} metadata."
        )
    try:
        payload = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise Krea2TurboMlxError(
            f"Source image has invalid {PNG_METADATA_KEY} metadata."
        ) from exc
    if not isinstance(payload, dict):
        raise Krea2TurboMlxError(
            f"Source image {PNG_METADATA_KEY} metadata must be a JSON object."
        )
    return payload


def _source_settings_from_metadata(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings: dict[str, Any] = {}
    supported: list[dict[str, Any]] = []

    for key in _SOURCE_IMAGE_SETTING_KEYS:
        if key == "loras":
            continue
        if key not in metadata:
            continue
        value = _source_setting_value(key, metadata[key])
        settings[key] = value
        supported.append(_metadata_entry(key, value))

    if not settings:
        raise Krea2TurboMlxError("Source image has no supported generation settings.")

    loras = _source_lora_settings(metadata.get("loras"))
    settings["loras"] = loras
    supported.append(_metadata_entry("loras", loras))
    return settings, supported


def _source_setting_value(key: str, value: Any) -> Any:
    if key == "prompt":
        prompt = str(value).strip()
        if not prompt:
            raise Krea2TurboMlxError("Source image prompt metadata is empty.")
        return prompt
    if key in {"width", "height"}:
        return validate_generation_dimension(value, f"source image {key}")
    if key == "steps":
        parsed = _parse_gui_int(value, "steps")
        if parsed <= 0:
            raise Krea2TurboMlxError("Source image steps must be a positive integer.")
        return parsed
    if key == "seed":
        parsed = _parse_gui_int(value, "seed")
        if parsed < 0 or parsed > MAX_GENERATION_SEED:
            raise Krea2TurboMlxError(
                f"Source image seed must be an integer from 0 to {MAX_GENERATION_SEED}."
            )
        return parsed
    raise Krea2TurboMlxError(f"Unsupported source image setting: {key}.")


def _source_lora_settings(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise Krea2TurboMlxError("Source image loras metadata must be an array.")

    normalized_payload: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise Krea2TurboMlxError("Source image LoRA metadata must be objects.")
        normalized_payload.append(
            {
                "id": item.get("id") or item.get("display_name"),
                "scale": item.get("scale", LOCAL_LORA_DEFAULT_SCALE),
            }
        )
    return [
        reference.to_mapping()
        for reference in normalize_lora_payload(normalized_payload, clamp_scale=True)
    ]


def _other_source_metadata_entries(
    project_metadata: dict[str, Any],
    *,
    image_info: dict[str, Any],
    exif: dict[Any, Any],
) -> list[dict[str, Any]]:
    entries = [
        _metadata_entry(key, value)
        for key, value in sorted(project_metadata.items())
        if key not in _SOURCE_IMAGE_SETTING_KEYS
    ]

    for key, value in sorted(image_info.items(), key=lambda item: str(item[0])):
        if key == PNG_METADATA_KEY:
            continue
        entries.append(_metadata_entry(str(key), value))

    for key, value in sorted(exif.items(), key=lambda item: str(item[0])):
        entries.append(_metadata_entry(f"exif.{key}", value))

    return entries


def _metadata_entry(key: str, value: Any) -> dict[str, Any]:
    return {
        "key": key,
        "label": _source_metadata_label(key),
        "value": _metadata_display_value(value),
    }


def _source_metadata_label(key: str) -> str:
    labels = {
        "prompt": "Prompt",
        "width": "Width",
        "height": "Height",
        "steps": "Steps",
        "seed": "Seed",
        "loras": "LoRAs",
        "model_path": "Model path",
        "model_precision": "Model precision",
        "guidance_scale": "Guidance scale",
        "shift": "Shift",
        PNG_PARAMETERS_KEY: "Parameters",
    }
    return labels.get(key, key.replace("_", " ").capitalize())


def _metadata_display_value(value: Any) -> Any:
    if isinstance(value, bytes | bytearray):
        return f"<binary: {len(value)} bytes>"
    if isinstance(value, dict):
        return {str(key): _metadata_display_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_metadata_display_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _decode_source_image_base64(value: str) -> bytes:
    text = value.strip()
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise Krea2TurboMlxError("Source image upload is not valid base64.") from exc


def _source_path_from_text(text: str) -> Path:
    stripped = text.strip()
    if stripped.startswith("file://"):
        return Path(unquote(urlparse(stripped).path)).expanduser()
    return Path(stripped).expanduser()


def _read_source_image_clipboard() -> dict[str, Any]:
    file_path = _clipboard_file_path()
    if file_path is not None:
        return _source_image_metadata_from_path(file_path)

    text = _read_clipboard_text_raw(
        max_bytes=8_192,
        too_large_message="Clipboard image path is too long.",
        decode_error_message="Clipboard image path must be UTF-8 text.",
    ).strip()
    if not text:
        raise Krea2TurboMlxError("Clipboard image path is empty.")
    return _source_image_metadata_from_path(_source_path_from_text(text))


def _read_batch_clipboard() -> tuple[str, str]:
    file_path = _clipboard_file_path()
    if file_path is not None:
        return _read_clipboard_json_file(file_path), file_path.name

    text = _read_clipboard_text()
    _validate_clipboard_json_text(text)
    return text, "Clipboard"


def _clipboard_file_path() -> Path | None:
    if sys.platform != "darwin":
        return None

    file_url_class = (
        "\\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}"
        "class furl"
        "\\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}"
    ).encode("utf-8").decode("unicode_escape")
    script = "\n".join(
        (
            "try",
            f"POSIX path of (the clipboard as {file_url_class})",
            "on error",
            'return ""',
            "end try",
        )
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    text = result.stdout.strip()
    return Path(text).expanduser() if text else None


def _read_clipboard_json_file(path: Path) -> str:
    if path.suffix.lower() != ".json":
        raise Krea2TurboMlxError("Clipboard file must be a .json file.")
    try:
        stat = path.stat()
    except OSError as exc:
        raise Krea2TurboMlxError(f"Clipboard file is not readable: {path}") from exc
    if not path.is_file():
        raise Krea2TurboMlxError(f"Clipboard path must be a file: {path}")
    if stat.st_size > MAX_GUI_BATCH_BYTES:
        raise Krea2TurboMlxError(
            f"Batch JSON must be {_format_kb(MAX_GUI_BATCH_BYTES)} or smaller."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise Krea2TurboMlxError("Clipboard JSON file must be UTF-8 text.") from exc
    except OSError as exc:
        raise Krea2TurboMlxError(f"Clipboard file is not readable: {path}") from exc
    _validate_clipboard_json_text(text)
    return text


def _read_clipboard_text() -> str:
    return _read_clipboard_text_raw(
        max_bytes=MAX_GUI_BATCH_BYTES,
        too_large_message=(
            f"Batch JSON must be {_format_kb(MAX_GUI_BATCH_BYTES)} or smaller."
        ),
        decode_error_message="Clipboard JSON must be UTF-8 text.",
    )


def _read_clipboard_text_raw(
    *,
    max_bytes: int,
    too_large_message: str,
    decode_error_message: str,
) -> str:
    if sys.platform != "darwin":
        raise Krea2TurboMlxError("Clipboard paste requires macOS.")

    try:
        process = subprocess.Popen(
            ["pbpaste"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise Krea2TurboMlxError("Clipboard text is not available.") from exc

    assert process.stdout is not None
    data = process.stdout.read(max_bytes + 1)
    if len(data) > max_bytes:
        process.kill()
        process.communicate()
        raise Krea2TurboMlxError(too_large_message)

    _, stderr = process.communicate(timeout=2)
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise Krea2TurboMlxError(
            "Clipboard text is not available" + (f": {message}" if message else ".")
        )

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Krea2TurboMlxError(decode_error_message) from exc


def _validate_clipboard_json_text(text: str) -> None:
    stripped = text.strip()
    if not stripped:
        raise Krea2TurboMlxError("Clipboard JSON is empty.")
    if not stripped.startswith("["):
        raise Krea2TurboMlxError("Clipboard JSON must start with [.")


def _format_kb(bytes_count: int) -> str:
    return f"{round(bytes_count / 1024)} KB"


def _format_mb(bytes_count: int) -> str:
    return f"{round(bytes_count / (1024 * 1024))} MB"


def _generation_request_from_job(
    job: dict[str, Any],
    *,
    allow_random_seed: bool,
) -> _GenerationRequest:
    prompt = str(job.get("prompt", "")).strip()
    width, height = validate_generation_dimensions(job.get("width"), job.get("height"))
    steps = _parse_gui_int(job.get("steps"), "steps")
    if allow_random_seed and "seed" not in job:
        seed = secrets.randbelow(MAX_GENERATION_SEED + 1)
    else:
        seed = _parse_gui_int(job.get("seed"), "seed")
    if not prompt:
        raise Krea2TurboMlxError("Prompt cannot be empty.")
    if steps <= 0:
        raise Krea2TurboMlxError("Steps must be a positive integer.")
    if seed < 0 or seed > MAX_GENERATION_SEED:
        raise Krea2TurboMlxError(
            f"Seed must be an integer from 0 to {MAX_GENERATION_SEED}."
        )
    return _GenerationRequest(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
    )


def _parse_gui_int(value: Any, label: str) -> int:
    try:
        if isinstance(value, str):
            value = value.strip()
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Krea2TurboMlxError(f"{label.capitalize()} must be an integer.") from exc


def _preview_jpeg_from_frame(frame: PipelinePreviewFrame) -> tuple[bytes, int, int]:
    target_width = _validate_preview_dimension(frame.width, "preview width")
    target_height = _validate_preview_dimension(frame.height, "preview height")

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise Krea2TurboMlxError("Live preview requires numpy and Pillow.") from exc

    array = np.asarray(frame.image)
    if array.ndim == 4:
        if array.shape[0] < 1:
            raise Krea2TurboMlxError("Live preview image batch is empty.")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise Krea2TurboMlxError("Live preview image must be shaped [H, W, 3].")

    if array.dtype == np.uint8:
        image_u8 = array
    else:
        image_float = np.nan_to_num(
            array.astype(np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        image_u8 = (np.clip(image_float, 0.0, 1.0) * 255.0).round().astype("uint8")

    image = Image.fromarray(image_u8).convert("RGB")
    if image.size != (target_width, target_height):
        image = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=86)
    return buffer.getvalue(), target_width, target_height


def _validate_preview_dimension(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Krea2TurboMlxError(f"{label.capitalize()} must be an integer.") from exc
    if parsed <= 0:
        raise Krea2TurboMlxError(f"{label.capitalize()} must be positive.")
    if parsed > MAX_GUI_SIZE:
        raise Krea2TurboMlxError(
            f"{label.capitalize()} must be {MAX_GUI_SIZE} or smaller."
        )
    return parsed


def _preview_revision_from_query(query: str) -> int | None:
    values = parse_qs(query).get("rev")
    if not values:
        return None
    return int(values[0])


def _image_id_from_path(path_text: str) -> int:
    return int(path_text.rsplit("/", 1)[1])


def _generation_path_key(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _generation_file_entries(output_dir: Path) -> list[tuple[Path, int, int]]:
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    entries: list[tuple[Path, int, int]] = []
    for path in output_dir.glob("*.png"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path, stat.st_mtime_ns, stat.st_size))
    entries.sort(key=lambda item: item[1], reverse=True)
    return entries


def _generation_signature(
    entries: list[tuple[Path, int, int]],
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(_generation_path_key(path)), mtime_ns, size)
        for path, mtime_ns, size in entries[:MAX_INITIAL_RECENT_GENERATIONS]
    )


def _record_from_png(
    path: Path,
    *,
    record_id: int,
    default_model: str,
) -> _GenerationRecord | None:
    try:
        metadata, image_size = _read_png_details(path)
        stat = path.stat()
    except OSError:
        return None
    created_at = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    width = _optional_int(metadata.get("width"))
    height = _optional_int(metadata.get("height"))
    if image_size is not None:
        width = width or image_size[0]
        height = height or image_size[1]
    if width is None or height is None:
        return None
    seed = _optional_int(metadata.get("seed"))
    steps = _optional_int(metadata.get("steps")) or DEFAULT_GENERATION_STEPS
    filename_parts = _details_from_filename(path.name)
    if seed is None:
        seed = filename_parts.get("seed")
    if seed is None:
        seed = 0
    if "steps" in filename_parts and not metadata.get("steps"):
        steps = int(filename_parts["steps"])
    model_path = str(metadata.get("model_path") or default_model)
    return _GenerationRecord(
        id=record_id,
        path=path,
        prompt=str(metadata.get("prompt") or ""),
        seed=int(seed),
        width=int(width),
        height=int(height),
        steps=int(steps),
        model_path=model_path,
        model_precision=str(
            metadata.get("model_precision") or model_precision_label(model_path)
        ),
        created_at=created_at,
        loras=_generation_record_loras(metadata.get("loras")),
    )


def _generation_record_loras(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()

    loras: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        lora_id = str(item.get("id") or "").strip()
        display_name = str(item.get("display_name") or "").strip()
        try:
            scale = float(item.get("scale", LOCAL_LORA_DEFAULT_SCALE))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(scale) or not (lora_id or display_name):
            continue

        payload: dict[str, Any] = {"scale": scale}
        for key in (
            "id",
            "display_name",
            "source_type",
            "adapter_type",
            "path",
            "target_count",
            "skipped_count",
            "warnings",
            "sha256",
            "patch_hash",
        ):
            value = item.get(key)
            if value not in (None, ""):
                payload[key] = value
        if lora_id and "id" not in payload:
            payload["id"] = lora_id
        if display_name and "display_name" not in payload:
            payload["display_name"] = display_name
        loras.append(payload)

    return tuple(loras)


def _read_png_details(path: Path) -> tuple[dict[str, Any], tuple[int, int] | None]:
    try:
        from PIL import Image
    except ImportError:
        return {}, None
    with Image.open(path) as image:
        metadata_text = image.info.get(PNG_METADATA_KEY)
        metadata = {}
        if isinstance(metadata_text, str):
            try:
                payload = json.loads(metadata_text)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                metadata = payload
        return metadata, tuple(int(item) for item in image.size)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _details_from_filename(filename: str) -> dict[str, int]:
    details: dict[str, int] = {}
    for part in filename.replace(".", "-").split("-"):
        if part.startswith("seed"):
            seed = _optional_int(part[4:])
            if seed is not None:
                details["seed"] = seed
        elif part.startswith("steps"):
            steps = _optional_int(part[5:])
            if steps is not None:
                details["steps"] = steps
    return details


def _static_path_from_url(static_root: Path, url_path: str) -> Path:
    root = static_root.resolve()
    clean = unquote(url_path.split("?", 1)[0])
    relative = clean.lstrip("/") or "index.html"
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Krea2TurboMlxError("Static asset path escapes the GUI build.") from exc
    if candidate.is_dir():
        candidate = candidate / "index.html"
    if not candidate.exists() and "." not in Path(relative).name:
        candidate = root / "index.html"
    return candidate


def _index_html_with_initial_status(
    html: bytes,
    status: Mapping[str, Any],
) -> bytes:
    marker = b"</head>"
    script = (
        "<script>window."
        f"{_INITIAL_STATUS_GLOBAL}="
        f"{_script_json(status)};</script>"
    ).encode("utf-8")

    if marker not in html:
        return script + html

    return html.replace(marker, script + marker, 1)


def _preauth_initial_status() -> dict[str, Any]:
    return {
        "server": {"connected": True, "status": "auth_required"},
        "model": {
            "loaded": False,
            "name": PROJECT_NAME,
            "status": "locked",
            "path": "",
            "precision": "",
        },
        "busy": False,
        "load_running": False,
        "generation_running": False,
        "cancel_requested": False,
        "batch": None,
        "phase": "auth_required",
        "message": "Open the local GUI from the authenticated Terminal URL.",
        "progress": 0.0,
        "error": None,
        "image": None,
        "preview": None,
        "recent": [],
        "output_dir": {"path": ""},
        "loras": {"dir": "", "items": [], "warnings": [], "scanned_at_ms": None},
        "ui_settings": _default_gui_settings(),
        "constraints": {},
        "events": [],
        "task": {"name": None, "started_ms": None, "completed_ms": None},
    }


def _script_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(payload, sort_keys=True)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _choose_output_dir(initial_dir: Path) -> Path | None:
    if sys.platform != "darwin":
        raise Krea2TurboMlxError("Output directory selection requires macOS.")

    default_dir = _dialog_default_dir(initial_dir)
    script = (
        'POSIX path of (choose folder with prompt "Select output directory" '
        f"default location POSIX file {_applescript_string(str(default_dir))})"
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "User canceled" in stderr:
            return None
        raise Krea2TurboMlxError(
            "Could not select output directory"
            + (f": {stderr}" if stderr else ".")
        )

    selected = result.stdout.strip()
    if not selected:
        return None
    return Path(selected).expanduser()


def _dialog_default_dir(path: Path) -> Path:
    try:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.is_dir():
            return candidate
        for parent in candidate.parents:
            if parent.is_dir():
                return parent
    except OSError:
        pass
    return Path.home()


def _applescript_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _frontend_static_root() -> Path:
    candidates = [
        Path.cwd() / "frontend" / "build" / "client",
        Path(__file__).resolve().parents[2] / "frontend" / "build" / "client",
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return candidates[0]


def _validate_static_root(static_root: Path) -> None:
    if not (static_root / "index.html").is_file():
        raise Krea2TurboMlxError(
            "React GUI build is missing: "
            f"{static_root}. Use a release checkout that includes the GUI build, "
            "or set KREA_2_TURBO_MLX_BUILD_FRONTEND=1 when running `./setup.sh` "
            "to rebuild it."
        )


def _url_host(host: str) -> str:
    text = str(host)
    if ":" in text and not text.startswith("["):
        return f"[{text}]"
    return text


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        **_event_time_payload(),
        "kind": str(getattr(event, "kind", "event")),
        "stage": str(getattr(event, "stage", "event")),
        "message": str(getattr(event, "message", "")),
        "progress": getattr(event, "progress", None),
        "step_index": getattr(event, "step_index", None),
        "step_count": getattr(event, "step_count", None),
        "details": _safe_event_details(getattr(event, "details", {}) or {}),
    }


def _safe_event_details(value: Any) -> Any:
    return redact_session_tokens(json_safe(value))


def _log_unexpected_exception(stream: TextIO, exc: Exception) -> None:
    _log(stream, f"Unexpected {type(exc).__name__}: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)


def _format_event_for_terminal(payload: dict[str, Any]) -> str:
    progress = payload.get("progress")
    progress_text = "" if progress is None else f" progress={float(progress):.0%}"
    step_text = ""
    if payload.get("step_index") is not None and payload.get("step_count") is not None:
        step_text = f" step={int(payload['step_index']) + 1}/{payload['step_count']}"
    details = payload.get("details") or {}
    detail_text = "" if not details else " " + json.dumps(details, sort_keys=True)
    return (
        f"[{payload['time']}] "
        f"{payload['stage']}/{payload['kind']}: {payload['message']}"
        f"{progress_text}{step_text}{detail_text}"
    )


def _print_gui_banner(
    stream: TextIO,
    *,
    url: str,
    model: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    lines = [
        "Krea 2 Turbo GUI",
        "",
        f"Browser: {url}",
        f"Model:   {model}",
        f"Config:  {config_path}",
        f"Outputs: {output_dir}",
        "",
        "Keep this Terminal window open while using the browser GUI.",
        "Close this window or press Ctrl-C to stop the local process.",
    ]
    width = max(len(line) for line in lines) + 4
    border = "+" + "-" * (width - 2) + "+"
    stream.write("\n")
    stream.write(border + "\n")
    for line in lines:
        stream.write("| " + line.ljust(width - 4) + " |\n")
    stream.write(border + "\n\n")
    stream.flush()


def _clear_runtime_caches() -> None:
    from .pipeline import clear_runtime_caches

    clear_runtime_caches()


def _open_browser(url: str, stream: TextIO) -> None:
    webbrowser.open(url)
    _log(stream, "Browser requested. If it does not open, paste the URL above.")


def _timestamp_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _event_time_payload(time_ms: int | None = None) -> dict[str, Any]:
    if time_ms is None:
        now = datetime.now()
        time_ms = int(now.timestamp() * 1000)
    else:
        now = datetime.fromtimestamp(time_ms / 1000)
    return {
        "time": now.strftime("%H:%M:%S"),
        "time_ms": time_ms,
    }


def _clamp_progress(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _generation_request_snapshot(request: _GenerationRequest) -> dict[str, Any]:
    return {
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "seed": request.seed,
        "loras": [lora.to_mapping() for lora in request.loras],
    }


def _batch_overall_progress(index: int, total: int, current_progress: float) -> float:
    if total <= 0 or index <= 0:
        return 0.0
    completed_jobs = max(0, min(total, int(index) - 1))
    return _clamp_progress(
        (completed_jobs + _clamp_progress(current_progress)) / int(total)
    )


def _batch_task_name(total: int) -> str:
    suffix = "job" if int(total) == 1 else "jobs"
    return f"Generate batch ({int(total)} {suffix})"


def _log(stream: TextIO, message: str) -> None:
    stream.write(message + "\n")
    stream.flush()


__all__ = ["run_gui"]
