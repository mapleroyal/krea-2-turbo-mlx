from __future__ import annotations

import base64
import json
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from krea_2_turbo_mlx import cli
import krea_2_turbo_mlx.gui as gui
from krea_2_turbo_mlx.errors import Krea2TurboMlxError
from krea_2_turbo_mlx.pipeline import PipelinePreviewFrame, PipelineProgressEvent
from krea_2_turbo_mlx.png import (
    PNG_METADATA_KEY,
    PNG_PARAMETERS_KEY,
    generation_metadata_payload,
    save_generation_png,
)
from krea_2_turbo_mlx.setup_flow import SetupConfig
from safetensors_fixtures import write_safetensors_fixture

Image = pytest.importorskip("PIL.Image")


def test_gui_batch_validation_accepts_jobs_and_randomizes_missing_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "jobs": [
            {
                "prompt": "a glass observatory",
                "width": 16,
                "height": 16,
                "steps": 4,
                "seed": 7,
            },
            {
                "prompt": "a cedar observatory",
                "width": 32,
                "height": 16,
                "steps": 2,
            },
        ]
    }

    monkeypatch.setattr(gui.secrets, "randbelow", lambda limit: 123456)
    requests = gui._batch_jobs_from_payload(payload)

    assert [request.prompt for request in requests] == [
        "a glass observatory",
        "a cedar observatory",
    ]
    assert [request.seed for request in requests] == [7, 123456]


def test_gui_lora_payload_defaults_and_resolves_per_batch_job() -> None:
    single = gui._generation_request_from_payload(
        {
            "prompt": "a glass observatory",
            "width": 16,
            "height": 16,
            "steps": 4,
            "seed": 7,
            "loras": [{"id": "style.safetensors"}],
        }
    )

    assert single.loras[0].id == "style.safetensors"
    assert single.loras[0].scale == 1.0

    batch = gui._batch_jobs_from_payload(
        {
            "jobs": [
                {
                    "prompt": "one",
                    "width": 16,
                    "height": 16,
                    "steps": 1,
                    "seed": 1,
                    "loras": [{"id": "style.safetensors", "scale": 2}],
                },
                {
                    "prompt": "two",
                    "width": 16,
                    "height": 16,
                    "steps": 1,
                    "seed": 2,
                    "loras": [],
                },
                {
                    "prompt": "three",
                    "width": 16,
                    "height": 16,
                    "steps": 1,
                    "seed": 3,
                },
            ],
        }
    )

    assert [tuple(lora.scale for lora in request.loras) for request in batch] == [
        (2.0,),
        (),
        (),
    ]


def test_gui_live_preview_payload_applies_to_single_and_batch() -> None:
    single = gui._generation_request_from_payload(
        {
            "prompt": "a glass observatory",
            "width": 16,
            "height": 16,
            "steps": 4,
            "seed": 7,
            "live_preview": "Latent",
            "preview_interval_steps": "3",
        }
    )

    assert single.live_preview == "latent"
    assert single.preview_interval_steps == 3

    batch = gui._batch_jobs_from_payload(
        {
            "jobs": [
                {
                    "prompt": "one",
                    "width": 16,
                    "height": 16,
                    "steps": 1,
                    "seed": 1,
                },
                {
                    "prompt": "two",
                    "width": 16,
                    "height": 16,
                    "steps": 1,
                    "seed": 2,
                },
            ],
            "live_preview": "vae",
            "preview_interval_steps": 5,
        }
    )

    assert [request.live_preview for request in batch] == ["vae", "vae"]
    assert [request.preview_interval_steps for request in batch] == [5, 5]


def test_gui_ui_settings_persist_to_project_settings_file(tmp_path: Path) -> None:
    settings_path = tmp_path / ".krea-2-turbo-mlx" / "gui-settings.json"
    state = _state(
        tmp_path / "outputs",
        _FakePipeline(),
        settings_path=settings_path,
    )

    settings = state.update_ui_settings(
        {
            "theme": "dark",
            "generation": {
                "width": "512",
                "height": 768,
                "steps": "10",
                "randomization_locked": True,
            },
            "live_preview": {
                "mode": "Latent",
                "interval_steps": "3",
            },
            "loras": [{"id": "style.safetensors", "scale": 2.5}],
            "simple_batch": {
                "enabled": True,
                "count": "4",
            },
        }
    )

    assert settings == {
        "theme": "dark",
        "generation": {
            "width": 512,
            "height": 768,
            "steps": 10,
            "randomization_locked": True,
        },
        "live_preview": {
            "mode": "latent",
            "interval_steps": 3,
        },
        "loras": [{"id": "style.safetensors", "scale": 2.5}],
        "simple_batch": {
            "enabled": True,
            "count": 4,
        },
    }
    assert json.loads(settings_path.read_text(encoding="utf-8")) == settings

    restored = _state(
        tmp_path / "outputs",
        _FakePipeline(),
        settings_path=settings_path,
    )

    assert restored.snapshot()["ui_settings"] == settings


def test_gui_ui_settings_reject_invalid_generation_params(tmp_path: Path) -> None:
    state = _state(
        tmp_path / "outputs",
        _FakePipeline(),
        settings_path=tmp_path / ".krea-2-turbo-mlx" / "gui-settings.json",
    )

    with pytest.raises(Krea2TurboMlxError, match="multiple of 16"):
        state.update_ui_settings({"generation": {"width": 15}})

    with pytest.raises(Krea2TurboMlxError, match="Randomization lock"):
        state.update_ui_settings({"generation": {"randomization_locked": "yes"}})

    with pytest.raises(Krea2TurboMlxError, match="from 2 to 100"):
        state.update_ui_settings({"simple_batch": {"count": 1}})


def test_gui_index_html_bootstraps_script_safe_initial_status() -> None:
    html = b"<html><head><title>GUI</title></head><body></body></html>"
    status = {
        "ui_settings": {
            "generation": {
                "width": 768,
                "height": 512,
                "steps": 12,
            },
        },
        "recent": [{"prompt": "</script><script>alert(1)</script>"}],
    }

    result = gui._index_html_with_initial_status(html, status)

    assert b"__KREA2_TURBO_MLX_INITIAL_STATUS__" in result
    assert b"\\u003c/script>" in result
    assert result.index(b"__KREA2_TURBO_MLX_INITIAL_STATUS__") < result.index(
        b"</head>"
    )


def test_gui_validation_rejects_invalid_payloads() -> None:
    valid_job = {
        "prompt": "a glass observatory",
        "width": 16,
        "height": 16,
        "steps": 1,
        "seed": 1,
    }
    cases = [
        ([], "JSON object"),
        ({"jobs": {}}, "jobs array"),
        ({"jobs": []}, "at least one job"),
        ({"jobs": ["nope"]}, "Job 1 must be"),
        ({"jobs": [{**valid_job, "surprise": True}]}, "unsupported field"),
        ({"jobs": [{key: value for key, value in valid_job.items() if key != "prompt"}]}, "Prompt cannot"),
        ({"jobs": [{**valid_job, "width": 15}]}, "multiple of 16"),
        ({"jobs": [{**valid_job, "height": gui.MAX_GUI_SIZE + 16}]}, "2048 or smaller"),
        ({"jobs": [{**valid_job, "width": 0}]}, "positive"),
        ({"jobs": [{**valid_job, "steps": 0}]}, "positive integer"),
        ({"jobs": [{**valid_job, "seed": gui.MAX_GENERATION_SEED + 1}]}, "Seed must"),
        ({"jobs": [{**valid_job, "live_preview": "latent"}]}, "unsupported field"),
        (
            {"jobs": [valid_job], "loras": [{"id": "style.safetensors"}]},
            "inside each job",
        ),
        ({"jobs": [valid_job], "live_preview": "nope"}, "Live preview"),
        ({"jobs": [valid_job], "preview_interval_steps": 0}, "Preview interval"),
        ({"jobs": [valid_job], "preview_interval_steps": 2.5}, "Preview interval"),
        ({"jobs": [valid_job] * (gui.MAX_GUI_BATCH_JOBS + 1)}, "more than"),
    ]

    for payload, expected in cases:
        with pytest.raises(Krea2TurboMlxError, match=expected):
            gui._batch_jobs_from_payload(payload)  # type: ignore[arg-type]


def test_gui_load_eject_and_single_flight_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    state = _state(tmp_path, pipeline)
    monkeypatch.setattr(gui, "_clear_runtime_caches", lambda: None)

    accepted, _ = state.start_load()
    assert accepted is True
    _wait_for(state, lambda: state.pipeline is pipeline and not state.load_running)
    snapshot = state.snapshot()
    assert snapshot["model"]["loaded"] is True
    assert snapshot["progress"] == 0.0
    assert snapshot["task"]["name"] == "Load model"
    assert snapshot["events"][-1]["kind"] == "task"
    assert snapshot["events"][-1]["message"] == "Load model"

    accepted, _ = state.start_generation(
        gui._GenerationRequest("one", width=16, height=16, steps=1, seed=1)
    )
    assert accepted is True
    second, message = state.start_generation(
        gui._GenerationRequest("two", width=16, height=16, steps=1, seed=2)
    )
    assert second is False
    assert "already running" in message
    _wait_for_idle(state)

    accepted, _ = state.start_eject()
    assert accepted is True
    snapshot = state.snapshot()
    assert snapshot["model"]["loaded"] is False
    assert snapshot["progress"] == 0.0
    assert snapshot["task"] == {
        "name": None,
        "started_ms": None,
        "completed_ms": None,
    }


def test_gui_status_and_records_use_artifact_precision(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_artifact_precision(artifact, {"BF16": 12, "F32": 2})
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    pipeline = _FakePipeline()
    state = gui._GuiState(
        config=SetupConfig(
            artifact_dir=artifact,
            output_dir=output_dir,
            lora_dir=tmp_path / "loras",
        ),
        stream=StringIO(),
        pipeline_loader=lambda model, progress_callback: pipeline,
    )

    assert state.snapshot()["model"]["precision"] == "bf16"

    accepted, _ = state.start_generation(
        gui._GenerationRequest("one", width=16, height=16, steps=1, seed=1)
    )
    assert accepted is True
    _wait_for_idle(state)

    snapshot = state.snapshot()
    assert snapshot["image"]["model_precision"] == "bf16"
    with Image.open(state.current_image.path) as image:
        metadata = json.loads(image.info[PNG_METADATA_KEY])
    assert metadata["model_precision"] == "bf16"


def test_gui_status_exposes_lora_catalog_and_refresh(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    state = gui._GuiState(
        config=SetupConfig(
            artifact_dir=Path("artifact"),
            output_dir=output_dir,
            lora_dir=lora_dir,
        ),
        stream=StringIO(),
        pipeline_loader=lambda model, progress_callback: _FakePipeline(),
    )

    initial = state.snapshot()["loras"]
    assert initial["dir"] == str(lora_dir)
    assert initial["items"] == []

    write_safetensors_fixture(
        lora_dir / "glass.safetensors",
        {
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": ("F32", [1, 3]),
            "diffusion_model.blocks.0.attn.wq.lora_up.weight": ("F32", [4, 1]),
        },
        payloads={
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": struct.pack(
                "<3f", 1, 0, 0
            ),
            "diffusion_model.blocks.0.attn.wq.lora_up.weight": struct.pack(
                "<4f", 1, 2, 3, 4
            ),
        },
    )

    refreshed = state.refresh_lora_catalog()
    assert [item["id"] for item in refreshed["items"]] == ["glass.safetensors"]
    assert refreshed["items"][0]["adapter_type"] == "standard"
    assert state.snapshot()["loras"]["items"][0]["target_count"] == 1


def test_gui_batch_runs_jobs_in_order_and_records_metadata(tmp_path: Path) -> None:
    pipeline = _FakePipeline()
    state = _state(tmp_path, pipeline)
    requests = [
        gui._GenerationRequest("one", width=16, height=16, steps=2, seed=11),
        gui._GenerationRequest("two", width=32, height=16, steps=3, seed=22),
    ]

    accepted, _ = state.start_batch_generation(requests)
    assert accepted is True
    _wait_for_idle(state)

    assert [call["prompt"] for call in pipeline.calls] == ["one", "two"]
    assert [call["kwargs"]["seed"] for call in pipeline.calls] == [11, 22]
    assert len(list(tmp_path.glob("*.png"))) == 2
    assert [record.prompt for record in state.recent_generations] == ["two", "one"]
    assert state.current_image is not None
    assert state.current_image.prompt == "two"
    snapshot = state.snapshot()
    assert snapshot["batch"] is None
    assert snapshot["progress"] == 0.0
    assert snapshot["events"][-1]["kind"] == "task"
    assert snapshot["events"][-1]["message"] == "Generate batch (2 jobs)"
    assert "elapsed_seconds" in snapshot["events"][-1]["details"]

    with Image.open(state.current_image.path) as image:
        metadata = json.loads(image.info[PNG_METADATA_KEY])
    assert metadata["prompt"] == "two"
    assert metadata["guidance_scale"] == 0.0
    assert metadata["shift"] == 1.15


def test_gui_generation_passes_lora_patches_and_records_metadata(tmp_path: Path) -> None:
    pipeline = _FakePipeline()
    state = _state(tmp_path, pipeline)
    _write_projector_diff_lora(state.config.lora_dir / "filter.safetensors")
    request = gui._generation_request_from_payload(
        {
            "prompt": "with lora",
            "width": 16,
            "height": 16,
            "steps": 1,
            "seed": 3,
            "loras": [{"id": "filter.safetensors", "scale": 2}],
        }
    )

    accepted, _ = state.start_generation(request)
    assert accepted is True
    _wait_for_idle(state)

    call = pipeline.calls[0]
    assert call["kwargs"]["loras"][0].scale == 2.0
    assert state.current_image is not None
    assert state.current_image.loras[0]["display_name"] == "filter"
    assert state.snapshot()["image"]["loras"][0]["scale"] == 2.0
    with Image.open(state.current_image.path) as image:
        metadata = json.loads(image.info[PNG_METADATA_KEY])
    assert metadata["loras"][0]["id"] == "filter.safetensors"
    assert metadata["loras"][0]["scale"] == 2.0


def test_gui_preview_status_and_endpoint_serve_latest_frame(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<!doctype html>Krea", encoding="utf-8")
    state = _state(tmp_path / "outputs", _FakePipeline())
    token = "secret-token"
    state.record_preview_frame(
        PipelinePreviewFrame(
            mode="latent",
            step_index=1,
            step_count=4,
            width=16,
            height=16,
            image=np.zeros((2, 2, 3), dtype=np.float32),
        )
    )

    snapshot = state.snapshot()
    preview = snapshot["preview"]
    assert preview["mode"] == "latent"
    assert preview["step"] == 2
    assert preview["step_count"] == 4
    assert preview["width"] == 16
    assert preview["height"] == 16
    assert preview["url"] == "/api/preview/current?rev=1"

    handler = gui._gui_handler(
        state,
        static_root=static_root,
        session_token=token,
        allow_unsafe_host=False,
    )

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(
            f"{base}{preview['url']}&token={token}",
            timeout=2,
        ) as response:
            data = response.read()
            content_type = response.headers["Content-Type"]
            cache_control = response.headers["Cache-Control"]
        assert content_type == "image/jpeg"
        assert cache_control == "no-store"
        assert data.startswith(b"\xff\xd8")

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"{base}/api/preview/current?rev=999&token={token}",
                timeout=2,
            )
        assert exc.value.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_gui_preview_clears_on_cancel_error_and_completion(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    state = _state(tmp_path, _FakePipeline())
    frame = PipelinePreviewFrame(
        mode="latent",
        step_index=0,
        step_count=2,
        width=16,
        height=16,
        image=np.zeros((2, 2, 3), dtype=np.float32),
    )

    state.record_preview_frame(frame)
    with state.lock:
        state.generation_running = True
    accepted, _ = state.start_cancel_current_generation()
    assert accepted is True
    assert state.snapshot()["preview"] is None

    state.record_preview_frame(frame)
    state._finish_with_error(Krea2TurboMlxError("boom"), phase="generate")
    assert state.snapshot()["preview"] is None

    pipeline = _PreviewPipeline(frame)
    state = _state(tmp_path / "complete", pipeline)
    accepted, _ = state.start_generation(
        gui._GenerationRequest(
            "preview complete",
            width=16,
            height=16,
            steps=1,
            seed=8,
            live_preview="latent",
        )
    )
    assert accepted is True
    _wait_for_idle(state)

    assert pipeline.preview_emitted is True
    assert state.snapshot()["preview"] is None


def test_gui_batch_snapshot_exposes_job_and_batch_progress(tmp_path: Path) -> None:
    pipeline = _BlockingPipeline()
    state = _state(tmp_path, pipeline)
    request = gui._GenerationRequest(
        "a glass library",
        width=16,
        height=16,
        steps=4,
        seed=99,
    )
    upcoming = gui._GenerationRequest(
        "a cedar observatory",
        width=32,
        height=16,
        steps=3,
        seed=101,
        loras=(gui.LoraReference("style.safetensors", scale=2.0),),
    )

    accepted, _ = state.start_batch_generation([request, upcoming])
    assert accepted is True
    assert pipeline.started.wait(timeout=2.0)

    snapshot = state.snapshot()
    batch = snapshot["batch"]
    assert batch is not None
    assert batch["index"] == 1
    assert batch["total"] == 2
    assert batch["prompt"] == "a glass library"
    assert batch["progress"] == 0.85
    assert batch["overall_progress"] == 0.425
    assert batch["jobs"] == [
        {
            "index": 1,
            "status": "running",
            "prompt": "a glass library",
            "width": 16,
            "height": 16,
            "steps": 4,
            "seed": 99,
            "loras": [],
        },
        {
            "index": 2,
            "status": "queued",
            "prompt": "a cedar observatory",
            "width": 32,
            "height": 16,
            "steps": 3,
            "seed": 101,
            "loras": [{"id": "style.safetensors", "scale": 2.0}],
        },
    ]
    assert isinstance(batch["started_ms"], int)
    assert isinstance(batch["current_started_ms"], int)
    assert batch["current_started_ms"] >= batch["started_ms"]

    pipeline.release.set()
    _wait_for_idle(state)
    assert state.snapshot()["batch"] is None


def test_gui_batch_uses_selected_output_directory(tmp_path: Path) -> None:
    default_output = tmp_path / "default"
    selected_output = tmp_path / "selected"
    pipeline = _FakePipeline()
    state = _state(default_output, pipeline)
    requests = [
        gui._GenerationRequest("selected", width=16, height=16, steps=1, seed=9),
    ]

    accepted, _ = state.start_batch_generation(
        requests,
        output_dir=selected_output,
    )
    assert accepted is True
    _wait_for_idle(state)

    assert not list(default_output.glob("*.png"))
    selected_files = list(selected_output.glob("*.png"))
    assert len(selected_files) == 1
    snapshot = state.snapshot()
    assert snapshot["output_dir"]["path"] == str(selected_output)
    assert snapshot["recent"][0]["prompt"] == "selected"


def test_gui_clipboard_json_file_reader_is_bounded(tmp_path: Path) -> None:
    batch_file = tmp_path / "batch.json"
    batch_file.write_text('[{"prompt":"one"}]', encoding="utf-8")

    assert gui._read_clipboard_json_file(batch_file) == '[{"prompt":"one"}]'

    object_file = tmp_path / "object.json"
    object_file.write_text('{"jobs":[{"prompt":"one"}]}', encoding="utf-8")
    with pytest.raises(Krea2TurboMlxError, match=r"start with \["):
        gui._read_clipboard_json_file(object_file)

    text_file = tmp_path / "batch.txt"
    text_file.write_text('[{"prompt":"one"}]', encoding="utf-8")
    with pytest.raises(Krea2TurboMlxError, match=".json"):
        gui._read_clipboard_json_file(text_file)

    large_file = tmp_path / "large.json"
    large_file.write_bytes(b"[" + b" " * gui.MAX_GUI_BATCH_BYTES + b"]")
    with pytest.raises(Krea2TurboMlxError, match="smaller"):
        gui._read_clipboard_json_file(large_file)


def test_gui_generation_events_include_closed_timing(tmp_path: Path) -> None:
    state = _state(tmp_path, _FakePipeline())

    accepted, _ = state.start_generation(
        gui._GenerationRequest("timed", width=16, height=16, steps=1, seed=5)
    )
    assert accepted is True
    _wait_for_idle(state)

    snapshot = state.snapshot()
    task = snapshot["task"]
    events = snapshot["events"]

    assert snapshot["progress"] == 0.0
    assert task["name"] == "Generate image"
    assert isinstance(task["started_ms"], int)
    assert isinstance(task["completed_ms"], int)
    assert task["completed_ms"] >= task["started_ms"]
    assert events
    assert events[-1]["kind"] == "task"
    assert events[-1]["message"] == "Generate image"
    assert events[-1]["progress"] == 1.0
    assert events[-1]["completed_ms"] == task["completed_ms"]
    assert events[-1]["details"]["elapsed_seconds"] == pytest.approx(
        (task["completed_ms"] - task["started_ms"]) / 1000
    )
    event_ids = [event["id"] for event in events]
    assert all(isinstance(event_id, int) for event_id in event_ids)
    assert event_ids == sorted(set(event_ids))
    assert all(isinstance(event["time_ms"], int) for event in events)
    assert all(event["completed_ms"] >= event["time_ms"] for event in events)


def test_gui_batch_cancel_current_continues_with_next_job(tmp_path: Path) -> None:
    pipeline = _BlockingPipeline()
    state = _state(tmp_path, pipeline)
    requests = [
        gui._GenerationRequest("one", width=16, height=16, steps=1, seed=1),
        gui._GenerationRequest("two", width=16, height=16, steps=1, seed=2),
    ]

    accepted, _ = state.start_batch_generation(requests)
    assert accepted is True
    assert pipeline.started.wait(timeout=2.0)

    accepted, _ = state.start_cancel_current_generation()
    assert accepted is True
    snapshot = state.snapshot()
    assert snapshot["batch"]["jobs"][0]["status"] == "cancelling"
    assert snapshot["batch"]["jobs"][1]["status"] == "queued"

    pipeline.release.set()
    _wait_for_idle(state)

    assert [call["prompt"] for call in pipeline.calls] == ["one", "two"]
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert state.error is None


def test_gui_batch_clear_queue_finishes_current_job(tmp_path: Path) -> None:
    pipeline = _BlockingPipeline()
    state = _state(tmp_path, pipeline)
    requests = [
        gui._GenerationRequest("one", width=16, height=16, steps=1, seed=1),
        gui._GenerationRequest("two", width=16, height=16, steps=1, seed=2),
    ]

    accepted, _ = state.start_batch_generation(requests)
    assert accepted is True
    assert pipeline.started.wait(timeout=2.0)

    accepted, _ = state.start_clear_batch_queue()
    assert accepted is True
    snapshot = state.snapshot()
    assert snapshot["batch"]["clear_queue_requested"] is True
    assert snapshot["batch"]["queue_remaining"] == 0
    assert [job["status"] for job in snapshot["batch"]["jobs"]] == [
        "running",
        "cleared",
    ]

    pipeline.release.set()
    _wait_for_idle(state)

    assert [call["prompt"] for call in pipeline.calls] == ["one"]
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert state.error is None


def test_gui_batch_cancel_current_then_clear_queue_stops_batch(
    tmp_path: Path,
) -> None:
    pipeline = _BlockingPipeline()
    state = _state(tmp_path, pipeline)
    requests = [
        gui._GenerationRequest("one", width=16, height=16, steps=1, seed=1),
        gui._GenerationRequest("two", width=16, height=16, steps=1, seed=2),
    ]

    accepted, _ = state.start_batch_generation(requests)
    assert accepted is True
    assert pipeline.started.wait(timeout=2.0)

    accepted, _ = state.start_cancel_current_generation()
    assert accepted is True
    accepted, _ = state.start_clear_batch_queue()
    assert accepted is True
    snapshot = state.snapshot()
    assert [job["status"] for job in snapshot["batch"]["jobs"]] == [
        "cancelling",
        "cleared",
    ]

    pipeline.release.set()
    _wait_for_idle(state)

    assert [call["prompt"] for call in pipeline.calls] == ["one"]
    assert list(tmp_path.glob("*.png")) == []
    assert state.phase == "cancelled"
    assert state.error is None


def test_gui_shutdown_cancels_current_job_and_clears_batch_queue(
    tmp_path: Path,
) -> None:
    pipeline = _BlockingPipeline()
    state = _state(tmp_path, pipeline)
    requests = [
        gui._GenerationRequest("one", width=16, height=16, steps=1, seed=1),
        gui._GenerationRequest("two", width=16, height=16, steps=1, seed=2),
    ]

    accepted, _ = state.start_batch_generation(requests)
    assert accepted is True
    assert pipeline.started.wait(timeout=2.0)

    assert state.request_shutdown() is True
    snapshot = state.snapshot()
    assert snapshot["batch"]["cancel_current_requested"] is True
    assert snapshot["batch"]["clear_queue_requested"] is True
    assert [job["status"] for job in snapshot["batch"]["jobs"]] == [
        "cancelling",
        "cleared",
    ]

    pipeline.release.set()
    _wait_for_idle(state)

    assert [call["prompt"] for call in pipeline.calls] == ["one"]
    assert list(tmp_path.glob("*.png")) == []
    assert state.phase == "cancelled"
    assert state.error is None


def test_gui_gallery_loads_existing_png_metadata(tmp_path: Path) -> None:
    output = tmp_path / "krea-gui-20260625-120000-16x16-steps2-seed77.png"
    save_generation_png(
        Image.new("RGB", (1, 1), (77, 8, 7)),
        output,
        metadata=generation_metadata_payload(
            prompt="archival glass",
            seed=77,
            width=16,
            height=16,
            steps=2,
            guidance_scale=0.0,
            shift=1.15,
            model_path=Path("artifact"),
            elapsed_seconds=0.1,
            loras=[
                {
                    "id": "style.safetensors",
                    "display_name": "Style",
                    "scale": 1.3,
                    "warnings": ["skipped unsupported target"],
                }
            ],
        ),
        overwrite=False,
    )

    state = _state(tmp_path, _FakePipeline())

    assert len(state.recent_generations) == 1
    assert state.current_image is not None
    assert state.current_image.prompt == "archival glass"
    assert state.current_image.seed == 77
    assert state.snapshot()["recent"][0]["loras"] == [
        {
            "id": "style.safetensors",
            "display_name": "Style",
            "scale": 1.3,
            "warnings": ["skipped unsupported target"],
        }
    ]


def test_gui_source_image_metadata_extracts_supported_settings(tmp_path: Path) -> None:
    output = tmp_path / "source.png"
    metadata = generation_metadata_payload(
        prompt="source glass",
        seed=77,
        width=16,
        height=32,
        steps=2,
        guidance_scale=0.0,
        shift=1.15,
        model_path=Path("artifact"),
        elapsed_seconds=0.1,
        loras=[
            {
                "id": "style.safetensors",
                "display_name": "Style",
                "scale": 2.5,
            }
        ],
    )
    save_generation_png(
        Image.new("RGB", (1, 1), (77, 8, 7)),
        output,
        metadata=metadata,
    )

    result = gui._source_image_metadata_from_path(output)

    assert result["settings"] == {
        "prompt": "source glass",
        "width": 16,
        "height": 32,
        "steps": 2,
        "seed": 77,
        "loras": [{"id": "style.safetensors", "scale": 2.5}],
    }
    assert [entry["key"] for entry in result["supported"]] == [
        "prompt",
        "width",
        "height",
        "steps",
        "seed",
        "loras",
    ]
    other_keys = {entry["key"] for entry in result["other"]}
    assert "model_path" in other_keys
    assert PNG_PARAMETERS_KEY in other_keys

    raw = output.read_bytes()
    payload_result = gui._source_image_metadata_from_payload(
        {
            "filename": "source.png",
            "image_base64": base64.b64encode(raw).decode("ascii"),
        }
    )
    assert payload_result["settings"] == result["settings"]


def test_gui_source_image_metadata_rejects_missing_project_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "plain.png"
    Image.new("RGB", (1, 1), (77, 8, 7)).save(output)

    with pytest.raises(Krea2TurboMlxError, match="missing"):
        gui._source_image_metadata_from_path(output)


def test_gui_gallery_syncs_output_directory_changes(tmp_path: Path) -> None:
    state = _state(tmp_path, _FakePipeline())
    first = _write_output_png(tmp_path, "first", seed=1)

    snapshot = state.snapshot()

    assert [item["prompt"] for item in snapshot["recent"]] == ["first"]

    second = _write_output_png(tmp_path, "second", seed=2)
    snapshot = state.snapshot()
    prompts = {item["prompt"] for item in snapshot["recent"]}

    assert prompts == {"first", "second"}
    first.unlink()
    snapshot = state.snapshot()

    assert [item["prompt"] for item in snapshot["recent"]] == ["second"]
    assert snapshot["image"]["prompt"] == "second"
    assert second.exists()


def test_gui_delete_image_removes_file_and_refreshes_gallery(tmp_path: Path) -> None:
    path = _write_output_png(tmp_path, "delete me", seed=3)
    state = _state(tmp_path, _FakePipeline())
    image_id = state.snapshot()["recent"][0]["id"]

    accepted, message = state.delete_image(image_id)

    assert accepted is True
    assert message == "Image deleted."
    assert not path.exists()
    snapshot = state.snapshot()
    assert snapshot["recent"] == []
    assert snapshot["image"] is None
    event = snapshot["events"][-1]
    assert event["message"] == f"Deleted {path.name}"
    assert event["completed_ms"] >= event["time_ms"]
    assert event["details"]["elapsed_seconds"] >= 0


def test_gui_static_paths_block_traversal_and_fallback_to_spa(tmp_path: Path) -> None:
    static_root = tmp_path / "frontend" / "build" / "client"
    static_root.mkdir(parents=True)
    index = static_root / "index.html"
    index.write_text("<!doctype html>Krea", encoding="utf-8")

    assert gui._static_path_from_url(static_root, "/") == index.resolve()
    assert gui._static_path_from_url(static_root, "/workspace") == index
    with pytest.raises(Krea2TurboMlxError, match="escapes"):
        gui._static_path_from_url(static_root, "/../secret.txt")


def test_gui_http_handler_serves_static_without_token_and_protects_api(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<!doctype html>Krea", encoding="utf-8")
    state = _state(tmp_path / "outputs", _FakePipeline())
    token = "secret-token"
    handler = gui._gui_handler(
        state,
        static_root=static_root,
        session_token=token,
        allow_unsafe_host=False,
    )

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert urllib.request.urlopen(f"{base}/", timeout=2).status == 200
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/api/status", timeout=2)
        assert exc.value.code == 403

        with urllib.request.urlopen(f"{base}/api/status?token={token}", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["server"]["status"] == "running"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_gui_http_handler_rejects_cross_origin_post_with_valid_token(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<!doctype html>Krea", encoding="utf-8")
    state = _state(tmp_path / "outputs", _FakePipeline())
    token = "secret-token"
    handler = gui._gui_handler(
        state,
        static_root=static_root,
        session_token=token,
        allow_unsafe_host=False,
    )

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    batch_payload = json.dumps(
        {
            "jobs": [
                {
                    "prompt": "a glass observatory",
                    "width": 16,
                    "height": 16,
                    "steps": 1,
                    "seed": 1,
                }
            ]
        }
    ).encode("utf-8")
    try:
        valid_request = urllib.request.Request(
            f"{base}/api/validate-batch",
            data=batch_payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Origin": base,
                "X-Krea-Session-Token": token,
            },
        )
        with urllib.request.urlopen(valid_request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload == {"count": 1, "ok": True}

        cross_origin_request = urllib.request.Request(
            f"{base}/api/validate-batch",
            data=batch_payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://example.test",
                "X-Krea-Session-Token": token,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(cross_origin_request, timeout=2)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_gui_parser_exposes_public_flags() -> None:
    args = cli.build_parser().parse_args(
        [
            "gui",
            "--config",
            "config.json",
            "--host",
            "127.0.0.1",
            "--port",
            "9999",
            "--no-browser",
            "--no-preload",
            "--unsafe-host",
        ]
    )

    assert args.command == "gui"
    assert args.config == Path("config.json")
    assert args.host == "127.0.0.1"
    assert args.port == 9999
    assert args.no_browser is True
    assert args.no_preload is True
    assert args.unsafe_host is True


def test_gui_missing_frontend_build_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(Krea2TurboMlxError, match="React GUI build is missing"):
        gui._validate_static_root(tmp_path / "frontend" / "build" / "client")


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.transformer = SimpleNamespace(
            text_fusion=SimpleNamespace(
                projector=SimpleNamespace(
                    weight=SimpleNamespace(shape=(1, 12)),
                )
            )
        )

    def __call__(self, prompt: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(
                PipelineProgressEvent(
                    "denoise_step_end",
                    "denoise",
                    "Finished step 1/1",
                    progress=0.85,
                    step_index=0,
                    step_count=1,
                )
            )
        seed = int(kwargs["seed"])
        return SimpleNamespace(
            images=[Image.new("RGB", (1, 1), (seed % 255, 8, 7))],
            seed=seed,
            elapsed_seconds=0.1,
            truncation_warnings=(),
        )


class _BlockingPipeline(_FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, prompt: str, **kwargs: object) -> SimpleNamespace:
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(
                PipelineProgressEvent(
                    "denoise_step_end",
                    "denoise",
                    "Finished step 1/1",
                    progress=0.85,
                    step_index=0,
                    step_count=1,
                )
            )
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting for test release")
        return super().__call__(prompt, **kwargs)


class _PreviewPipeline(_FakePipeline):
    def __init__(self, frame: PipelinePreviewFrame) -> None:
        super().__init__()
        self.frame = frame
        self.preview_emitted = False

    def __call__(self, prompt: str, **kwargs: object) -> SimpleNamespace:
        preview_callback = kwargs.get("preview_callback")
        if callable(preview_callback):
            preview_callback(self.frame)
            self.preview_emitted = True
        return super().__call__(prompt, **kwargs)


def _state(
    output_dir: Path,
    pipeline: _FakePipeline,
    *,
    settings_path: Path | None = None,
) -> gui._GuiState:
    output_dir.mkdir(parents=True, exist_ok=True)
    return gui._GuiState(
        config=SetupConfig(
            artifact_dir=Path("artifact"),
            output_dir=output_dir,
            lora_dir=output_dir.parent / "loras",
        ),
        stream=StringIO(),
        settings_path=settings_path,
        pipeline_loader=lambda model, progress_callback: pipeline,
    )


def _write_artifact_precision(artifact: Path, dtypes: dict[str, int]) -> None:
    artifact.joinpath("artifact.json").write_text(
        json.dumps(
            {
                "precision": {
                    "selected_dtype_histogram": dtypes,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_output_png(output_dir: Path, prompt: str, *, seed: int) -> Path:
    output = output_dir / f"krea-gui-test-16x16-steps2-seed{seed}.png"
    save_generation_png(
        Image.new("RGB", (1, 1), (seed % 255, 8, 7)),
        output,
        metadata=generation_metadata_payload(
            prompt=prompt,
            seed=seed,
            width=16,
            height=16,
            steps=2,
            guidance_scale=0.0,
            shift=1.15,
            model_path=Path("artifact"),
            elapsed_seconds=0.1,
        ),
        overwrite=False,
    )
    return output


def _write_projector_diff_lora(path: Path) -> None:
    key = "diffusion_model.txtfusion.projector.diff"
    write_safetensors_fixture(
        path,
        {key: ("F32", [1, 12])},
        payloads={
            key: struct.pack(
                "<12f",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.5,
                -0.75,
                -0.25,
                0,
            )
        },
    )


def _wait_for_idle(state: gui._GuiState, *, timeout: float = 2.0) -> None:
    _wait_for(state, lambda: not state.generation_running, timeout=timeout)


def _wait_for(
    state: gui._GuiState,
    predicate,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with state.lock:
            if predicate():
                return
        time.sleep(0.01)
    raise AssertionError("GUI state did not reach the expected condition")
