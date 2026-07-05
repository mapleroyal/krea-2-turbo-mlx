from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

import krea_2_turbo_mlx.gui as gui
import krea_2_turbo_mlx.setup_flow as setup_flow
from krea_2_turbo_mlx.setup_flow import SetupConfig


def test_static_index_validates_host_and_bootstraps_only_with_valid_token(
    tmp_path: Path,
) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text(
        "<!doctype html><html><head></head><body>Krea</body></html>",
        encoding="utf-8",
    )
    state = _state(tmp_path / "outputs")
    token = "secret-token"

    with _gui_server(state, static_root=static_root, token=token) as base:
        with urllib.request.urlopen(f"{base}/?token={token}", timeout=2) as response:
            authenticated = response.read().decode("utf-8")
        assert "auth_required" not in authenticated
        assert str(state.config.artifact_dir) in authenticated
        assert token not in authenticated
        assert "token=" not in authenticated

        with urllib.request.urlopen(f"{base}/", timeout=2) as response:
            preauth = response.read().decode("utf-8")
        assert "auth_required" in preauth
        assert str(state.config.artifact_dir) not in preauth
        assert '"ui_settings": null' in preauth
        assert token not in preauth
        assert "token=" not in preauth

        bad_host = urllib.request.Request(
            f"{base}/",
            headers={"Host": "example.test"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_host, timeout=2)
        assert exc.value.code == 403


def test_gui_events_and_terminal_logs_redact_tokenized_urls(tmp_path: Path) -> None:
    stream = StringIO()
    state = _state(tmp_path / "outputs", stream=stream)

    state.record_system_event(
        "server",
        "started",
        details={
            "url": "http://127.0.0.1:7860/?token=secret-token&x=1",
            "token": "secret-token",
            "nested": ["/api/status?session_token=other-secret"],
        },
    )

    snapshot_text = json.dumps(state.snapshot(), sort_keys=True)
    log_text = stream.getvalue()
    assert "secret-token" not in snapshot_text
    assert "other-secret" not in snapshot_text
    assert "secret-token" not in log_text
    assert "other-secret" not in log_text
    assert "[redacted]" in snapshot_text
    assert "[redacted]" in log_text


def test_gui_http_expected_errors_are_400_and_unexpected_errors_are_generic_500(
    tmp_path: Path,
) -> None:
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<!doctype html>Krea", encoding="utf-8")
    stream = StringIO()
    state = _state(tmp_path / "outputs", stream=stream)
    token = "secret-token"

    def boom(payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("exploded internal detail")

    state.update_ui_settings = boom  # type: ignore[method-assign]

    with _gui_server(state, static_root=static_root, token=token) as base:
        expected = urllib.request.Request(
            f"{base}/api/validate-batch",
            data=json.dumps({"jobs": []}).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Origin": base,
                "X-Krea-Session-Token": token,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(expected, timeout=2)
        assert exc.value.code == 400
        expected_payload = json.loads(exc.value.read().decode("utf-8"))
        assert expected_payload["message"] == "Batch must include at least one job."

        unexpected = urllib.request.Request(
            f"{base}/api/ui-settings",
            data=b"{}",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Origin": base,
                "X-Krea-Session-Token": token,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(unexpected, timeout=2)
        assert exc.value.code == 500
        unexpected_payload = json.loads(exc.value.read().decode("utf-8"))
        assert "unexpected" in unexpected_payload["message"].lower()
        assert "exploded internal detail" not in unexpected_payload["message"]

    assert "Traceback" in stream.getvalue()
    assert "exploded internal detail" in stream.getvalue()


def test_snapshot_scans_output_directory_outside_state_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(tmp_path / "outputs")
    state.lock = threading.Lock()  # type: ignore[assignment]
    observed = {}

    def fake_entries(output_dir: Path) -> list[tuple[Path, int, int]]:
        acquired = state.lock.acquire(blocking=False)
        observed["lock_was_free"] = acquired
        if acquired:
            state.lock.release()
        return []

    monkeypatch.setattr(gui, "_generation_file_entries", fake_entries)

    state.snapshot()

    assert observed["lock_was_free"] is True


def test_setup_handler_returns_400_for_expected_and_500_for_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "secret-token"
    with _setup_server(token=token) as base:
        expected = urllib.request.Request(
            f"{base}/?token={token}",
            data=urllib.parse.urlencode(
                {
                    "session_token": token,
                    "source_dir": ".",
                    "artifact_dir": "artifact",
                    "output_dir": "outputs",
                    "lora_dir": "loras",
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": base,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(expected, timeout=2)
        assert exc.value.code == 400
        assert "Setup source_dir must be a real path" in exc.value.read().decode("utf-8")

    def boom(cls: type[SetupConfig], fields: dict[str, list[str]]) -> SetupConfig:
        raise RuntimeError("setup internal detail")

    monkeypatch.setattr(setup_flow.SetupConfig, "from_form", classmethod(boom))
    with _setup_server(token=token) as base:
        unexpected = urllib.request.Request(
            f"{base}/?token={token}",
            data=urllib.parse.urlencode(
                {
                    "session_token": token,
                    "source_dir": "source",
                    "artifact_dir": "artifact",
                    "output_dir": "outputs",
                    "lora_dir": "loras",
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": base,
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(unexpected, timeout=2)
        body = exc.value.read().decode("utf-8")
        assert exc.value.code == 500
        assert "unexpected setup error" in body.lower()
        assert "setup internal detail" not in body

    assert "Traceback" in capsys.readouterr().err


@contextmanager
def _gui_server(
    state: gui._GuiState,
    *,
    static_root: Path,
    token: str,
) -> Iterator[str]:
    handler = gui._gui_handler(
        state,
        static_root=static_root,
        session_token=token,
        allow_unsafe_host=False,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@contextmanager
def _setup_server(*, token: str) -> Iterator[str]:
    done = threading.Event()
    result: dict[str, object] = {}
    handler = setup_flow._setup_handler(
        SetupConfig(),
        done,
        result,
        session_token=token,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _state(output_dir: Path, *, stream: StringIO | None = None) -> gui._GuiState:
    output_dir.mkdir(parents=True, exist_ok=True)
    return gui._GuiState(
        config=SetupConfig(
            artifact_dir=Path("artifact"),
            output_dir=output_dir,
            lora_dir=output_dir.parent / "loras",
        ),
        stream=stream if stream is not None else StringIO(),
        pipeline_loader=lambda model, progress_callback: SimpleNamespace(),
    )
