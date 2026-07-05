from __future__ import annotations

import json
import os
import subprocess
import sys
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from krea_2_turbo_mlx import cli
from krea_2_turbo_mlx.constants import OFFICIAL_HF_REVISION
from safetensors_fixtures import write_safetensors_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _stub_png_save(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_save(
        images: object,
        output: Path,
        *,
        metadata: dict[str, object],
        overwrite: bool,
    ) -> None:
        del images, metadata, overwrite
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(cli, "_save_generated_png", fake_save)
    monkeypatch.setattr(cli, "_raise_runtime_doctor_errors", lambda *, model=None: None)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "krea_2_turbo_mlx", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_help_has_no_quantization_surface() -> None:
    """M0 is full-precision only; the CLI must not expose a quantization surface."""
    result = run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "quantize" not in result.stdout
    assert "q8" not in result.stdout.lower()


def test_download_command_json_reports_component_only_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    downloaded = tmp_path / "Krea-2-Turbo"
    monkeypatch.setattr(cli, "download_source", lambda *args, **kwargs: downloaded)
    args = cli.build_parser().parse_args(
        ["download", "--dest", str(tmp_path), "--json"]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "source_ready"
    assert payload["revision"] == OFFICIAL_HF_REVISION
    assert "transformer/**" in payload["allow_patterns"]
    assert "turbo.safetensors" in payload["ignore_patterns"]


def test_convert_command_prints_local_dry_plan_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "artifact"
    _write_cli_source(source)
    result = run_cli(
        "convert",
        "--source",
        str(source),
        "--output",
        str(output),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["status"] == "dry_conversion_plan_generated"
    assert plan["output"] == str(output)
    assert plan["full_precision_only"] is True
    assert output.exists() is False

    missing_revision = run_cli("convert", "--source", "someone/model")
    assert missing_revision.returncode == 1
    assert "requires --revision" in missing_revision.stderr
    assert "Traceback" not in missing_revision.stderr


def test_generate_command_writes_png_and_reports_effective_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "generated.png"
    calls: dict[str, object] = {}

    class FakePipeline:
        def __call__(
            self,
            prompt: str,
            *,
            width: int,
            height: int,
            steps: int,
            guidance_scale: float,
            seed: int | None,
            progress_callback: object | None = None,
        ) -> object:
            assert progress_callback is None
            calls["generate"] = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            }
            return SimpleNamespace(images=[_FakeImage()], seed=12345)

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass observatory",
            "--width",
            "32",
            "--height",
            "16",
            "--steps",
            "2",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "image_generated"
    assert payload["output"] == str(output.resolve())
    assert payload["seed"] == 12345
    assert payload["shift"] == 1.15
    assert "loras" not in payload
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert calls["generate"] == {
        "prompt": "a glass observatory",
        "width": 32,
        "height": 16,
        "steps": 2,
        "guidance_scale": 0.0,
        "seed": None,
    }


def test_generate_command_resolves_lora_and_reports_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "generated.png"
    lora_path = tmp_path / "filter.safetensors"
    _write_projector_diff_lora(lora_path)
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self) -> None:
            self.transformer = SimpleNamespace(
                text_fusion=SimpleNamespace(
                    projector=SimpleNamespace(
                        weight=SimpleNamespace(shape=(1, 12)),
                    )
                )
            )

        def __call__(self, prompt: str, **kwargs: object) -> object:
            calls["prompt"] = prompt
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                images=[_FakeImage()],
                seed=7,
                elapsed_seconds=0.1,
                truncation_warnings=(),
            )

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass observatory",
            "--output",
            str(output),
            "--lora",
            f"{lora_path}:2",
            "--json",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    loras = payload["loras"]
    assert loras[0]["id"] == "filter.safetensors"
    assert loras[0]["display_name"] == "filter"
    assert loras[0]["scale"] == 2.0
    assert loras[0]["source_type"] == "path"
    assert loras[0]["adapter_type"] == "weight-diff"
    assert calls["prompt"] == "a glass observatory"
    assert calls["kwargs"]["loras"][0].scale == 2.0


def test_generate_command_resolves_catalog_lora_from_lora_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "generated.png"
    lora_dir = tmp_path / "loras"
    lora_path = lora_dir / "glass.safetensors"
    write_safetensors_fixture(
        lora_path,
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
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self) -> None:
            self.transformer = SimpleNamespace(
                text_fusion=SimpleNamespace(
                    projector=SimpleNamespace(weight=SimpleNamespace(shape=(1, 12)))
                ),
                transformer_blocks=[
                    SimpleNamespace(
                        attn=SimpleNamespace(
                            to_q=SimpleNamespace(weight=SimpleNamespace(shape=(4, 3)))
                        )
                    )
                ],
            )

        def __call__(self, prompt: str, **kwargs: object) -> object:
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                images=[_FakeImage()],
                seed=8,
                elapsed_seconds=0.2,
                truncation_warnings=(),
            )

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass observatory",
            "--output",
            str(output),
            "--lora-dir",
            str(lora_dir),
            "--lora",
            "glass.safetensors:0.5",
            "--json",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    lora = payload["loras"][0]
    assert lora["id"] == "glass.safetensors"
    assert lora["source_type"] == "catalog"
    assert lora["adapter_type"] == "standard"
    assert lora["scale"] == 0.5
    assert lora["target_count"] == 1
    assert len(lora["sha256"]) == 64
    assert calls["kwargs"]["loras"][0].id == "glass.safetensors"


def test_generate_command_resolves_explicit_lora_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "generated.png"
    lora_path = tmp_path / "direct.safetensors"
    write_safetensors_fixture(
        lora_path,
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
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self) -> None:
            self.transformer = SimpleNamespace(
                text_fusion=SimpleNamespace(
                    projector=SimpleNamespace(weight=SimpleNamespace(shape=(1, 12)))
                ),
                transformer_blocks=[
                    SimpleNamespace(
                        attn=SimpleNamespace(
                            to_q=SimpleNamespace(weight=SimpleNamespace(shape=(4, 3)))
                        )
                    )
                ],
            )

        def __call__(self, prompt: str, **kwargs: object) -> object:
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                images=[_FakeImage()],
                seed=9,
                elapsed_seconds=0.2,
                truncation_warnings=(),
            )

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass observatory",
            "--output",
            str(output),
            "--lora",
            f"{lora_path}:0.25",
            "--json",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    lora = payload["loras"][0]
    assert lora["id"] == "direct.safetensors"
    assert lora["source_type"] == "path"
    assert lora["adapter_type"] == "standard"
    assert lora["scale"] == 0.25
    assert lora["path"] == str(lora_path)
    assert calls["kwargs"]["loras"][0].path == str(lora_path)


def test_generate_command_reports_lora_path_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.transformer = SimpleNamespace(
                text_fusion=SimpleNamespace(
                    projector=SimpleNamespace(weight=SimpleNamespace(shape=(1, 12)))
                )
            )

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )

    missing = tmp_path / "missing.safetensors"
    assert (
        cli.main(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass observatory",
                "--output",
                str(tmp_path / "missing-output.png"),
                "--lora",
                str(missing),
                "--json",
            ]
        )
        == 1
    )
    stderr = capsys.readouterr().err
    assert f"LoRA file not found: {missing}" in stderr
    assert "Traceback" not in stderr

    wrong_extension = tmp_path / "adapter.bin"
    wrong_extension.write_bytes(b"not a safetensors file")
    assert (
        cli.main(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass observatory",
                "--output",
                str(tmp_path / "wrong-extension-output.png"),
                "--lora",
                str(wrong_extension),
                "--json",
            ]
        )
        == 1
    )
    stderr = capsys.readouterr().err
    assert f"LoRA file must end in .safetensors: {wrong_extension}" in stderr
    assert "Traceback" not in stderr


def test_generate_command_runs_sequential_batch_with_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "outputs"
    calls: list[tuple[str, int | None]] = []

    class FakePipeline:
        def __call__(
            self,
            prompt: str,
            *,
            width: int,
            height: int,
            steps: int,
            guidance_scale: float,
            seed: int | None,
            progress_callback: object | None = None,
        ) -> object:
            del width, height, steps, guidance_scale, progress_callback
            calls.append((prompt, seed))
            return SimpleNamespace(
                images=[_FakeImage()],
                seed=seed,
                elapsed_seconds=0.25,
                truncation_warnings=(
                    {
                        "prompt_index": 0,
                        "token_count": 999,
                        "max_length": 512,
                        "truncated_tokens": 487,
                    },
                ),
            )

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "first",
            "--prompt",
            "second",
            "--num-images",
            "2",
            "--seeds",
            "10,11,12,13",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "images_generated"
    assert payload["count"] == 4
    assert [item["seed"] for item in payload["outputs"]] == [10, 11, 12, 13]
    assert calls == [("first", 10), ("first", 11), ("second", 12), ("second", 13)]
    assert len(list(output_dir.glob("*.png"))) == 4
    assert payload["outputs"][0]["prompt_truncation"][0]["truncated_tokens"] == 487


def test_generate_command_uses_unique_implicit_output_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing = output_dir / "image-seed77.png"
    existing.write_bytes(b"existing")

    class FakePipeline:
        def __call__(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                images=[_FakeImage()],
                seed=77,
                elapsed_seconds=0.1,
                truncation_warnings=(),
            )

    monkeypatch.setattr(
        cli,
        "_load_generate_pipeline",
        lambda model, *, progress_callback=None: FakePipeline(),
    )
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass observatory",
            "--output-dir",
            str(output_dir),
            "--output-template",
            "image-seed{seed}.png",
            "--json",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str((output_dir / "image-seed77-1.png").resolve())
    assert existing.read_bytes() == b"existing"
    assert (output_dir / "image-seed77-1.png").is_file()


def test_generate_progress_modes_emit_or_suppress_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def run_with_progress(mode: str, output: Path) -> tuple[list[object | None], str]:
        load_callbacks: list[object | None] = []
        call_callbacks: list[object | None] = []

        class FakePipeline:
            def __call__(self, *args: object, **kwargs: object) -> object:
                progress_callback = kwargs["progress_callback"]
                call_callbacks.append(progress_callback)
                if progress_callback is not None:
                    progress_callback(
                        SimpleNamespace(
                            stage="denoise",
                            message="Denoising step",
                            step_index=0,
                            step_count=1,
                        )
                    )
                return SimpleNamespace(
                    images=[_FakeImage()],
                    seed=5,
                    elapsed_seconds=0.1,
                    truncation_warnings=(),
                )

        def fake_load(
            model: Path,
            *,
            progress_callback: object | None = None,
        ) -> FakePipeline:
            del model
            load_callbacks.append(progress_callback)
            return FakePipeline()

        monkeypatch.setattr(cli, "_load_generate_pipeline", fake_load)
        args = cli.build_parser().parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass observatory",
                "--output",
                str(output),
                "--progress",
                mode,
            ]
        )

        assert args.handler(args) == 0
        captured = capsys.readouterr()
        assert len(load_callbacks) == 1
        assert len(call_callbacks) == 1
        return [load_callbacks[0], call_callbacks[0]], captured.err

    callbacks, stderr = run_with_progress("always", tmp_path / "always.png")
    assert callbacks[0] is not None
    assert callbacks[1] is not None
    assert "[denoise 1/1] Denoising step" in stderr
    assert "[save] Saved" in stderr

    callbacks, stderr = run_with_progress("never", tmp_path / "never.png")
    assert callbacks == [None, None]
    assert stderr == ""


def test_generate_validation_fails_before_runtime_load(
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = run_cli(
        "generate",
        "--model",
        "artifact",
        "--prompt",
        "",
        "--width",
        "1024",
        "--height",
        "1024",
    )
    assert invalid.returncode == 1
    assert "prompt must be a non-empty string" in invalid.stderr
    assert "Traceback" not in invalid.stderr

    invalid = run_cli(
        "generate",
        "--model",
        "artifact",
        "--prompt",
        "a glass observatory",
        "--width",
        "1025",
        "--height",
        "1024",
    )
    assert invalid.returncode == 1
    assert "width must be a multiple of 16" in invalid.stderr
    assert "Traceback" not in invalid.stderr

    assert (
        cli.main(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass observatory",
                "--seeds",
                "nope",
            ]
        )
        == 1
    )
    stderr = capsys.readouterr().err
    assert "--seeds must be from 0" in stderr
    assert "Traceback" not in stderr

    invalid = run_cli(
        "generate",
        "--model",
        "artifact",
        "--prompt",
        "a glass observatory",
        "--output-template",
        "bad-{unknown}.png",
    )
    assert invalid.returncode == 1
    assert "unknown output template field" in invalid.stderr
    assert "Missing converted artifact metadata" not in invalid.stderr
    assert "Traceback" not in invalid.stderr


def test_manifest_output_path_uses_atomic_json_writer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model_index.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "nested" / "manifest.json"
    args = cli.build_parser().parse_args(
        ["manifest", "--source", str(source), "--output", str(output)]
    )

    assert args.handler(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "local_manifest_generated"


def _write_cli_source(root: Path) -> None:
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (root / "vae" / "config.json").write_text("{}", encoding="utf-8")
    write_safetensors_fixture(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        {"transformer_blocks.0.weight": ("BF16", [1])},
    )
    write_safetensors_fixture(
        root / "text_encoder" / "model.safetensors",
        {"language_model.layers.0.weight": ("BF16", [1])},
    )
    write_safetensors_fixture(
        root / "vae" / "diffusion_pytorch_model.safetensors",
        {"decoder.conv.weight": ("F32", [1])},
    )


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


class _FakeImage:
    def save(self, path: Path, *, format: str, pnginfo: object | None = None) -> None:
        assert format == "PNG"
        assert pnginfo is not None
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
