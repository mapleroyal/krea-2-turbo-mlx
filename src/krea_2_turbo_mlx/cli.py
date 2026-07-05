from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from . import __version__
from .constants import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DISTILLED_SHIFT,
    DEFAULT_GENERATION_HEIGHT,
    DEFAULT_GENERATION_STEPS,
    DEFAULT_GENERATION_WIDTH,
    DEFAULT_GUI_PORT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LORA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_TEMPLATE,
    DEFAULT_SOURCE_DIR,
    MAX_GENERATION_SEED,
    OFFICIAL_HF_REPO_ID,
    PROJECT_NAME,
)
from .convert import build_conversion_plan, run_conversion
from .doctor import format_doctor_report, run_doctor
from .errors import Krea2TurboMlxError, ValidationError
from .generation_validation import validate_generation_dimensions
from .hf import (
    DOWNLOAD_ALLOW_PATTERNS,
    DOWNLOAD_IGNORE_PATTERNS,
    build_download_plan,
    download_source,
)
from .json_io import write_json
from .lora import (
    ResolvedLoraPatch,
    lora_metadata,
    resolve_lora_patches,
    validate_lora_spec_syntax,
)
from .manifest import generate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROJECT_NAME,
        description="MLX-native Krea 2 Turbo tooling.",
    )
    parser.add_argument("--version", action="store_true", help="Print the package version.")

    subparsers = parser.add_subparsers(dest="command")

    setup = subparsers.add_parser(
        "setup",
        help="Configure, download, convert, and validate a local Krea 2 Turbo artifact.",
    )
    setup.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Project-local setup config path.",
    )
    setup.add_argument("--source", help="Hugging Face source repo to download.")
    setup.add_argument(
        "--revision",
        help="Pinned Hugging Face revision. The official source uses the audited revision by default.",
    )
    setup.add_argument("--source-dir", type=Path, help="Local official source folder.")
    setup.add_argument("--artifact-dir", type=Path, help="Converted MLX artifact folder.")
    setup.add_argument("--output-dir", type=Path, help="Default generated image folder.")
    setup.add_argument("--lora-dir", type=Path, help="Default local LoRA folder.")
    source_cleanup = setup.add_mutually_exclusive_group()
    source_cleanup.add_argument(
        "--cleanup-source",
        dest="cleanup_source",
        action="store_true",
        default=None,
        help="Remove a project-local source folder after artifact validation.",
    )
    source_cleanup.add_argument(
        "--keep-source",
        dest="cleanup_source",
        action="store_false",
        help="Keep the downloaded source folder after setup.",
    )
    setup.add_argument(
        "--accept-defaults",
        action="store_true",
        help="Use saved/default choices without opening the setup page.",
    )
    setup.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the setup URL without opening a browser.",
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the setup plan without downloading or converting.",
    )
    setup.set_defaults(handler=_run_setup)

    gui = subparsers.add_parser(
        "gui",
        help="Open the local browser GUI for Krea 2 Turbo generation.",
    )
    gui.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GUI_PORT,
        help=f"Port for the local GUI. Defaults to {DEFAULT_GUI_PORT}; use 0 to pick a random free port.",
    )
    gui.add_argument("--no-browser", action="store_true", help="Print the GUI URL without opening a browser.")
    gui.add_argument("--no-preload", action="store_true", help="Start the GUI without loading the model immediately.")
    gui.add_argument(
        "--unsafe-host",
        action="store_true",
        help="Allow binding and Host headers outside loopback addresses.",
    )
    gui.set_defaults(handler=_run_gui)

    manifest = subparsers.add_parser(
        "manifest",
        help="Inspect a Krea 2 Turbo source without loading tensor payloads.",
    )
    manifest.add_argument("--source", required=True, help="Diffusers source directory or HF model id.")
    manifest.add_argument(
        "--revision",
        help="Pinned Hugging Face revision. The official source defaults to the audited revision.",
    )
    manifest.add_argument("--output", type=Path, help="Optional JSON output path. Defaults to stdout.")
    manifest.set_defaults(handler=_run_manifest)

    download = subparsers.add_parser(
        "download",
        help="Download the official Krea 2 Turbo Diffusers source into models/.",
    )
    download.add_argument("--source", default=OFFICIAL_HF_REPO_ID, help="Hugging Face model id or local source.")
    download.add_argument(
        "--revision",
        help="Pinned Hugging Face revision. The official source defaults to the audited revision.",
    )
    download.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_SOURCE_DIR.parent,
        help="Destination root for Hugging Face snapshots.",
    )
    download.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    download.set_defaults(handler=_run_download)

    convert = subparsers.add_parser(
        "convert",
        help="Build a dry plan or write a local full-precision artifact.",
    )
    convert.add_argument("--source", required=True, help="Diffusers source directory or HF model id.")
    convert.add_argument(
        "--revision",
        help="Pinned Hugging Face revision when --source is a model id.",
    )
    convert.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="MLX artifact output directory.",
    )
    convert.add_argument("--dry-run", action="store_true", help="Print the exact conversion plan and write nothing.")
    convert.add_argument(
        "--source-dir",
        type=Path,
        help="Local download/reuse path when --source is a Hugging Face model id.",
    )
    convert.set_defaults(handler=_run_convert)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check the local environment, downloaded source, and converted model files.",
    )
    doctor.add_argument("--source", type=Path, help="Local Krea 2 Turbo source directory to validate.")
    doctor.add_argument("--model", type=Path, help="Converted MLX artifact directory to validate.")
    doctor.add_argument(
        "--runtime",
        action="store_true",
        help="Require runtime dependencies needed by setup, GUI, and generation.",
    )
    doctor.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    doctor.set_defaults(handler=_run_doctor)

    generate = subparsers.add_parser(
        "generate",
        help="Generate Krea 2 Turbo images from a converted full-precision artifact.",
    )
    generate.add_argument("--model", required=True, help="Converted MLX artifact directory.")
    generate.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt. Repeat to generate a sequential prompt batch.",
    )
    generate.add_argument("--width", type=_parse_positive_int, default=DEFAULT_GENERATION_WIDTH)
    generate.add_argument("--height", type=_parse_positive_int, default=DEFAULT_GENERATION_HEIGHT)
    generate.add_argument("--seed", type=_parse_seed)
    generate.add_argument("--seeds", help="Comma-separated per-image seeds for batch generation.")
    generate.add_argument("--steps", type=_parse_positive_int, default=DEFAULT_GENERATION_STEPS)
    generate.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=(
            f"Fixed at {DEFAULT_GUIDANCE_SCALE} for Krea 2 Turbo; only the default "
            "value is accepted."
        ),
    )
    generate.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="VALUE",
        help=(
            "Apply a LoRA patch. Accepts a catalog id from --lora-dir or a "
            ".safetensors path, each with optional :scale."
        ),
    )
    generate.add_argument(
        "--lora-dir",
        type=Path,
        default=DEFAULT_LORA_DIR,
        help="Directory containing local .safetensors LoRA adapters.",
    )
    generate.add_argument("--num-images", type=_parse_positive_int, default=1)
    generate.add_argument("--output", type=Path, help="PNG output path for single-image generation.")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate.add_argument("--output-template", default=DEFAULT_OUTPUT_TEMPLATE)
    generate.add_argument("--overwrite", action="store_true", help="Replace an existing explicit --output path.")
    generate.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    generate.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Progress reporting mode; progress is written to stderr.",
    )
    generate.set_defaults(handler=_run_generate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    try:
        return int(args.handler(args) or 0)
    except (Krea2TurboMlxError, ValueError, NotImplementedError) as exc:
        print(f"{PROJECT_NAME}: {exc}", file=sys.stderr)
        return 1


def _run_manifest(args: argparse.Namespace) -> int:
    manifest = generate_manifest(args.source, revision=args.revision)
    if args.output:
        write_json(args.output, manifest)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _run_setup(args: argparse.Namespace) -> int:
    from .setup_flow import run_setup_cli

    return run_setup_cli(args, stream=sys.stderr)


def _run_gui(args: argparse.Namespace) -> int:
    from .gui import run_gui

    _raise_runtime_doctor_errors()
    return run_gui(
        config_path=args.config,
        host=args.host,
        port=args.port,
        allow_unsafe_host=args.unsafe_host,
        open_browser=not args.no_browser,
        preload=not args.no_preload,
    )


def _run_download(args: argparse.Namespace) -> int:
    plan = build_download_plan(args.source, revision=args.revision, dest=args.dest)
    local_path = download_source(
        args.source,
        revision=args.revision,
        dest_root=args.dest,
    )
    resolved_path = Path(local_path).expanduser().resolve()
    if args.json:
        payload = dict(plan)
        payload.update(
            {
                "status": "source_ready",
                "path": str(resolved_path),
                "allow_patterns": list(DOWNLOAD_ALLOW_PATTERNS),
                "ignore_patterns": list(DOWNLOAD_IGNORE_PATTERNS),
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(resolved_path)
    return 0


def _run_convert(args: argparse.Namespace) -> int:
    if args.dry_run:
        plan = build_conversion_plan(args.source, revision=args.revision, output=args.output)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    result = run_conversion(
        args.source,
        revision=args.revision,
        output=args.output,
        source_dir=args.source_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(
        model=args.model,
        source=args.source,
        runtime_required=args.runtime,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_doctor_report(report), end="")
    return 1 if report.get("status") == "error" else 0


def _run_generate(args: argparse.Namespace) -> int:
    _validate_generate_args(args)
    _raise_runtime_doctor_errors(model=args.model)
    prompts = [prompt for prompt in args.prompt]
    total_count = len(prompts) * args.num_images
    seeds = _resolve_batch_seed_args(args, total_count)
    single_mode = total_count == 1
    if args.output is not None and not single_mode:
        raise ValidationError("--output is only valid for single-image generation")

    progress_callback = _progress_callback(args.progress)
    pipeline = _load_generate_pipeline(
        Path(args.model).expanduser(),
        progress_callback=progress_callback,
    )
    lora_patches = _resolve_generate_loras(args.lora, pipeline, lora_dir=args.lora_dir)
    active_lora_metadata = lora_metadata(lora_patches)
    outputs: list[dict[str, object]] = []
    batch_index = 0
    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(args.num_images):
            seed = seeds[batch_index]
            started = time.perf_counter()
            generate_kwargs = {
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": seed,
                "progress_callback": progress_callback,
            }
            if lora_patches:
                generate_kwargs["loras"] = lora_patches
            result = pipeline(prompt, **generate_kwargs)
            elapsed = (
                result.elapsed_seconds
                if getattr(result, "elapsed_seconds", None) is not None
                else time.perf_counter() - started
            )
            output = _resolve_generation_output(
                args,
                result_seed=int(result.seed),
                prompt_index=prompt_index,
                image_index=image_index,
                batch_index=batch_index,
                explicit_output=args.output is not None,
            )
            metadata = _generation_metadata(
                prompt=prompt,
                seed=int(result.seed),
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                model=args.model,
                elapsed_seconds=elapsed,
                truncation_warnings=tuple(
                    getattr(result, "truncation_warnings", ()) or ()
                ),
                loras=active_lora_metadata,
            )
            _emit_save_progress(progress_callback, output, start=True)
            _save_generated_png(
                result.images,
                output,
                metadata=metadata,
                overwrite=args.overwrite if args.output is not None else True,
            )
            _emit_save_progress(progress_callback, output, start=False)
            output_record = {
                "status": "image_generated",
                "output": str(output.resolve()),
                "prompt": prompt,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "shift": DEFAULT_DISTILLED_SHIFT,
                "seed": int(result.seed),
                "elapsed_seconds": elapsed,
                "prompt_truncation": metadata["prompt_truncation"],
            }
            if active_lora_metadata:
                output_record["loras"] = active_lora_metadata
            outputs.append(output_record)
            batch_index += 1

    if args.json:
        payload: dict[str, object] = {
            "status": "images_generated" if len(outputs) > 1 else "image_generated",
            "count": len(outputs),
            "outputs": outputs,
        }
        if len(outputs) == 1:
            payload.update(outputs[0])
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in outputs:
            print(
                "Saved "
                f"{item['output']} "
                f"({item['width']}x{item['height']}, seed {item['seed']}, "
                f"steps {item['steps']}, {float(item['elapsed_seconds']):.2f}s)"
            )
    return 0


def _validate_generate_args(args: argparse.Namespace) -> None:
    for prompt in args.prompt:
        if not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")
    args.width, args.height = validate_generation_dimensions(args.width, args.height)
    if args.steps <= 0:
        raise ValidationError("steps must be a positive integer")
    if args.guidance_scale != DEFAULT_GUIDANCE_SCALE:
        raise ValidationError(
            f"Krea 2 Turbo runs at guidance_scale={DEFAULT_GUIDANCE_SCALE} only; "
            "the --guidance-scale flag is fixed at that value."
        )
    if args.output is not None and args.output.suffix.lower() != ".png":
        raise ValidationError("output path must end in .png")
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise ValidationError(
            f"output already exists: {args.output}. Pass --overwrite to replace it."
        )
    if args.output is None:
        _validate_output_template(args.output_template)
    if args.model:
        Path(args.model).expanduser()
    validate_lora_spec_syntax(args.lora)
    if args.seed is None:
        return
    if not 0 <= args.seed <= MAX_GENERATION_SEED:
        raise ValidationError(f"seed must be from 0 to {MAX_GENERATION_SEED}")


def _load_generate_pipeline(model: Path, *, progress_callback=None):
    from .pipeline import KreaTurboPipeline

    return KreaTurboPipeline.from_artifact(model, progress_callback=progress_callback)


def _resolve_generate_loras(
    specs: list[str],
    pipeline: object,
    *,
    lora_dir: Path = DEFAULT_LORA_DIR,
) -> tuple[ResolvedLoraPatch, ...]:
    if not specs:
        return ()
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValidationError("LoRA generation requires a loaded transformer.")
    return resolve_lora_patches(
        specs,
        transformer=transformer,
        lora_dir=lora_dir,
    )


def _save_generated_png(
    images,
    output: Path,
    *,
    metadata: dict[str, object],
    overwrite: bool,
) -> None:
    from .png import save_generation_png

    save_generation_png(images, output, metadata=metadata, overwrite=overwrite)


def _generation_metadata(
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    model: str | Path,
    elapsed_seconds: float | None,
    truncation_warnings: tuple[dict[str, int], ...],
    loras: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    from .png import generation_metadata_payload

    return generation_metadata_payload(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        shift=DEFAULT_DISTILLED_SHIFT,
        model_path=model,
        elapsed_seconds=elapsed_seconds,
        truncation_warnings=truncation_warnings,
        loras=loras or (),
    )


def _resolve_generation_output(
    args: argparse.Namespace,
    *,
    result_seed: int,
    prompt_index: int,
    image_index: int,
    batch_index: int,
    explicit_output: bool,
) -> Path:
    if explicit_output:
        return Path(args.output).expanduser()
    try:
        filename = args.output_template.format(
            seed=result_seed,
            prompt_index=prompt_index,
            image_index=image_index,
            batch_index=batch_index,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValidationError(_output_template_error_message(exc)) from exc
    # Trusted local CLI input: template path traversal hardening is deferred for
    # pre-release source-checkout usage.
    path = Path(args.output_dir).expanduser() / filename
    if path.suffix.lower() != ".png":
        raise ValidationError("output template must produce a .png filename")
    return path if args.overwrite else _unique_output_path(path)


def _validate_output_template(template: str) -> None:
    try:
        filename = template.format(
            seed=0,
            prompt_index=0,
            image_index=0,
            batch_index=0,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValidationError(_output_template_error_message(exc)) from exc
    # Trusted local CLI input: this validates shape/suffix only; containment
    # hardening for template-derived paths is intentionally deferred.
    if Path(filename).suffix.lower() != ".png":
        raise ValidationError("output template must produce a .png filename")


def _output_template_error_message(exc: Exception) -> str:
    if isinstance(exc, KeyError):
        return f"unknown output template field: {exc}"
    if isinstance(exc, IndexError):
        return "output template must use named fields"
    return f"invalid output template: {exc}"


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValidationError(f"could not choose a unique output path for {path}")


def _resolve_batch_seed_args(args: argparse.Namespace, total_count: int) -> list[int | None]:
    if args.seed is not None and args.seeds is not None:
        raise ValidationError("provide either --seed or --seeds, not both")
    if args.seeds is not None:
        seeds = _parse_seed_list(args.seeds)
        if len(seeds) != total_count:
            raise ValidationError(f"--seeds must provide {total_count} seed(s), got {len(seeds)}")
        return seeds
    if args.seed is None:
        return [None] * total_count
    return [int(args.seed + index) & MAX_GENERATION_SEED for index in range(total_count)]


def _parse_seed_list(value: str) -> list[int]:
    if not value.strip():
        raise ValidationError("--seeds must not be empty")
    try:
        return [_parse_seed(item.strip()) for item in value.split(",")]
    except argparse.ArgumentTypeError as exc:
        raise ValidationError(f"--seeds {exc}") from exc


def _progress_callback(mode: str):
    if mode == "never":
        return None
    if mode == "auto" and not sys.stderr.isatty():
        return None

    def callback(event) -> None:
        step = ""
        if event.step_index is not None and event.step_count is not None:
            step = f" {event.step_index + 1}/{event.step_count}"
        print(f"[{event.stage}{step}] {event.message}", file=sys.stderr, flush=True)

    return callback


def _emit_save_progress(callback, output: Path, *, start: bool) -> None:
    if callback is None:
        return
    from .pipeline import PipelineProgressEvent

    callback(
        PipelineProgressEvent(
            "save_start" if start else "save_end",
            "save",
            f"{'Saving' if start else 'Saved'} {output}",
            progress=0.99 if start else 1.0,
        )
    )


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be from 0 to {MAX_GENERATION_SEED}"
        ) from exc
    if not 0 <= parsed <= MAX_GENERATION_SEED:
        raise argparse.ArgumentTypeError(f"must be from 0 to {MAX_GENERATION_SEED}")
    return parsed


def _raise_runtime_doctor_errors(*, model: str | Path | None = None) -> None:
    report = run_doctor(model=model, runtime_required=True)
    errors = [
        f"{check.get('name')}: {check.get('message')}"
        for check in report.get("checks", [])
        if check.get("status") == "error"
    ]
    if errors:
        raise Krea2TurboMlxError("; ".join(errors))


__all__ = ["build_parser", "main"]
