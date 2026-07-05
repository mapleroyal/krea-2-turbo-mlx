from __future__ import annotations

from argparse import Namespace

import pytest

from krea_2_turbo_mlx import cli
from krea_2_turbo_mlx.errors import ValidationError
from krea_2_turbo_mlx.generation_validation import (
    validate_generation_dimension,
    validate_generation_dimensions,
)
from krea_2_turbo_mlx.pipeline import KreaTurboPipeline


def test_validate_generation_dimensions_returns_normalized_ints() -> None:
    class IntLike:
        def __index__(self) -> int:
            return 1024

    width, height = validate_generation_dimensions(IntLike(), 512)

    assert (width, height) == (1024, 512)
    assert type(width) is int
    assert type(height) is int


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (True, "width must be an integer"),
        ("1024", "width must be an integer"),
        (0, "width must be a positive integer"),
        (-16, "width must be a positive integer"),
        (1025, "width must be a multiple of 16"),
        (2064, "width must be 2048 or smaller"),
    ],
)
def test_validate_generation_dimension_rejects_invalid_values(
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        validate_generation_dimension(value, "width")


def test_cli_generate_validation_uses_shared_size_limit(capsys: pytest.CaptureFixture[str]) -> None:
    status = cli.main(
        [
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass observatory",
            "--width",
            "2064",
            "--height",
            "1024",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "width must be 2048 or smaller" in captured.err
    assert "Missing converted artifact metadata" not in captured.err


def test_cli_generate_validation_rejects_bool_dimensions() -> None:
    args = Namespace(
        prompt=["a glass observatory"],
        width=True,
        height=1024,
        steps=8,
        guidance_scale=0.0,
        output=None,
        output_template="image-{seed}.png",
        model="artifact",
        lora=[],
        seed=None,
    )

    with pytest.raises(ValidationError, match="width must be an integer"):
        cli._validate_generate_args(args)


def test_pipeline_generation_rejects_invalid_dimensions_before_mlx() -> None:
    pipe = KreaTurboPipeline(
        scheduler=object(),
        text_conditioner=object(),
        transformer=object(),
        vae=object(),
    )

    with pytest.raises(ValidationError, match="width must be 2048 or smaller"):
        pipe("a glass observatory", width=2064, height=1024)
