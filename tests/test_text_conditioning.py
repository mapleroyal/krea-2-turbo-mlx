from __future__ import annotations

from types import SimpleNamespace

import pytest

from krea_2_turbo_mlx.constants import (
    KREA_TEXT_PREFIX_DROP_INDEX,
    KREA_TEXT_PROMPT_PREFIX,
    KREA_TEXT_PROMPT_SUFFIX,
    KREA_TEXT_SUFFIX_TOKEN_COUNT,
)
from krea_2_turbo_mlx.text_conditioning import (
    build_prompt_token_layout,
    encode_prompt,
)


def test_prompt_layout_uses_fixed_template_and_suffix_after_padding() -> None:
    tokenizer = FakeTokenizer()

    layout = build_prompt_token_layout(
        "x",
        tokenizer=tokenizer,
        max_sequence_length=10,
    )

    assert tokenizer.calls[0]["texts"] == [KREA_TEXT_PROMPT_PREFIX + "x"]
    assert tokenizer.calls[0]["truncation"] is False
    assert tokenizer.calls[1]["texts"] == [KREA_TEXT_PROMPT_PREFIX + "x"]
    assert tokenizer.calls[1]["truncation"] is True
    assert tokenizer.calls[1]["padding"] == "max_length"
    assert tokenizer.calls[1]["max_length"] == 39
    assert tokenizer.calls[2]["texts"] == [KREA_TEXT_PROMPT_SUFFIX]
    assert layout.prompt_token_max_length == 39
    assert layout.truncation_warnings == ()

    row = layout.input_ids[0]
    mask = layout.attention_mask[0]
    positions = layout.position_ids[0][0]
    assert len(row) == 44
    assert row[KREA_TEXT_PREFIX_DROP_INDEX] == 200
    assert row[35:39] == [0, 0, 0, 0]
    assert row[39:] == [900, 901, 902, 903, 904]
    assert mask[35:39] == [False, False, False, False]
    assert mask[39:] == [True] * KREA_TEXT_SUFFIX_TOKEN_COUNT
    assert positions[35:39] == [34, 34, 34, 34]
    assert positions[39:] == [35, 36, 37, 38, 39]


def test_prompt_layout_allows_empty_string_and_batches() -> None:
    layout = build_prompt_token_layout(
        ["", "yz"],
        tokenizer=FakeTokenizer(),
        max_sequence_length=8,
    )

    assert len(layout.input_ids) == 2
    assert len(layout.input_ids[0]) == 42
    assert layout.input_ids[0][34:37] == [0, 0, 0]
    assert layout.input_ids[0][37:] == [900, 901, 902, 903, 904]
    assert layout.input_ids[1][34:36] == [200, 201]
    assert layout.input_ids[1][36] == 0


def test_prompt_layout_reports_truncation_before_tokenizer_clips() -> None:
    layout = build_prompt_token_layout(
        "abcdef",
        tokenizer=FakeTokenizer(),
        max_sequence_length=1,
    )

    warning = layout.truncation_warnings[0]
    assert warning.prompt_index == 0
    assert warning.token_count == KREA_TEXT_PREFIX_DROP_INDEX + 6
    assert warning.max_length == 30
    assert warning.truncated_tokens == 10


def test_encode_prompt_stacks_selected_layers_slices_prefix_and_repeats() -> None:
    mx = pytest.importorskip("mlx.core")
    np = pytest.importorskip("numpy")
    encoder = FakeEncoder(mx)

    conditioning = encode_prompt(
        ["a", "bc"],
        tokenizer=FakeTokenizer(),
        encoder=encoder,
        max_sequence_length=10,
        num_images_per_prompt=2,
        select_layers=(2, 5),
    )
    mx.eval(
        conditioning.hidden_states,
        conditioning.attention_mask,
        conditioning.input_ids,
        conditioning.position_ids,
    )

    assert conditioning.hidden_states.shape == (4, 10, 2, 3)
    assert conditioning.attention_mask.shape == (4, 10)
    assert conditioning.input_ids.shape == (4, 44)
    assert conditioning.position_ids.shape == (3, 4, 44)
    assert encoder.input_ids.shape == (2, 44)
    assert encoder.position_ids.shape == (3, 2, 44)

    hidden = np.array(conditioning.hidden_states)
    assert hidden[0, 0, 0, 0] == 2
    assert hidden[0, 0, 1, 0] == 5
    mask = np.array(conditioning.attention_mask)
    assert mask[0].tolist() == [True, False, False, False, False, True, True, True, True, True]


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool | None = None,
        padding: str | None = None,
        max_length: int | None = None,
        return_tensors: object | None = None,
    ) -> dict[str, list[list[int]]]:
        self.calls.append(
            {
                "texts": texts,
                "truncation": truncation,
                "padding": padding,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        if all(text == KREA_TEXT_PROMPT_SUFFIX for text in texts):
            return {
                "input_ids": [[900, 901, 902, 903, 904] for _ in texts],
                "attention_mask": [[1, 1, 1, 1, 1] for _ in texts],
            }

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for text in texts:
            assert text.startswith(KREA_TEXT_PROMPT_PREFIX)
            prompt = text.removeprefix(KREA_TEXT_PROMPT_PREFIX)
            raw = list(range(100, 100 + KREA_TEXT_PREFIX_DROP_INDEX))
            raw.extend(200 + index for index, _ in enumerate(prompt))
            if truncation and max_length is not None:
                raw = raw[:max_length]
            mask = [1] * len(raw)
            if padding == "max_length" and max_length is not None:
                pad_count = max_length - len(raw)
                raw = raw + [0] * pad_count
                mask = mask + [0] * pad_count
            input_ids.append(raw)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeEncoder:
    def __init__(self, mx: object) -> None:
        self.mx = mx
        self.input_ids = None
        self.position_ids = None

    def __call__(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        position_ids: object,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True
        self.input_ids = input_ids
        self.position_ids = position_ids
        batch, seq_len = input_ids.shape
        hidden_states = tuple(
            self.mx.full((batch, seq_len, 3), layer, dtype=self.mx.float32)
            for layer in range(6)
        )
        return SimpleNamespace(hidden_states=hidden_states)
