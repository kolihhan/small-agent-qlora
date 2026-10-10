import pytest

from gaia_small_agent.training.qlora import _tokenize_completion_examples


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) for ch in text]}


def test_completion_preserving_tokenization_left_trims_prompt_only():
    rows = _tokenize_completion_examples(
        FakeTokenizer(),
        [{"prompt": "abcdefgh", "completion": "XYZ"}],
        max_length=6,
    )

    assert rows == [
        {
            "input_ids": [ord(ch) for ch in "fghXYZ"],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, ord("X"), ord("Y"), ord("Z")],
        }
    ]


def test_completion_preserving_tokenization_keeps_full_short_example():
    rows = _tokenize_completion_examples(
        FakeTokenizer(),
        [{"prompt": "ab", "completion": "CD"}],
        max_length=8,
    )

    assert rows[0]["input_ids"] == [ord(ch) for ch in "abCD"]
    assert rows[0]["labels"] == [-100, -100, ord("C"), ord("D")]


def test_completion_preserving_tokenization_rejects_completion_that_cannot_fit():
    with pytest.raises(ValueError, match="completion exceeds max_length"):
        _tokenize_completion_examples(
            FakeTokenizer(),
            [{"prompt": "ab", "completion": "TOO-LONG"}],
            max_length=4,
        )


def test_completion_preserving_tokenization_rejects_empty_completion():
    with pytest.raises(ValueError, match="completion tokenization is empty"):
        _tokenize_completion_examples(
            FakeTokenizer(),
            [{"prompt": "ab", "completion": ""}],
            max_length=4,
        )
