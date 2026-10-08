from contextlib import nullcontext
import itertools

import pytest

from gaia_small_agent.agent.types import ModelRuntimeError
from gaia_small_agent.model.transformers_qwen import TransformersQwenModel, parse_qwen_turn


def test_parse_native_qwen_tool_call():
    text = "<tool_call>\n<function=python>\n<parameter=code>\nprint(6*7)\n</parameter>\n</function>\n</tool_call>"
    turn = parse_qwen_turn(text)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "python"
    assert turn.tool_calls[0].arguments == {"code": "print(6*7)"}


def test_parse_native_qwen_tool_call_accepts_json_parameter():
    text = '<tool_call>\n<function=inspect>\n<parameter=path>\n"report.xlsx"\n</parameter>\n</function>\n</tool_call>'
    turn = parse_qwen_turn(text)
    assert turn.tool_calls[0].arguments == {"path": "report.xlsx"}


def test_parse_final_text_removes_think_block():
    turn = parse_qwen_turn("<think>private reasoning</think>\n\n42")
    assert turn.content == "42"
    assert turn.tool_calls == []


def test_transformers_complete_maps_non_capacity_failure_to_model_error():
    class FakeTensor:
        def to(self, device):
            return self

    class FakeParameter:
        device = "cpu"

    class FakeModel:
        def parameters(self):
            return iter([FakeParameter()])

        def generate(self, **kwargs):
            raise RuntimeError("generation exploded")

    class FakeTokenizer:
        eos_token_id = 1
        pad_token_id = 1

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": FakeTensor()}

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    model = object.__new__(TransformersQwenModel)
    model.torch = FakeTorch()
    model.tokenizer = FakeTokenizer()
    model.model = FakeModel()
    model.enable_thinking = False
    model.max_new_tokens = 8
    model.cache_implementation = None
    model._ids = itertools.count(1)

    with pytest.raises(ModelRuntimeError) as exc_info:
        model.complete([], [])

    assert exc_info.value.stop_reason == "model_error"
    assert "generation exploded" in exc_info.value.message
