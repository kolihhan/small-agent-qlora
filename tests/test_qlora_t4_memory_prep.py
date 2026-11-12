"""T4 Qwen3.5-9B preparation avoids upcasting giant frozen embeddings."""
from types import SimpleNamespace
import pytest

from gaia_small_agent.training.qlora import (
    _prepare_4bit_model_preserving_large_half_weights,
)


class FakeTensor:
    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count

    def to(self, dtype):
        return FakeTensor(dtype, self.count)


class FakeParam:
    def __init__(self, dtype, count):
        self.data = FakeTensor(dtype, count)
        self.requires_grad = True

    @property
    def dtype(self):
        return self.data.dtype

    def numel(self):
        return self.data.count


class Params4bit(FakeParam):
    pass


class FakeModel:
    def __init__(self, params):
        self._params = params
        self.input_grads_enabled = False
        self.checkpointing_enabled = False

    def parameters(self):
        return iter(self._params)

    def enable_input_require_grads(self):
        self.input_grads_enabled = True

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True


DTYPES = SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32")


def test_t4_prep_freezes_backbone_preserves_large_half_and_enables_checkpointing():
    huge_embedding = FakeParam("fp16", 950_000_000)
    small_norm = FakeParam("bf16", 4096)
    quant_weight = Params4bit("fp16", 1234)
    model = FakeModel([huge_embedding, small_norm, quant_weight])
    result = _prepare_4bit_model_preserving_large_half_weights(
        model, torch=DTYPES, use_gradient_checkpointing=True
    )
    assert result is model
    assert huge_embedding.dtype == "fp16"
    assert small_norm.dtype == "fp32"
    assert quant_weight.dtype == "fp16"
    assert all(not p.requires_grad for p in model.parameters())
    assert model.input_grads_enabled
    assert model.checkpointing_enabled


def test_t4_prep_can_disable_checkpointing_and_reject_invalid_threshold():
    model = FakeModel([FakeParam("fp16", 4)])
    _prepare_4bit_model_preserving_large_half_weights(
        model, torch=DTYPES, use_gradient_checkpointing=False
    )
    assert not model.input_grads_enabled
    assert not model.checkpointing_enabled
    with pytest.raises(ValueError, match="max_upcast_elements"):
        _prepare_4bit_model_preserving_large_half_weights(
            model, torch=DTYPES, use_gradient_checkpointing=True,
            max_upcast_elements=0,
        )
