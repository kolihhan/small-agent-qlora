"""T4 QLoRA must not upcast enormous frozen Qwen embeddings to fp32."""
from gaia_small_agent.training.qlora import _prepare_kbit_model_memory_safe


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, dtype):
        return FakeTensor(dtype)


class FakeParameter:
    def __init__(self, dtype):
        self.data = FakeTensor(dtype)
        self.requires_grad = True

    @property
    def dtype(self):
        return self.data.dtype

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class FakeRMSNorm:
    def __init__(self):
        self.weight = FakeParameter("fp16")

    def parameters(self, recurse=False):
        return [self.weight]


class FakeModel:
    def __init__(self):
        self.embeddings = FakeParameter("fp16")
        self.norm = FakeRMSNorm()
        self.checkpoint_kwargs = None

    def parameters(self):
        return [self.embeddings, self.norm.weight]

    def modules(self):
        return [self, self.norm]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs


class FakeTorch:
    float16 = "fp16"
    bfloat16 = "bf16"
    float32 = "fp32"


def test_t4_preparation_preserves_large_frozen_weights_but_upcasts_norms():
    model = FakeModel()
    result = _prepare_kbit_model_memory_safe(model, FakeTorch, gradient_checkpointing=True)
    assert result is model
    assert model.embeddings.dtype == "fp16"
    assert model.norm.weight.dtype == "fp32"
    assert all(not p.requires_grad for p in model.parameters())
    assert model.checkpoint_kwargs == {"use_reentrant": False}


def test_t4_preparation_skips_checkpointing_when_disabled():
    model = FakeModel()
    _prepare_kbit_model_memory_safe(model, FakeTorch, gradient_checkpointing=False)
    assert model.checkpoint_kwargs is None


def test_t4_load_explicitly_converts_bf16_source_weights_to_fp16():
    from gaia_small_agent.training.qlora import _low_vram_load_kwargs

    assert _low_vram_load_kwargs(True, "fp16") == {"dtype": "fp16"}
    assert _low_vram_load_kwargs(False, "fp16") == {}


def test_t4_preflight_rejects_unquantized_bf16_before_backward():
    from gaia_small_agent.training.qlora import _assert_cuda_precision_compatible

    class NamedModel(FakeModel):
        def named_parameters(self):
            return [("embed_tokens.weight", self.embeddings), ("norm.weight", self.norm.weight)]

    model = NamedModel()
    model.embeddings = FakeParameter("bf16")
    import pytest
    with pytest.raises(RuntimeError, match="unsupported BF16"):
        _assert_cuda_precision_compatible(model, FakeTorch, cuda_bf16_supported=False)
    model.embeddings = FakeParameter("fp16")
    _assert_cuda_precision_compatible(model, FakeTorch, cuda_bf16_supported=False)
