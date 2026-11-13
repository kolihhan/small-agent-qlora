"""T4 Qwen3.5-9B preparation avoids upcasting giant frozen embeddings."""
from types import SimpleNamespace
import pytest

from gaia_small_agent.training.qlora import (
    _prepare_4bit_model_preserving_large_half_weights,
    _qlora_model_load_kwargs,
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


def test_t4_safe_loading_explicitly_sets_half_dtype_without_affecting_legacy():
    t4 = _qlora_model_load_kwargs(
        "Qwen/Qwen3.5-9B", quantization_config="nf4",
        memory_safe_half_weights=True, compute_dtype="fp16",
    )
    assert t4["dtype"] == "fp16"
    assert t4["quantization_config"] == "nf4"
    default = _qlora_model_load_kwargs(
        "Qwen/Qwen3.5-9B", quantization_config="nf4",
        memory_safe_half_weights=False, compute_dtype="fp16",
    )
    assert "dtype" not in default


def test_t4_uses_native_loss_instead_of_sm80_only_trl_fused_gemm():
    from gaia_small_agent.training.qlora import _training_runtime_defaults

    t4 = _training_runtime_defaults(cuda_available=True, cuda_bf16_supported=False)
    assert t4["compute_dtype_name"] == "float16"
    assert t4["trainer_loss_path"] == "native-t4"
    assert t4["use_gradient_checkpointing"] is True

    sm80 = _training_runtime_defaults(cuda_available=True, cuda_bf16_supported=True)
    assert sm80["trainer_loss_path"] == "trl-fused"
    assert sm80["compute_dtype_name"] == "bfloat16"

    cpu = _training_runtime_defaults(cuda_available=False, cuda_bf16_supported=False)
    assert cpu["trainer_loss_path"] == "native-cpu"
    assert cpu["use_gradient_checkpointing"] is False


def test_t4_native_trainer_args_preserve_optimization_and_strip_trl_hooks():
    from gaia_small_agent.training.qlora import _native_t4_training_args

    source = {
        "output_dir": "test-output",
        "learning_rate": 1e-4,
        "max_steps": 2,
        "save_steps": 1,
        "save_strategy": "steps",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_length": 1024,
        "completion_only_loss": False,
        "loss_type": "nll",
        "fp16": True,
        "bf16": False,
    }
    original = dict(source)
    actual = _native_t4_training_args(source)
    assert source == original
    assert actual["max_steps"] == 2
    assert actual["gradient_accumulation_steps"] == 1
    assert actual["fp16"] is True and actual["bf16"] is False
    assert actual["label_names"] == ["labels"]
    assert actual["remove_unused_columns"] is False
    assert actual["gradient_checkpointing"] is False
    assert "loss_type" not in actual
    assert "completion_only_loss" not in actual
    assert "max_length" not in actual


def test_native_bf16_gate_checks_real_cuda_arch_even_when_torch_emulates_bf16():
    from gaia_small_agent.training.qlora import _native_cuda_bf16_supported

    def fake_torch(*, available, major, reports_bf16):
        return SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: available,
            current_device=lambda: 0,
            get_device_capability=lambda idx: (major, 5 if major == 7 else 0),
            is_bf16_supported=lambda: reports_bf16,
        ))

    # The exact failure: torch may report BF16 support on an SM75 T4
    # via emulation, although TRL's BF16 backward GEMM requires SM80.
    assert not _native_cuda_bf16_supported(fake_torch(
        available=True, major=7, reports_bf16=True,
    ))
    assert not _native_cuda_bf16_supported(fake_torch(
        available=True, major=8, reports_bf16=False,
    ))
    assert _native_cuda_bf16_supported(fake_torch(
        available=True, major=8, reports_bf16=True,
    ))
    assert not _native_cuda_bf16_supported(fake_torch(
        available=False, major=8, reports_bf16=True,
    ))
