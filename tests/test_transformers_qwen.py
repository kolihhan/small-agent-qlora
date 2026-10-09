import pytest
import sys
import types

from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
from gaia_small_agent.agent.types import ModelCapacityError


def _patch_constructor_imports(monkeypatch, cuda_available):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_transformers = types.ModuleType("transformers")

    class Tokenizer:
        pass

    class LoadedModel:
        def eval(self):
            return self

    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: LoadedModel())
    fake_transformers.BitsAndBytesConfig = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


class FakeTensor:
    device = "cpu"
    shape = (1, 1)

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self


class FakeCuda:
    def __init__(self, available):
        self.available = available
        self.empty_cache_calls = 0

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def inference_mode(self):
        return self._InferenceMode()


class FakeProcessor:
    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 2

        def decode(self, ids, skip_special_tokens=False):
            return "done"

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": FakeTensor()}


class FakeModel:
    def __init__(self, error=None):
        self.error = error
        self.generate_kwargs = None

    def parameters(self):
        return iter([FakeTensor()])

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if self.error:
            raise self.error
        return [FakeTensor()]


def _model(*, cuda_available=True, generation_error=None, cache_implementation=None):
    model = object.__new__(TransformersQwenModel)
    model.torch = FakeTorch(cuda_available)
    model.tokenizer = FakeProcessor.Tokenizer()
    model.model = FakeModel(generation_error)
    model.cache_implementation = cache_implementation
    model.enable_thinking = False
    model.max_new_tokens = 4
    return model


def test_complete_does_not_force_cuda_cache_clear_on_success():
    model = _model()

    model.complete([], [])

    assert model.torch.cuda.empty_cache_calls == 0


def test_complete_does_not_force_cuda_cache_clear_on_generation_error():
    model = _model(generation_error=RuntimeError("generation failed"), cache_implementation="offloaded")

    with pytest.raises(RuntimeError, match="generation failed"):
        model.complete([], [])

    assert model.torch.cuda.empty_cache_calls == 0


def test_runtime_uses_text_only_causal_lm_loader():
    import inspect
    import gaia_small_agent.model.transformers_qwen as module
    source = inspect.getsource(module.TransformersQwenModel.__init__)
    assert "AutoModelForCausalLM" in source
    assert "AutoModelForMultimodalLM" not in source
    assert "AutoTokenizer" in source


def test_cpu_quantized_runtime_uses_float32_compute():
    from gaia_small_agent.model.transformers_qwen import _quant_compute_dtype

    class Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_bf16_supported():
            return False

    class Torch:
        cuda = Cuda()
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

    assert _quant_compute_dtype(Torch()) == "float32"


def test_complete_skips_cuda_cache_when_cuda_unavailable():
    model = _model(cuda_available=False)

    model.complete([], [])

    assert model.torch.cuda.empty_cache_calls == 0


def test_complete_forwards_offloaded_cache_with_existing_generation_kwargs():
    model = _model(cache_implementation="offloaded")

    model.complete([], [])

    assert model.model.generate_kwargs == {
        "input_ids": model.model.generate_kwargs["input_ids"],
        "max_new_tokens": 4,
        "do_sample": False,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "cache_implementation": "offloaded",
    }


def test_complete_omits_cache_implementation_when_non_cuda():
    model = _model(cuda_available=False, cache_implementation=None)

    model.complete([], [])

    assert "cache_implementation" not in model.model.generate_kwargs


@pytest.mark.parametrize(("cuda_available", "expected"), [(True, None), (False, None)])
def test_constructor_sets_effective_cache_implementation_without_loading_weights(monkeypatch, cuda_available, expected):
    _patch_constructor_imports(monkeypatch, cuda_available)

    model = TransformersQwenModel(quantize_4bit=False)

    assert model.cache_implementation == expected


def test_complete_translates_cuda_capacity_error_without_secret_render(monkeypatch):
    class CapacityError(RuntimeError):
        pass

    model = _model(generation_error=CapacityError("CUDA out of memory; secret=token"), cache_implementation="offloaded")
    model.torch.OutOfMemoryError = CapacityError
    model.torch.cuda.OutOfMemoryError = CapacityError

    with pytest.raises(ModelCapacityError) as caught:
        model.complete([], [])

    assert str(caught.value) == ""
    assert caught.value.__cause__ is None
    assert model.torch.cuda.empty_cache_calls == 0


def test_complete_does_not_translate_capacity_marker_without_cuda():
    class CapacityError(RuntimeError):
        pass

    model = _model(cuda_available=False, generation_error=CapacityError("out of memory; secret=token"))
    model.torch.OutOfMemoryError = CapacityError
    model.torch.cuda.OutOfMemoryError = CapacityError

    with pytest.raises(CapacityError, match="secret=token"):
        model.complete([], [])


def test_complete_translates_marked_accelerator_error_only():
    class AcceleratorError(RuntimeError):
        pass

    model = _model(generation_error=AcceleratorError("memory allocation failed; secret=token"))
    model.torch.AcceleratorError = AcceleratorError

    with pytest.raises(ModelCapacityError):
        model.complete([], [])

    unrelated = _model(generation_error=AcceleratorError("unrelated failure"))
    unrelated.torch.AcceleratorError = AcceleratorError
    with pytest.raises(AcceleratorError, match="unrelated failure"):
        unrelated.complete([], [])
