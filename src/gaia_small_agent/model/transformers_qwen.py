from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

from ..agent.types import AssistantTurn, ModelCapacityError, ModelRuntimeError, ToolCall

_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def _decode_param(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_qwen_turn(text: str) -> AssistantTurn:
    cleaned = _THINK_RE.sub("", text).strip()
    calls: list[ToolCall] = []
    for idx, match in enumerate(_TOOL_BLOCK_RE.finditer(cleaned), 1):
        name = match.group(1).strip()
        body = match.group(2)
        args = {m.group(1).strip(): _decode_param(m.group(2)) for m in _PARAM_RE.finditer(body)}
        calls.append(ToolCall(id=f"call-{idx}", name=name, arguments=args))
    if calls:
        residual = _TOOL_BLOCK_RE.sub("", cleaned).strip()
        return AssistantTurn(content=residual, tool_calls=calls)
    return AssistantTurn(content=cleaned, tool_calls=[])


def _is_capacity_error(exc: Exception, torch) -> bool:
    if not torch.cuda.is_available():
        return False
    oom_types = tuple(item for item in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(item, type))
    if oom_types and isinstance(exc, oom_types):
        return True
    accelerator_type = getattr(torch, "AcceleratorError", None)
    if isinstance(accelerator_type, type) and isinstance(exc, accelerator_type):
        markers = (
            "out of memory",
            "cuda_error_out_of_memory",
            "memory allocation failed",
            "failed to allocate",
            "insufficient gpu memory",
            "not enough gpu memory",
        )
        args_text = " ".join(str(arg) for arg in exc.args).casefold()
        return any(marker in args_text for marker in markers)
    return False


class TransformersQwenModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        *,
        quantize_4bit: bool = True,
        adapter: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 512,
        cache_implementation: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install train/runtime dependencies: pip install -e '.[train]'") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if quantize_4bit:
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Adapter loading requires PEFT: pip install -e '.[train]'") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.cache_implementation = cache_implementation if cache_implementation and torch.cuda.is_available() else None
        self._ids = itertools.count(1)

    def _device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cpu")

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        inputs = generated = new_ids = None
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=self.enable_thinking,
            )
            device = self._device()
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            tokenizer = self.tokenizer
            with self.torch.inference_mode():
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if self.cache_implementation is not None:
                    generation_kwargs["cache_implementation"] = self.cache_implementation
                generated = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = generated[0][prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            turn = parse_qwen_turn(text)
            if turn.tool_calls:
                turn.tool_calls = [ToolCall(id=f"call-{next(self._ids)}", name=c.name, arguments=c.arguments) for c in turn.tool_calls]
            return turn
        except ModelRuntimeError:
            raise
        except Exception as exc:
            if _is_capacity_error(exc, self.torch):
                raise ModelCapacityError from None
            if type(exc) is RuntimeError:
                raise ModelRuntimeError("model_error", f"Transformers inference failed: {exc}") from None
            raise
        finally:
            inputs = generated = new_ids = None
