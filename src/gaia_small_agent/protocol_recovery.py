"""Research-only bounded protocol retry for malformed native tool-call arguments.

No silent JSON fixing, no arbitrary code execution, no score feedback.
Exactly one extra model invocation on a narrowly recognized parser failure.
"""
from __future__ import annotations

from .agent.types import ModelRuntimeError

_FORMAT_PROMPT = (
    "The previous response was rejected because the native function-call arguments "
    "were not a valid JSON object. Reissue the requested function call with well-formed "
    "JSON object arguments, or answer the user directly if no tool is needed. "
    "Do not output text-form <tool_call> markup. Do not invent observations."
)

class ProtocolRepairModel:
    def __init__(self, model):
        self.model = model
        self.protocol_retries = 0
        self.protocol_retry_successes = 0

    def _invoke(self, method: str, messages: list[dict], tools: list[dict]):
        perform = getattr(self.model, method)
        try:
            return perform(messages, tools)
        except ModelRuntimeError as exc:
            if exc.stop_reason != "model_error" or not exc.message.startswith("Invalid tool arguments JSON:"):
                raise
            self.protocol_retries += 1
            # Fresh same-turn retry, without exposing the malformed arguments to tools.
            retry_messages = list(messages) + [{"role": "system", "content": _FORMAT_PROMPT}]
            try:
                result = perform(retry_messages, tools)
            except ModelRuntimeError:
                raise exc from None
            self.protocol_retry_successes += 1
            return result

    def complete(self, messages: list[dict], tools: list[dict]):
        return self._invoke("complete", messages, tools)

    def complete_final(self, messages: list[dict], tools: list[dict]):
        if not callable(getattr(self.model, "complete_final", None)):
            return self._invoke("complete", messages, [])
        return self._invoke("complete_final", messages, tools)
