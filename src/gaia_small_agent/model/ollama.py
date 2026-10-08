from __future__ import annotations

import itertools
from typing import Any

import requests

from ..agent.types import AssistantTurn, ModelRuntimeError, ToolCall


class OllamaModel:
    def __init__(self, model: str = "qwen3.5:4b", base_url: str = "http://localhost:11434", timeout_s: float = 180.0, max_new_tokens: int = 512, enable_thinking: bool = False):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._ids = itertools.count(1)

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": self.enable_thinking,
            "options": {"temperature": 0, "num_predict": self.max_new_tokens},
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise ModelRuntimeError("model_timeout", f"Ollama request timed out: {exc}") from None
        except requests.ConnectionError as exc:
            raise ModelRuntimeError("model_unavailable", f"Ollama is unavailable: {exc}") from None
        except requests.RequestException as exc:
            raise ModelRuntimeError("model_error", f"Ollama request failed: {exc}") from None
        except ValueError as exc:
            raise ModelRuntimeError("model_error", f"Ollama returned invalid JSON: {exc}") from None

        if not isinstance(data, dict):
            raise ModelRuntimeError("model_error", "Ollama response must be a JSON object")
        message = data.get("message")
        if not isinstance(message, dict):
            raise ModelRuntimeError("model_error", "Ollama response is missing a valid message object")

        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ModelRuntimeError("model_error", "Ollama message tool_calls must be a list")
        for raw in raw_calls:
            if not isinstance(raw, dict):
                raise ModelRuntimeError("model_error", "Ollama tool call must be an object")
            fn = raw.get("function") or {}
            if not isinstance(fn, dict):
                raise ModelRuntimeError("model_error", "Ollama tool call function must be an object")
            args: Any = fn.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append(ToolCall(
                id=str(raw.get("id") or f"call-{next(self._ids)}"),
                name=str(fn.get("name") or ""),
                arguments=args,
            ))
        return AssistantTurn(content=str(message.get("content") or ""), tool_calls=calls)
