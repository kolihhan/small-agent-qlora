from __future__ import annotations

import itertools
import json
from typing import Any

import requests

from ..agent.types import AssistantTurn, ModelRuntimeError, ToolCall


class OpenAICompatModel:
    """Minimal OpenAI-compatible chat backend for llama-server and similar runtimes."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8080",
        timeout_s: float = 180.0,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_new_tokens = max_new_tokens
        self._ids = itertools.count(1)

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise ModelRuntimeError("model_timeout", f"Model request timed out: {exc}") from None
        except requests.ConnectionError as exc:
            raise ModelRuntimeError("model_unavailable", f"Model server is unavailable: {exc}") from None
        except requests.RequestException as exc:
            raise ModelRuntimeError("model_error", f"Model request failed: {exc}") from None
        except ValueError as exc:
            raise ModelRuntimeError("model_error", f"Model server returned invalid JSON: {exc}") from None

        if not isinstance(data, dict):
            raise ModelRuntimeError("model_error", "Model response must be a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelRuntimeError("model_error", "Model response is missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ModelRuntimeError("model_error", "Model response is missing a valid message object")

        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ModelRuntimeError("model_error", "Model message tool_calls must be a list")
        for raw in raw_calls:
            if not isinstance(raw, dict):
                raise ModelRuntimeError("model_error", "Model tool call must be an object")
            fn = raw.get("function") or {}
            if not isinstance(fn, dict):
                raise ModelRuntimeError("model_error", "Model tool call function must be an object")
            args: Any = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as exc:
                    raise ModelRuntimeError("model_error", f"Invalid tool arguments JSON: {exc}") from None
            if not isinstance(args, dict):
                raise ModelRuntimeError("model_error", "Model tool arguments must decode to an object")
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call-{next(self._ids)}"),
                    name=str(fn.get("name") or ""),
                    arguments=args,
                )
            )

        return AssistantTurn(content=str(message.get("content") or ""), tool_calls=calls)
