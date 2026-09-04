from __future__ import annotations

import itertools
from typing import Any

import requests

from ..agent.types import AssistantTurn, ToolCall


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
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args: Any = fn.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append(ToolCall(
                id=str(raw.get("id") or f"call-{next(self._ids)}"),
                name=str(fn.get("name") or ""),
                arguments=args,
            ))
        return AssistantTurn(content=str(message.get("content") or ""), tool_calls=calls)
