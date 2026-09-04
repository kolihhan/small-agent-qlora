from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    content: str
    error_code: str | None = None

    def observation(self) -> str:
        if self.ok:
            return self.content
        return f"ToolError[{self.error_code or 'ERROR'}]: {self.content}"


class Tool(ABC):
    name: str
    description: str
    schema: dict[str, Any]

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }

    @abstractmethod
    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        raise NotImplementedError
