from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


@dataclass
class ToolResult:
    ok: bool
    content: str
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def observation(self) -> str:
        metadata = ""
        if self.metadata:
            metadata = f"ToolMeta: {json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)}\n"
        if self.ok:
            return metadata + self.content
        return metadata + f"ToolError[{self.error_code or 'ERROR'}]: {self.content}"


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

    def validate_arguments(self, arguments: dict[str, Any]) -> str | None:
        errors = sorted(
            Draft202012Validator(self.schema).iter_errors(arguments),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        if not errors:
            return None
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path)
        location = path or "<root>"
        return f"{location}: {error.message}"

    @abstractmethod
    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        raise NotImplementedError
