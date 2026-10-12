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

    @staticmethod
    def _metadata_prefix(metadata: dict[str, Any]) -> str:
        if not metadata:
            return ""
        return f"ToolMeta: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n"

    def _body_prefix(self) -> str:
        if self.ok:
            return ""
        return f"ToolError[{self.error_code or 'ERROR'}]: "

    def observation(self) -> str:
        return self._metadata_prefix(self.metadata) + self._body_prefix() + self.content

    def bounded_observation(self, max_chars: int) -> tuple[str, dict[str, Any]]:
        """Render an observation without making pagination metadata skip unseen content."""
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")

        metadata = dict(self.metadata)
        full = self.observation()
        if len(full) <= max_chars:
            return full, metadata

        body_prefix = self._body_prefix()
        offset = metadata.get("offset_chars")
        total = metadata.get("total_chars")
        pagination = (
            self.ok
            and isinstance(offset, int)
            and not isinstance(offset, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and offset >= 0
            and total >= offset
        )
        if pagination:
            marker = f"\n...[truncated from {len(self.content)} content characters; continue from next_offset_chars when provided]"
        else:
            marker = (
                f"\n...[truncated from {len(self.content)} characters; request a narrower result "
                "or use another tool call to inspect only what is needed]"
            )

        visible = 0
        for _ in range(6):
            prefix = self._metadata_prefix(metadata)
            visible = max(0, min(len(self.content), max_chars - len(prefix) - len(body_prefix) - len(marker)))
            if not pagination:
                break
            previous = (metadata.get("returned_chars"), metadata.get("next_offset_chars"))
            metadata["returned_chars"] = visible
            next_offset = offset + visible
            metadata["next_offset_chars"] = next_offset if next_offset < total else None
            current = (metadata["returned_chars"], metadata["next_offset_chars"])
            if current == previous:
                break

        prefix = self._metadata_prefix(metadata)
        visible = max(0, min(len(self.content), max_chars - len(prefix) - len(body_prefix) - len(marker)))
        if pagination and metadata.get("returned_chars") != visible:
            metadata["returned_chars"] = visible
            next_offset = offset + visible
            metadata["next_offset_chars"] = next_offset if next_offset < total else None
            prefix = self._metadata_prefix(metadata)
            visible = max(0, min(len(self.content), max_chars - len(prefix) - len(body_prefix) - len(marker)))

        observation = prefix + body_prefix + self.content[:visible] + marker
        return observation[:max_chars], metadata


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
