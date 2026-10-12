from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


class SearchTool(Tool):
    name = "search"
    description = "Search the public web and return titles, URLs, and snippets."
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = float(timeout_s)

    def _client(self, ddgs_cls):
        try:
            return ddgs_cls(timeout=self.timeout_s)
        except TypeError:
            # Keep compatibility with simple injected clients while real DDGS gets a hard timeout.
            return ddgs_cls()

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, "'query' must be a non-empty string", "BAD_ARGUMENTS")
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(False, "Install search support: pip install -e '.[search]'", "MISSING_DEPENDENCY")
        try:
            from ddgs.exceptions import TimeoutException
        except ImportError:
            class TimeoutException(Exception):
                pass

        max_results = max(1, min(int(arguments.get("max_results", 5)), 10))
        try:
            rows = list(
                self._client(DDGS).text(
                    query,
                    backend="auto",
                    max_results=max_results,
                )
            )
        except TimeoutException as exc:
            return ToolResult(False, f"Search timed out: {exc}", "SEARCH_TIMEOUT")
        except Exception as exc:
            return ToolResult(False, f"Search failed: {exc}", "SEARCH_ERROR")

        if not rows:
            return ToolResult(False, "Search failed: no results", "SEARCH_ERROR")

        lines = []
        for i, row in enumerate(rows, 1):
            title = row.get("title", "")
            url = row.get("href") or row.get("url") or ""
            body = row.get("body") or row.get("snippet") or ""
            lines.append(f"[{i}] {title}\nURL: {url}\n{body}")
        return ToolResult(True, "\n\n".join(lines))
