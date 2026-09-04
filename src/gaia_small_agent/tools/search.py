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

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, "'query' must be a non-empty string", "BAD_ARGUMENTS")
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(False, "Install search support: pip install -e '.[search]'", "MISSING_DEPENDENCY")
        max_results = max(1, min(int(arguments.get("max_results", 5)), 10))
        try:
            rows = list(DDGS().text(query, max_results=max_results))
        except Exception as exc:
            return ToolResult(False, f"Search failed: {exc}", "SEARCH_ERROR")
        if not rows:
            return ToolResult(True, "No results.")
        lines = []
        for i, row in enumerate(rows, 1):
            title = row.get("title", "")
            url = row.get("href") or row.get("url") or ""
            body = row.get("body") or row.get("snippet") or ""
            lines.append(f"[{i}] {title}\nURL: {url}\n{body}")
        return ToolResult(True, "\n\n".join(lines))
