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
        errors: list[str] = []
        rows = []
        for backend in ("brave", "duckduckgo"):
            try:
                rows = list(DDGS().text(query, backend=backend, max_results=max_results))
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                continue
            if rows:
                break
            errors.append(f"{backend}: no results")

        if not rows:
            detail = "; ".join(errors) if errors else "no results"
            return ToolResult(False, f"Search failed: {detail}", "SEARCH_ERROR")

        lines = []
        for i, row in enumerate(rows, 1):
            title = row.get("title", "")
            url = row.get("href") or row.get("url") or ""
            body = row.get("body") or row.get("snippet") or ""
            lines.append(f"[{i}] {title}\nURL: {url}\n{body}")
        return ToolResult(True, "\n\n".join(lines))
