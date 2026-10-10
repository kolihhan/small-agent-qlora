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
        errors: list[str] = []
        rows = []
        timeout_count = 0
        backends = ("brave", "duckduckgo")
        for backend in backends:
            try:
                rows = list(
                    self._client(DDGS).text(
                        query,
                        backend=backend,
                        max_results=max_results,
                    )
                )
            except TimeoutException as exc:
                timeout_count += 1
                errors.append(f"{backend}: timed out: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                continue
            if rows:
                break
            errors.append(f"{backend}: no results")

        if not rows:
            detail = "; ".join(errors) if errors else "no results"
            error_code = "SEARCH_TIMEOUT" if timeout_count == len(backends) else "SEARCH_ERROR"
            return ToolResult(False, f"Search failed: {detail}", error_code)

        lines = []
        for i, row in enumerate(rows, 1):
            title = row.get("title", "")
            url = row.get("href") or row.get("url") or ""
            body = row.get("body") or row.get("snippet") or ""
            lines.append(f"[{i}] {title}\nURL: {url}\n{body}")
        return ToolResult(True, "\n\n".join(lines))
