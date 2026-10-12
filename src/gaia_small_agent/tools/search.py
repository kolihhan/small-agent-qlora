from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


class SearchTool(Tool):
    name = "search"
    description = "Search the public web and return source links and excerpts."
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, timeout_s: float = 20.0, endpoint: str = "https://search.parallel.ai/mcp"):
        self.timeout_s = float(timeout_s)
        self.endpoint = endpoint

    async def _search_parallel(self, query: str) -> ToolResult:
        try:
            from mcp import Client
            from mcp.types import TextContent
        except ImportError:
            return ToolResult(False, "Install search support: pip install -e '.[search]'", "MISSING_DEPENDENCY")

        try:
            async with Client(self.endpoint) as client:
                result = await client.call_tool(
                    "web_search",
                    {
                        "objective": query,
                        "search_queries": [query],
                    },
                )
        except Exception as exc:
            return ToolResult(False, f"Search failed: {exc}", "SEARCH_ERROR")

        text = "\n\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        ).strip()
        if result.is_error:
            return ToolResult(False, text or "Parallel search failed", "SEARCH_ERROR")
        if not text:
            return ToolResult(False, "Search failed: no results", "SEARCH_ERROR")
        return ToolResult(True, text)

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, "'query' must be a non-empty string", "BAD_ARGUMENTS")

        # Keep max_results in the public schema for trained-model compatibility.
        # Parallel's anonymous MCP endpoint controls result count server-side.
        try:
            return asyncio.run(
                asyncio.wait_for(self._search_parallel(query.strip()), timeout=self.timeout_s)
            )
        except TimeoutError:
            return ToolResult(False, "Search timed out", "SEARCH_TIMEOUT")
        except RuntimeError as exc:
            return ToolResult(False, f"Search failed: {exc}", "SEARCH_ERROR")
