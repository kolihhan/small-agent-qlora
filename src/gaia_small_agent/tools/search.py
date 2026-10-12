from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


def _compact_parallel_results(text: str, max_results: int) -> str | None:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None

    blocks: list[str] = []
    for index, raw in enumerate(results[:max_results], 1):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "Untitled")
        url = str(raw.get("url") or "")
        excerpts = raw.get("excerpts") or []
        if isinstance(excerpts, str):
            excerpt = excerpts.strip()
        elif isinstance(excerpts, list):
            excerpt = " ".join(str(item).strip() for item in excerpts if str(item).strip())
        else:
            excerpt = ""
        lines = [f"{index}. {title}"]
        if url:
            lines.append(url)
        if excerpt:
            lines.append(excerpt)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else None


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
        raw_max_results = arguments.get("max_results", 5)
        if not isinstance(raw_max_results, int) or isinstance(raw_max_results, bool):
            return ToolResult(False, "'max_results' must be an integer", "BAD_ARGUMENTS")
        max_results = max(1, min(raw_max_results, 10))

        try:
            result = asyncio.run(
                asyncio.wait_for(self._search_parallel(query.strip()), timeout=self.timeout_s)
            )
        except TimeoutError:
            return ToolResult(False, "Search timed out", "SEARCH_TIMEOUT")
        except RuntimeError as exc:
            return ToolResult(False, f"Search failed: {exc}", "SEARCH_ERROR")

        if result.ok:
            compacted = _compact_parallel_results(result.content, max_results)
            if compacted is not None:
                return ToolResult(True, compacted)
        return result
