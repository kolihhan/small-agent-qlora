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


def _format_ddgs_results(rows: list[dict], max_results: int) -> str:
    blocks: list[str] = []
    for index, row in enumerate(rows[:max_results], 1):
        title = str(row.get("title") or "Untitled")
        url = str(row.get("href") or row.get("url") or "")
        body = str(row.get("body") or row.get("snippet") or "")
        lines = [f"{index}. {title}"]
        if url:
            lines.append(url)
        if body:
            lines.append(body)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class SearchTool(Tool):
    name = "search"
    description = "Search the public web and return source links and excerpts."
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
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

    def _search_fallback(self, query: str, max_results: int) -> ToolResult:
        try:
            from ddgs import DDGS
            from ddgs.exceptions import TimeoutException
        except ImportError:
            return ToolResult(False, "DDGS fallback dependency unavailable", "MISSING_DEPENDENCY")

        try:
            rows = list(DDGS(timeout=self.timeout_s).text(query, max_results=max_results, backend="auto"))
        except TimeoutException as exc:
            return ToolResult(False, f"Fallback search timed out: {exc}", "SEARCH_TIMEOUT")
        except Exception as exc:
            return ToolResult(False, f"Fallback search failed: {exc}", "SEARCH_ERROR")
        if not rows:
            return ToolResult(False, "Fallback search returned no results", "SEARCH_ERROR")
        return ToolResult(True, _format_ddgs_results(rows, max_results))

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, "'query' must be a non-empty string", "BAD_ARGUMENTS")
        raw_max_results = arguments.get("max_results", 5)
        if not isinstance(raw_max_results, int) or isinstance(raw_max_results, bool):
            return ToolResult(False, "'max_results' must be an integer", "BAD_ARGUMENTS")
        max_results = max(1, min(raw_max_results, 10))
        query = query.strip()

        try:
            result = asyncio.run(
                asyncio.wait_for(self._search_parallel(query), timeout=self.timeout_s)
            )
        except TimeoutError:
            result = ToolResult(False, "Search timed out", "SEARCH_TIMEOUT")
        except RuntimeError as exc:
            result = ToolResult(False, f"Search failed: {exc}", "SEARCH_ERROR")

        if result.ok:
            compacted = _compact_parallel_results(result.content, max_results)
            if compacted is not None:
                return ToolResult(True, compacted)
            return result

        fallback = self._search_fallback(query, max_results)
        if fallback.ok:
            fallback.metadata = {"fallback_from": result.error_code or "SEARCH_ERROR"}
            return fallback
        return ToolResult(
            False,
            f"Primary search failed: {result.content}; fallback failed: {fallback.content}",
            fallback.error_code or result.error_code or "SEARCH_ERROR",
        )
