import asyncio

from gaia_small_agent.tools.base import ToolResult
from gaia_small_agent.tools.search import SearchTool


def test_search_tool_calls_parallel_mcp_without_changing_public_contract(monkeypatch, tmp_path):
    seen = {}

    async def fake_parallel_search(self, query):
        seen["endpoint"] = self.endpoint
        seen["query"] = query
        return ToolResult(True, "FastAPI docs\nhttps://fastapi.tiangolo.com/")

    monkeypatch.setattr(SearchTool, "_search_parallel", fake_parallel_search, raising=False)

    result = SearchTool(timeout_s=7.0).run({"query": "bounded search", "max_results": 3}, tmp_path)

    assert result.ok is True
    assert seen == {
        "endpoint": "https://search.parallel.ai/mcp",
        "query": "bounded search",
    }
    assert "https://fastapi.tiangolo.com/" in result.content


def test_search_tool_classifies_parallel_timeout(monkeypatch, tmp_path):
    async def slow_parallel_search(self, query):
        await asyncio.sleep(0.05)
        return ToolResult(True, "too late")

    def timeout_fallback(self, query, max_results):
        return ToolResult(False, "fallback timed out", "SEARCH_TIMEOUT")

    monkeypatch.setattr(SearchTool, "_search_parallel", slow_parallel_search, raising=False)
    monkeypatch.setattr(SearchTool, "_search_fallback", timeout_fallback, raising=False)

    result = SearchTool(timeout_s=0.001).run({"query": "timeout case"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_TIMEOUT"
    assert "timed out" in result.content.casefold()


def test_search_tool_reports_error_when_primary_and_fallback_fail(monkeypatch, tmp_path):
    async def failed_parallel_search(self, query):
        return ToolResult(False, "Parallel rate limit reached", "SEARCH_ERROR")

    def failed_fallback(self, query, max_results):
        return ToolResult(False, "fallback unavailable", "SEARCH_ERROR")

    monkeypatch.setattr(SearchTool, "_search_parallel", failed_parallel_search, raising=False)
    monkeypatch.setattr(SearchTool, "_search_fallback", failed_fallback, raising=False)

    result = SearchTool().run({"query": "rate limited"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_ERROR"
    assert "rate limit" in result.content.casefold()
    assert "fallback unavailable" in result.content.casefold()
