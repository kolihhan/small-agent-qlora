import sys
import types

from gaia_small_agent.tools.search import SearchTool


def _install_fake_ddgs(monkeypatch, script):
    calls = []

    class FakeDDGS:
        def text(self, query, *, backend, max_results):
            calls.append((query, backend, max_results))
            action = script[backend]
            if isinstance(action, Exception):
                raise action
            return action

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    return calls


def test_search_falls_back_sequentially_when_first_backend_raises(monkeypatch, tmp_path):
    calls = _install_fake_ddgs(
        monkeypatch,
        {
            "brave": RuntimeError("No results found."),
            "duckduckgo": [{"title": "Example", "href": "https://example.com", "body": "ok"}],
        },
    )

    result = SearchTool().run({"query": "example", "max_results": 3}, tmp_path)

    assert result.ok is True
    assert "https://example.com" in result.content
    assert [backend for _, backend, _ in calls] == ["brave", "duckduckgo"]


def test_search_falls_back_after_empty_result_set(monkeypatch, tmp_path):
    calls = _install_fake_ddgs(
        monkeypatch,
        {
            "brave": [],
            "duckduckgo": [{"title": "Fallback", "href": "https://example.org", "body": "ok"}],
        },
    )

    result = SearchTool().run({"query": "fallback", "max_results": 2}, tmp_path)

    assert result.ok is True
    assert "https://example.org" in result.content
    assert [backend for _, backend, _ in calls] == ["brave", "duckduckgo"]


def test_search_returns_error_only_when_all_backends_fail(monkeypatch, tmp_path):
    _install_fake_ddgs(
        monkeypatch,
        {
            "brave": RuntimeError("backend down"),
            "duckduckgo": RuntimeError("backend down too"),
        },
    )

    result = SearchTool().run({"query": "example", "max_results": 1}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_ERROR"
    assert "brave" in result.content
    assert "duckduckgo" in result.content
