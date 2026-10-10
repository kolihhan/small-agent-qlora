import sys
import types

from gaia_small_agent.tools.search import SearchTool


def _install_fake_ddgs(monkeypatch, ddgs_cls, timeout_exc_cls):
    ddgs_module = types.ModuleType("ddgs")
    ddgs_module.DDGS = ddgs_cls
    exceptions_module = types.ModuleType("ddgs.exceptions")
    exceptions_module.TimeoutException = timeout_exc_cls
    monkeypatch.setitem(sys.modules, "ddgs", ddgs_module)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exceptions_module)


def test_search_tool_bounds_each_backend_and_falls_back_after_timeout(monkeypatch, tmp_path):
    seen = {"timeouts": [], "backends": []}

    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            seen["timeouts"].append(timeout)

        def text(self, query, *, backend, max_results):
            seen["backends"].append(backend)
            if backend == "brave":
                raise FakeTimeout("brave timed out")
            return [{"title": "ok", "href": "https://example.com", "body": "result"}]

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool(timeout_s=7.0).run({"query": "bounded search"}, tmp_path)

    assert result.ok is True
    assert seen["timeouts"] == [7.0, 7.0]
    assert seen["backends"] == ["brave", "duckduckgo"]
    assert "https://example.com" in result.content


def test_search_tool_classifies_all_backend_timeouts(monkeypatch, tmp_path):
    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def text(self, query, *, backend, max_results):
            raise FakeTimeout(f"{backend} timed out")

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool(timeout_s=3.0).run({"query": "timeout case"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_TIMEOUT"
    assert "timed out" in result.content.casefold()
