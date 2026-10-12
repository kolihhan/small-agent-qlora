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


def test_search_tool_uses_ddgs_auto_backend_with_timeout(monkeypatch, tmp_path):
    seen = {"timeouts": [], "calls": []}

    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            seen["timeouts"].append(timeout)

        def text(self, query, *, backend, max_results):
            seen["calls"].append((query, backend, max_results))
            return [{"title": "ok", "href": "https://example.com", "body": "result"}]

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool(timeout_s=7.0).run({"query": "bounded search", "max_results": 3}, tmp_path)

    assert result.ok is True
    assert seen["timeouts"] == [7.0]
    assert seen["calls"] == [("bounded search", "auto", 3)]
    assert "https://example.com" in result.content


def test_search_tool_classifies_auto_backend_timeout(monkeypatch, tmp_path):
    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def text(self, query, *, backend, max_results):
            assert backend == "auto"
            raise FakeTimeout("auto search timed out")

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool(timeout_s=3.0).run({"query": "timeout case"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_TIMEOUT"
    assert "timed out" in result.content.casefold()


def test_search_tool_reports_empty_auto_results(monkeypatch, tmp_path):
    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def text(self, query, *, backend, max_results):
            assert backend == "auto"
            return []

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool().run({"query": "empty case"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_ERROR"
    assert "no results" in result.content.casefold()
