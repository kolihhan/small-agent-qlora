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


def test_search_tool_passes_explicit_timeout_to_ddgs(monkeypatch, tmp_path):
    seen = {}

    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            seen["timeout"] = timeout

        def text(self, query, max_results):
            return []

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool(timeout_s=7.0).run({"query": "bounded search"}, tmp_path)

    assert result.ok is True
    assert seen["timeout"] == 7.0


def test_search_tool_classifies_ddgs_timeout_separately(monkeypatch, tmp_path):
    class FakeTimeout(Exception):
        pass

    class FakeDDGS:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def text(self, query, max_results):
            raise FakeTimeout("backend timed out")

    _install_fake_ddgs(monkeypatch, FakeDDGS, FakeTimeout)

    result = SearchTool(timeout_s=3.0).run({"query": "timeout case"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "SEARCH_TIMEOUT"
    assert "timed out" in result.content.casefold()
