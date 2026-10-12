import io
import json
import sys
import types

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.tools.read import ReadTool
from gaia_small_agent.tools.search import SearchTool


def _install_fake_mcp(monkeypatch, payload_by_tool):
    calls = []

    class FakeTextContent:
        def __init__(self, text):
            self.text = text

    class FakeClient:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            payload = payload_by_tool[name]
            return types.SimpleNamespace(
                content=[FakeTextContent(json.dumps(payload))],
                is_error=False,
            )

    mcp_module = types.ModuleType("mcp")
    mcp_types_module = types.ModuleType("mcp.types")
    mcp_module.Client = FakeClient
    mcp_types_module.TextContent = FakeTextContent
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types_module)
    return calls


def test_search_compacts_parallel_json_and_honors_max_results(monkeypatch, tmp_path):
    calls = _install_fake_mcp(
        monkeypatch,
        {
            "web_search": {
                "results": [
                    {"title": "First", "url": "https://one.example", "excerpts": ["alpha"]},
                    {"title": "Second", "url": "https://two.example", "excerpts": ["beta"]},
                    {"title": "Third", "url": "https://three.example", "excerpts": ["gamma"]},
                ]
            }
        },
    )

    result = SearchTool().run({"query": "test query", "max_results": 2}, tmp_path)

    assert result.ok is True
    assert "1. First" in result.content
    assert "https://one.example" in result.content
    assert "alpha" in result.content
    assert "2. Second" in result.content
    assert "Third" not in result.content
    assert '"results"' not in result.content
    assert calls[0][0] == "web_search"


def test_read_uses_parallel_web_fetch_for_public_url(monkeypatch, tmp_path):
    url = "https://example.com/article"
    calls = _install_fake_mcp(
        monkeypatch,
        {
            "web_fetch": {
                "results": [
                    {"url": url, "title": "Example", "full_content": "# Hello\nUseful body"}
                ]
            }
        },
    )

    import gaia_small_agent.tools.read as read_module

    monkeypatch.setattr(read_module, "_public_url_error", lambda value: None)
    monkeypatch.setattr(
        read_module.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct HTTP fallback should not run")),
    )

    result = ReadTool().run({"source": url}, tmp_path)

    assert result.ok is True
    assert result.content == "# Hello\nUseful body"
    assert calls == [
        (
            "web_fetch",
            {"urls": [url], "objective": "Full page content"},
        )
    ]


def test_read_remote_pdf_fallback_extracts_text_instead_of_binary(monkeypatch, tmp_path):
    from pypdf import PdfWriter
    import gaia_small_agent.tools.read as read_module

    async def failed_parallel_fetch(self, url):
        return ToolResult(False, "fetch unavailable", "FETCH_ERROR")

    monkeypatch.setattr(ReadTool, "_fetch_parallel", failed_parallel_fetch, raising=False)
    monkeypatch.setattr(read_module, "_public_url_error", lambda value: None)

    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    pdf_bytes = buffer.getvalue()

    class FakeResponse:
        headers = {"content-type": "application/pdf"}
        encoding = None
        is_redirect = False
        is_permanent_redirect = False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=65536):
            yield pdf_bytes

        def close(self):
            pass

    monkeypatch.setattr(read_module.requests, "get", lambda *args, **kwargs: FakeResponse())

    result = ReadTool().run({"source": "https://example.com/report.pdf"}, tmp_path)

    assert result.ok is True
    assert result.content == ""
    assert not result.content.startswith("%PDF-")


def test_runtime_reserves_last_model_call_for_final_synthesis(tmp_path):
    class EchoTool(Tool):
        name = "echo"
        description = "echo input"
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        def run(self, arguments, workspace):
            return ToolResult(True, arguments["text"])

    class FinalAwareModel:
        def __init__(self):
            self.tool_surfaces = []

        def complete(self, messages, tools):
            self.tool_surfaces.append([tool["function"]["name"] for tool in tools])
            if tools:
                return AssistantTurn(
                    content="",
                    tool_calls=[ToolCall("c1", "echo", {"text": "evidence"})],
                )
            return AssistantTurn(content="final answer", tool_calls=[])

    model = FinalAwareModel()
    result = AgentRuntime(model, [EchoTool()], max_steps=2).run("answer", tmp_path)

    assert result.completed is True
    assert result.answer == "final answer"
    assert result.metrics.steps == 2
    assert result.metrics.tool_calls == 1
    assert model.tool_surfaces == [["echo"], []]
