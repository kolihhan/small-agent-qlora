from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.tools.read import ReadTool
from gaia_small_agent.tools.search import SearchTool


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def complete(self, messages, tools):
        return next(self.turns)


class StrictEchoTool(Tool):
    name = "strict_echo"
    description = "echo a value"
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.calls = 0

    def run(self, arguments, workspace):
        self.calls += 1
        return ToolResult(True, arguments["text"])


class FlakyTool(Tool):
    name = "flaky"
    description = "fails once with a retryable timeout"
    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.calls = 0

    def run(self, arguments, workspace):
        self.calls += 1
        if self.calls == 1:
            return ToolResult(False, "temporary timeout", "SEARCH_TIMEOUT")
        return ToolResult(True, "recovered")


def test_read_supports_offset_chars_and_reports_next_offset(tmp_path):
    text = "A" * 2100 + "TARGET" + "B" * 200
    (tmp_path / "long.txt").write_text(text, encoding="utf-8")

    result = ReadTool().run(
        {"source": "long.txt", "offset_chars": 2000, "max_chars": 300},
        tmp_path,
    )

    assert result.ok is True
    assert "TARGET" in result.content
    assert result.metadata == {
        "offset_chars": 2000,
        "returned_chars": 300,
        "total_chars": len(text),
        "next_offset_chars": 2300,
    }


def test_runtime_rejects_schema_extra_property_before_tool_execution(tmp_path):
    tool = StrictEchoTool()
    model = ScriptedModel([
        AssistantTurn("", [ToolCall("c1", "strict_echo", {"text": "ok", "extra": 1})]),
        AssistantTurn("done", []),
    ])

    result = AgentRuntime(model, [tool], max_steps=4).run("validate", tmp_path)

    assert tool.calls == 0
    assert result.metrics.invalid_tool_calls == 1
    assert result.metrics.tool_errors == 0
    assert any(
        event.kind == "tool_result" and event.data.get("error_code") == "BAD_ARGUMENTS"
        for event in result.trace
    )


def test_duplicate_block_does_not_count_as_tool_error(tmp_path):
    tool = StrictEchoTool()
    model = ScriptedModel([
        AssistantTurn("", [ToolCall("c1", "strict_echo", {"text": "x"})]),
        AssistantTurn("", [ToolCall("c2", "strict_echo", {"text": "x"})]),
        AssistantTurn("done", []),
    ])

    result = AgentRuntime(model, [tool], max_steps=5).run("dup", tmp_path)

    assert result.metrics.duplicate_calls_blocked == 1
    assert result.metrics.tool_errors == 0
    assert result.metrics.tool_successes == 1


def test_same_call_can_retry_once_after_retryable_tool_failure(tmp_path):
    tool = FlakyTool()
    model = ScriptedModel([
        AssistantTurn("", [ToolCall("c1", "flaky", {"key": "x"})]),
        AssistantTurn("", [ToolCall("c2", "flaky", {"key": "x"})]),
        AssistantTurn("done", []),
    ])

    result = AgentRuntime(model, [tool], max_steps=5).run("retry", tmp_path)

    assert tool.calls == 2
    assert result.metrics.tool_retries == 1
    assert result.metrics.duplicate_calls_blocked == 0
    assert result.metrics.tool_errors == 1
    assert result.metrics.tool_successes == 1


def test_search_falls_back_when_parallel_search_fails(monkeypatch, tmp_path):
    async def failed_parallel(self, query):
        return ToolResult(False, "parallel unavailable", "SEARCH_ERROR")

    def fallback(self, query, max_results):
        return ToolResult(True, "Fallback result\nhttps://example.com/")

    monkeypatch.setattr(SearchTool, "_search_parallel", failed_parallel)
    monkeypatch.setattr(SearchTool, "_search_fallback", fallback, raising=False)

    result = SearchTool().run({"query": "fallback test", "max_results": 3}, tmp_path)

    assert result.ok is True
    assert "Fallback result" in result.content
