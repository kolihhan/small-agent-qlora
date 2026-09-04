import pytest

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.agent.types import ModelCapacityError


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)
    def complete(self, messages, tools):
        return next(self.turns)


class EchoTool(Tool):
    name = "echo"
    description = "echo input"
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    def run(self, arguments, workspace):
        return ToolResult(ok=True, content=arguments["text"])


class FailingThenWorkingTool(Tool):
    name = "fragile"
    description = "fails for bad mode"
    schema = {"type": "object", "properties": {"mode": {"type": "string"}}, "required": ["mode"]}
    def run(self, arguments, workspace):
        if arguments["mode"] == "bad":
            return ToolResult(ok=False, content="simulated failure", error_code="SIMULATED")
        return ToolResult(ok=True, content="recovered")


@pytest.mark.parametrize("tool_name", ["search", "read", "inspect", "python"])
def test_long_named_tool_observation_is_bounded_before_model_and_trace(tmp_path, tool_name):
    class CapturingModel(ScriptedModel):
        def __init__(self, turns):
            super().__init__(turns)
            self.messages = []

        def complete(self, messages, tools):
            self.messages.append(messages)
            return super().complete(messages, tools)

    observation = "x" * 3000
    tool = EchoTool()
    tool.name = tool_name
    model = CapturingModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", tool_name, {"text": observation})]),
        AssistantTurn(content="done", tool_calls=[]),
    ])

    result = AgentRuntime(model, [tool], max_steps=4).run("bound", tmp_path)

    delivered = model.messages[1][-1]["content"]
    assert len(delivered) <= 2000
    assert delivered.endswith("\n...[truncated from 3000 characters; request a narrower result or use another tool call to inspect only what is needed]")
    assert result.trace[1].data["content"] == delivered


def test_long_unknown_tool_observation_is_bounded_before_model_and_trace(tmp_path):
    class CapturingModel(ScriptedModel):
        def __init__(self, turns):
            super().__init__(turns)
            self.messages = []

        def complete(self, messages, tools):
            self.messages.append(messages)
            return super().complete(messages, tools)

    tool_name = "unknown-" + ("u" * 3000)
    model = CapturingModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", tool_name, {})]),
        AssistantTurn(content="done", tool_calls=[]),
    ])

    result = AgentRuntime(model, [], max_steps=4).run("unknown", tmp_path)

    delivered = model.messages[1][-1]["content"]
    assert len(delivered) <= 2000
    assert delivered.startswith("ToolError[UNKNOWN_TOOL]: Unknown tool: ")
    assert result.trace[1].data["content"] == delivered
    assert result.trace[1].data["error_code"] == "UNKNOWN_TOOL"


def test_long_tool_exception_observation_is_bounded_before_model_and_trace(tmp_path):
    class CapturingModel(ScriptedModel):
        def __init__(self, turns):
            super().__init__(turns)
            self.messages = []

        def complete(self, messages, tools):
            self.messages.append(messages)
            return super().complete(messages, tools)

    class ExplodingTool(Tool):
        name = "explode"
        description = "explode"
        schema = {"type": "object"}

        def run(self, arguments, workspace):
            raise RuntimeError("boom " + ("x" * 3000))

    model = CapturingModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", "explode", {})]),
        AssistantTurn(content="done", tool_calls=[]),
    ])

    result = AgentRuntime(model, [ExplodingTool()], max_steps=4).run("exception", tmp_path)

    delivered = model.messages[1][-1]["content"]
    assert len(delivered) <= 2000
    assert delivered.startswith("ToolError[TOOL_EXCEPTION]: RuntimeError: boom ")
    assert result.trace[1].data["content"] == delivered
    assert result.trace[1].data["error_code"] == "TOOL_EXCEPTION"


def test_short_tool_observation_is_preserved_exactly(tmp_path):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", "echo", {"text": "short"})]),
        AssistantTurn(content="done", tool_calls=[]),
    ])

    result = AgentRuntime(model, [EchoTool()], max_steps=4).run("short", tmp_path)

    assert result.trace[1].data["content"] == "short"


def test_bounded_tool_error_preserves_structured_error_prefix(tmp_path):
    class LongFailingTool(Tool):
        name = "long-fail"
        description = "long failure"
        schema = {"type": "object"}

        def run(self, arguments, workspace):
            return ToolResult(False, "e" * 3000, "LONG_ERROR")

    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", "long-fail", {})]),
        AssistantTurn(content="done", tool_calls=[]),
    ])

    result = AgentRuntime(model, [LongFailingTool()], max_steps=4).run("error", tmp_path)

    delivered = result.trace[1].data["content"]
    assert delivered.startswith("ToolError[LONG_ERROR]: ")
    assert len(delivered) <= 2000
    assert result.trace[1].data["ok"] is False
    assert result.trace[1].data["error_code"] == "LONG_ERROR"


def test_model_selects_tool_then_finishes(tmp_path):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", "echo", {"text": "42"})]),
        AssistantTurn(content="42", tool_calls=[]),
    ])
    result = AgentRuntime(model, [EchoTool()], max_steps=4).run("answer", tmp_path)
    assert result.answer == "42"
    assert [e.kind for e in result.trace] == ["tool_call", "tool_result", "final"]


def test_tool_error_is_observation_and_model_can_recover(tmp_path):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", "fragile", {"mode": "bad"})]),
        AssistantTurn(content="", tool_calls=[ToolCall("c2", "fragile", {"mode": "good"})]),
        AssistantTurn(content="done", tool_calls=[]),
    ])
    result = AgentRuntime(model, [FailingThenWorkingTool()], max_steps=5).run("recover", tmp_path)
    assert result.answer == "done"
    assert result.metrics.tool_errors == 1
    assert result.metrics.tool_successes == 1


def test_exact_duplicate_tool_call_is_blocked(tmp_path):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[ToolCall("c1", "echo", {"text": "x"})]),
        AssistantTurn(content="", tool_calls=[ToolCall("c2", "echo", {"text": "x"})]),
        AssistantTurn(content="stop", tool_calls=[]),
    ])
    result = AgentRuntime(model, [EchoTool()], max_steps=5).run("dup", tmp_path)
    assert result.metrics.duplicate_calls_blocked == 1
    assert any(e.data.get("error_code") == "DUPLICATE_CALL" for e in result.trace if e.kind == "tool_result")


def test_runtime_run_state_is_isolated_between_tasks(tmp_path):
    class RepeatingModel:
        def complete(self, messages, tools):
            if messages[-1]["role"] == "user":
                return AssistantTurn(content="", tool_calls=[ToolCall("c1", "echo", {"text": "same"})])
            return AssistantTurn(content="done", tool_calls=[])

    runtime = AgentRuntime(RepeatingModel(), [EchoTool()], max_steps=4)

    first = runtime.run("first", tmp_path / "first")
    second = runtime.run("second", tmp_path / "second")

    assert first.completed and second.completed
    assert first.metrics.duplicate_calls_blocked == 0
    assert second.metrics.duplicate_calls_blocked == 0
    assert first.metrics.tool_successes == 1
    assert second.metrics.tool_successes == 1


def test_model_capacity_stops_without_retry_and_retains_prior_trace(tmp_path):
    class CapacityModel:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return AssistantTurn(content="", tool_calls=[ToolCall("c1", "echo", {"text": "before"})])
            raise ModelCapacityError

    model = CapacityModel()
    result = AgentRuntime(model, [EchoTool()], max_steps=4).run("capacity", tmp_path)

    assert model.calls == 2
    assert result.answer == ""
    assert result.completed is False
    assert result.stop_reason == "model_capacity"
    assert [event.kind for event in result.trace] == ["tool_call", "tool_result", "model_capacity"]
    assert result.metrics.tool_successes == 1


def test_empty_final_answer_is_not_completed(tmp_path):
    model = ScriptedModel([AssistantTurn(content="   ", tool_calls=[])])

    result = AgentRuntime(model, [], max_steps=2).run("answer", tmp_path)

    assert result.completed is False
    assert result.stop_reason == "empty_final"
    assert result.answer == ""
    assert result.trace[-1].kind == "empty_final"
