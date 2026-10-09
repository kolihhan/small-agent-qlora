from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "echo input"
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, arguments, workspace):
        return ToolResult(ok=True, content=arguments["text"])


def test_step_limit_gets_one_tool_free_finalization_call(tmp_path):
    class Model:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append((list(messages), list(tools)))
            if tools:
                call_id = f"c{len(self.calls)}"
                return AssistantTurn(
                    content="",
                    tool_calls=[ToolCall(call_id, "echo", {"text": f"evidence-{len(self.calls)}"})],
                )
            return AssistantTurn(content="42", tool_calls=[])

    model = Model()
    result = AgentRuntime(model, [EchoTool()], max_steps=2).run("Find the answer", tmp_path)

    assert result.completed is True
    assert result.answer == "42"
    assert result.stop_reason == "final_after_step_limit"
    assert len(model.calls) == 3
    assert model.calls[-1][1] == []
    assert "tool budget" in model.calls[-1][0][-1]["content"].casefold()
    assert result.trace[-1].kind == "final_after_step_limit"


def test_step_limit_finalization_never_executes_new_tools(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            return AssistantTurn(
                content="",
                tool_calls=[ToolCall(f"c{self.calls}", "echo", {"text": "again"})],
            )

    model = Model()
    result = AgentRuntime(model, [EchoTool()], max_steps=2).run("Find the answer", tmp_path)

    assert model.calls == 3
    assert result.completed is False
    assert result.stop_reason == "max_steps"
    assert result.metrics.tool_calls == 2
    assert result.trace[-1].kind == "step_limit_finalization_failed"


def test_empty_final_gets_one_tool_free_recovery_call(tmp_path):
    class Model:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append((list(messages), list(tools)))
            if len(self.calls) == 1:
                return AssistantTurn(content="   ", tool_calls=[])
            return AssistantTurn(content="recovered answer", tool_calls=[])

    model = Model()
    result = AgentRuntime(model, [EchoTool()], max_steps=4).run("answer", tmp_path)

    assert result.completed is True
    assert result.answer == "recovered answer"
    assert result.stop_reason == "final_after_empty"
    assert len(model.calls) == 2
    assert model.calls[-1][1] == []
    assert "no final answer" in model.calls[-1][0][-1]["content"].casefold()
    assert result.trace[-1].kind == "final_after_empty"


def test_empty_final_recovery_never_executes_new_tools(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return AssistantTurn(content="", tool_calls=[])
            return AssistantTurn(
                content="",
                tool_calls=[ToolCall("retry-tool", "echo", {"text": "should-not-run"})],
            )

    model = Model()
    result = AgentRuntime(model, [EchoTool()], max_steps=4).run("answer", tmp_path)

    assert model.calls == 2
    assert result.completed is False
    assert result.stop_reason == "empty_final"
    assert result.metrics.tool_calls == 0
    assert result.trace[-1].kind == "empty_final_recovery_failed"
