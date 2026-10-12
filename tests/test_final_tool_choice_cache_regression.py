import requests

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
        return ToolResult(True, arguments["text"])


def test_runtime_final_turn_preserves_tool_schema_via_final_completion(tmp_path):
    class CacheAwareModel:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append(("auto", [tool["function"]["name"] for tool in tools]))
            return AssistantTurn(
                content="",
                tool_calls=[ToolCall("c1", "echo", {"text": "evidence"})],
            )

        def complete_final(self, messages, tools):
            self.calls.append(("none", [tool["function"]["name"] for tool in tools]))
            return AssistantTurn(content="final answer", tool_calls=[])

    model = CacheAwareModel()
    result = AgentRuntime(model, [EchoTool()], max_steps=2).run("answer", tmp_path)

    assert result.completed is True
    assert result.answer == "final answer"
    assert model.calls == [("auto", ["echo"]), ("none", ["echo"])]


def test_openai_compat_final_request_keeps_tools_and_sets_tool_choice_none(monkeypatch):
    from gaia_small_agent.model.openai_compat import OpenAICompatModel

    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "final answer", "tool_calls": []}}]}

    def fake_post(url, *, json, timeout):
        seen.update(url=url, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "echo input",
            "parameters": {"type": "object"},
        },
    }]
    model = OpenAICompatModel(model="qwen3.5-9b", base_url="http://127.0.0.1:8080")

    turn = model.complete_final([{"role": "user", "content": "answer"}], tools)

    assert seen["payload"]["tools"] == tools
    assert seen["payload"]["tool_choice"] == "none"
    assert turn.content == "final answer"
    assert turn.tool_calls == []
