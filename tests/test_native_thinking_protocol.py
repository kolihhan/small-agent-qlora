import requests

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "echo input"
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, arguments, workspace):
        return ToolResult(True, arguments["text"])


def test_openai_compat_sends_bounded_thinking_budget_and_preserves_reasoning(monkeypatch):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": "I should inspect the evidence first.",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"text":"evidence"}',
                            },
                        }],
                    }
                }]
            }

    def fake_post(url, *, json, timeout):
        seen.update(url=url, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    model = OpenAICompatModel(
        model="qwen3.5-9b",
        base_url="http://127.0.0.1:8080",
        thinking_budget_tokens=256,
    )
    tools = [EchoTool().definition()]

    turn = model.complete([{"role": "user", "content": "answer"}], tools)

    assert seen["payload"]["thinking_budget_tokens"] == 256
    assert turn.reasoning_content == "I should inspect the evidence first."
    assert turn.tool_calls[0].name == "echo"


def test_runtime_preserves_reasoning_content_across_tool_turns(tmp_path):
    class ReasoningAwareModel:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AssistantTurn(
                    content="",
                    reasoning_content="Need one piece of evidence.",
                    tool_calls=[ToolCall("call-1", "echo", {"text": "evidence"})],
                )
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            assert assistant_messages[-1]["reasoning_content"] == "Need one piece of evidence."
            return AssistantTurn(content="done")

    model = ReasoningAwareModel()
    result = AgentRuntime(model, [EchoTool()], max_steps=3).run("answer", tmp_path)

    assert result.completed is True
    assert result.answer == "done"
