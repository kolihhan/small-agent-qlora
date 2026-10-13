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


def test_openai_compat_preserves_reasoning_content(monkeypatch):
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
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text":"evidence"}'},
                        }],
                    }
                }]
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())
    model = OpenAICompatModel(model="qwen3.5-9b", base_url="http://127.0.0.1:8080")

    turn = model.complete([{"role": "user", "content": "answer"}], [])

    assert turn.reasoning_content == "I should inspect the evidence first."


def test_runtime_replays_reasoning_content_on_next_model_call(tmp_path):
    class ReasoningAwareModel:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AssistantTurn(
                    content="",
                    reasoning_content="Need one tool result.",
                    tool_calls=[ToolCall("c1", "echo", {"text": "evidence"})],
                )
            return AssistantTurn(content="final answer")

    model = ReasoningAwareModel()
    result = AgentRuntime(model, [EchoTool()], max_steps=3).run("answer", tmp_path)

    assert result.completed is True
    second_call_messages = model.calls[1]
    assistant_message = next(
        message
        for message in second_call_messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "Need one tool result."
