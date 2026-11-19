import pytest

from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.benchmark.fork_probe import run_pair
from gaia_small_agent.tools.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "fixture echo"
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"]}

    def run(self, arguments, workspace):
        return ToolResult(ok=True, content=arguments["text"])


class Model:
    def complete(self, messages, tools):
        if any(m.get("role") == "tool" and m.get("content") == "evidence 2" for m in messages):
            return AssistantTurn(content="verified answer")
        if not tools:
            return AssistantTurn(content="early guess")
        return AssistantTurn(content="", tool_calls=[
            ToolCall("later", "echo", {"text": "evidence 2"})])


def run(tmp_path, **overrides):
    wa, wc = tmp_path / "a", tmp_path / "b"
    wa.mkdir(exist_ok=True)
    wc.mkdir(exist_ok=True)
    args = dict(
        question="Answer from fixture", source_id="non-GAIA-fixture-sha",
        prefix=[AssistantTurn(content="", tool_calls=[
            ToolCall("first", "echo", {"text": "evidence 1"})])],
        model_factory=Model, tools_factory=lambda: [EchoTool()],
        answer_workspace=wa, continue_workspace=wc, max_steps=4,
    )
    args.update(overrides)
    return run_pair(**args)


def test_same_prefix_forks_and_keeps_remaining_step_budget(tmp_path):
    result = run(tmp_path)
    assert result["arms"]["answer_now"]["answer"] == "early guess"
    assert result["arms"]["continue_tools"]["answer"] == "verified answer"
    assert result["arms"]["answer_now"]["postfork_tool_calls"] == 0
    assert result["arms"]["continue_tools"]["postfork_tool_calls"] == 1
    assert result["prefix_tool_turns"] == 1
    assert result["not_an_accuracy_result"] is True


def test_mismatched_fixture_rejected_before_any_model_call(tmp_path):
    wa, wc = tmp_path / "a", tmp_path / "b"
    wa.mkdir()
    wc.mkdir()
    (wa / "data.txt").write_text("different")
    with pytest.raises(ValueError, match="identical source fixtures"):
        run_pair(question="q", source_id="non-GAIA", prefix=[
            AssistantTurn(content="", tool_calls=[
                ToolCall("p", "echo", {"text": "x"})])],
            model_factory=Model, tools_factory=lambda: [EchoTool()],
            answer_workspace=wa, continue_workspace=wc)


def test_non_tool_prefix_and_insufficient_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="only of original"):
        run(tmp_path, prefix=[AssistantTurn(content="answer", tool_calls=[])])
    with pytest.raises(ValueError, match="remaining steps"):
        run(tmp_path, max_steps=1)


def test_final_arm_cannot_call_tools_even_if_model_wants_to(tmp_path):
    class IgnoresToolConstraint(Model):
        def complete(self, messages, tools):
            return AssistantTurn(content="", tool_calls=[
                ToolCall("later", "echo", {"text": "illegal after fork"})])
    result = run(tmp_path, model_factory=IgnoresToolConstraint, max_steps=3)
    assert result["arms"]["answer_now"]["completed"] is False
    assert result["arms"]["answer_now"]["postfork_tool_calls"] == 0
    assert result["arms"]["answer_now"]["unexpected_tool_at_final"] is True


def test_different_prefix_observation_rejected(tmp_path):
    class ChangingEcho(EchoTool):
        def __init__(self, token):
            self.token = token

        def run(self, arguments, workspace):
            return ToolResult(ok=True, content=self.token)

    calls = iter(["one", "two"])
    with pytest.raises(ValueError, match="pre-fork observations diverged"):
        run(tmp_path, tools_factory=lambda: [ChangingEcho(next(calls))])
