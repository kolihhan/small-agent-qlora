"""Non-GAIA unit regression: tool-loop guard changes only opted-in behavior."""
from gaia_small_agent.agent.loop import AgentRuntime, _dedup_key
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.cli import build_parser


class FakeSearch(Tool):
    name = "search"
    description = "fake, deterministic search"
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.calls = 0

    def run(self, arguments, workspace):
        self.calls += 1
        return ToolResult(True, "verified source answer: 42")


class Model:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.prompts = []

    def complete(self, messages, tools):
        self.prompts.append(messages[0]["content"])
        return next(self.turns)


def test_normalized_search_dedup_only_opted_in(tmp_path):
    turns = [
        AssistantTurn("", [ToolCall("1", "search", {"query": "  Moon   LANDING "})]),
        AssistantTurn("", [ToolCall("2", "search", {"query": "moon landing", "max_results": 5})]),
        AssistantTurn("42", []),
    ]
    tool, model = FakeSearch(), Model(turns)
    result = AgentRuntime(model, [tool], max_steps=5, tool_policy_guard=True).run(
        "When was the landing?", tmp_path)
    assert result.completed and result.answer == "42"
    assert tool.calls == 1
    assert result.metrics.duplicate_calls_blocked == 1
    assert "Progress checkpoint" not in model.prompts[0]
    assert "Progress checkpoint" in model.prompts[2]

    unguarded_tool, unguarded_model = FakeSearch(), Model(turns)
    old = AgentRuntime(unguarded_model, [unguarded_tool], max_steps=5).run(
        "When?", tmp_path)
    assert old.completed
    assert unguarded_tool.calls == 2
    assert old.metrics.duplicate_calls_blocked == 0
    assert all("Progress checkpoint" not in prompt for prompt in unguarded_model.prompts)


def test_guard_does_not_conflate_different_searches_or_file_paths():
    key = lambda name, args: _dedup_key(name, args, normalize_search=True)
    assert key("search", {"query": "  WHO   was Apollo?"}) == key("search", {"query": "who was apollo?", "max_results": 5})
    assert key("search", {"query": "Apollo", "max_results": 10}) != key("search", {"query": "apollo", "max_results": 5})
    assert key("search", {"query": "Moon"}) != key("search", {"query": "Mars"})
    assert key("read", {"source": "A.txt"}) != key("read", {"source": "a.txt"})


def test_deadline_nudge_does_not_force_unverified_final(tmp_path):
    model = Model([
        AssistantTurn("", [ToolCall("1", "search", {"query": "x"})]),
        AssistantTurn("", [ToolCall("2", "search", {"query": "y"})]),
        AssistantTurn("I cannot establish the answer", []),
    ])
    tool = FakeSearch()
    result = AgentRuntime(model, [tool], max_steps=4, tool_policy_guard=True).run(
        "hard question", tmp_path)
    assert result.completed
    assert tool.calls == 2
    assert "Budget checkpoint" in model.prompts[2]
    assert "Budget checkpoint" not in model.prompts[0]


def test_cli_option_is_explicit_and_default_off():
    parser = build_parser()
    default = parser.parse_args(["run", "question"])
    on = parser.parse_args(["run", "question", "--tool-policy-guard"])
    assert default.tool_policy_guard is False
    assert on.tool_policy_guard is True
