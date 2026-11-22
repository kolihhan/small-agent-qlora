from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult
from scripts.run_gaia_eval_shard import build_parser


class LongTool(Tool):
    name = "long"
    description = "Return fixed long passage"
    schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, arguments, workspace):
        return ToolResult(True, "A" * 6500 + "END")


class CaptureModel:
    def __init__(self):
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        if len(self.seen) == 1:
            return AssistantTurn("", [ToolCall("t1", "long", {})])
        return AssistantTurn("final")


def test_observation_2k_vs_8k_only_changes_visible_budget(tmp_path):
    observations = {}
    for cap in (2000, 8000):
        model = CaptureModel()
        result = AgentRuntime(model, [LongTool()], max_steps=2,
                              max_tool_observation_chars=cap).run("same task", tmp_path / str(cap))
        assert result.completed and result.metrics.tool_calls == 1
        obs = model.seen[1][-1]["content"]
        assert len(obs) <= cap
        observations[cap] = obs
    assert "[truncated" in observations[2000]
    assert "END" not in observations[2000]
    assert "END" in observations[8000]
    assert "[truncated" not in observations[8000]


def test_default_budget_unchanged_and_flag_restricted():
    assert MAX_TOOL_OBSERVATION_CHARS == 2000
    parser = build_parser()
    args = ["--hf-token", "test", "--shard-index", "0",
            "--work-root", "run", "--output", "run/summary.json"]
    assert parser.parse_args(args).max_tool_observation_chars == 2000
    assert parser.parse_args(args + ["--max-tool-observation-chars", "8000"]).max_tool_observation_chars == 8000
