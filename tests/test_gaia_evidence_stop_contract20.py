from gaia_small_agent.agent.loop import AgentRuntime, SYSTEM_PROMPT, EVIDENCE_STOP_CONTRACT
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult
from scripts.run_gaia_eval_shard import build_parser


class CaptureModel:
    def __init__(self):
        self.messages = []

    def complete(self, messages, tools):
        self.messages.append([dict(m) for m in messages])
        if len(self.messages) == 1:
            return AssistantTurn(content="", tool_calls=[ToolCall("id1", "search", {"query": "example"})])
        return AssistantTurn(content="supported answer", tool_calls=[])


class SearchStub(Tool):
    name = "search"
    description = "Return fixed evidence"
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, arguments, workspace):
        return ToolResult(True, "evidence for supported answer")


def test_stop_contract_is_strictly_opt_in_and_preserves_same_tools(tmp_path):
    controls = []
    for enabled in (False, True):
        model = CaptureModel()
        runtime = AgentRuntime(
            model, [SearchStub()], max_steps=12,
            evidence_stop_contract=enabled,
        )
        result = runtime.run("Find supported answer", tmp_path / str(enabled))
        assert result.completed and result.answer == "supported answer"
        assert result.metrics.tool_calls == 1
        assert model.messages[0][0]["content"].startswith(SYSTEM_PROMPT)
        assert (EVIDENCE_STOP_CONTRACT in model.messages[0][0]["content"]) is enabled
        controls.append(model.messages)
    assert controls[0][1][-1] == controls[1][1][-1]
    assert len(controls[0][1][-1]["content"]) < 2000


def test_experimental_flag_defaults_to_off():
    parser = build_parser()
    base = parser.parse_args([
        "--hf-token", "test", "--shard-index", "0",
        "--work-root", "run", "--output", "run/summary.json",
    ])
    intervention = parser.parse_args([
        "--hf-token", "test", "--shard-index", "0",
        "--work-root", "run", "--output", "run/summary.json",
        "--evidence-stop-contract",
    ])
    assert not base.evidence_stop_contract
    assert intervention.evidence_stop_contract
    assert base.partition == intervention.partition == "evaluation"
