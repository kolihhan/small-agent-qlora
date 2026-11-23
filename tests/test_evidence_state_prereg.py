"""No training labels, GAIA examples or test gold in these synthetic controls."""
from __future__ import annotations

from gaia_small_agent.agent.evidence_state import EvidenceState
from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult


class ReadStub(Tool):
    name = "read"
    description = "Synthetic read for agent state test"
    schema = {
        "type": "object",
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "additionalProperties": False,
    }
    def run(self, arguments, workspace):
        data = {
            "passages/p1.txt": "Professor Alice Chen taught Leon Martin at the academy.",
            "passages/p2.txt": "Leon Martin was born in Vienna in 1890.",
        }
        return ToolResult(True, data[arguments["source"]])


class CaptureModel:
    def __init__(self):
        self.round = 0
        self.history = []
    def complete(self, messages, tools):
        self.history.append([dict(m) for m in messages])
        self.round += 1
        if self.round == 1:
            return AssistantTurn("", [ToolCall("1", "read", {"source":"passages/p1.txt"})])
        if self.round == 2:
            return AssistantTurn("", [ToolCall("2", "read", {"source":"passages/p2.txt"})])
        return AssistantTurn("Vienna")


def test_default_arm_unchanged_and_same_tool_surface(tmp_path):
    models = [CaptureModel(),CaptureModel()]
    arms = [AgentRuntime(models[0],[ReadStub()],max_steps=5),
            AgentRuntime(models[1],[ReadStub()],max_steps=5,evidence_state=True)]
    a,b = [arms[i].run("Where was the teacher's student born?",tmp_path/f"arm{i}")
           for i in (0,1)]
    assert a.completed and b.completed and a.answer == b.answer == "Vienna"
    assert a.metrics.tool_calls == b.metrics.tool_calls == 2
    assert [e.kind for e in a.trace] == [e.kind for e in b.trace]
    assert [e.data["name"] for e in a.trace if e.kind=="tool_call"] == [
           e.data["name"] for e in b.trace if e.kind=="tool_call"]
    assert "Evidence index" not in str(models[0].history)
    assert "Evidence index" in str(models[1].history)
    assert "Leon Martin" in models[1].history[-1][-1]["content"]
    for e in b.trace:
        if e.kind=="tool_result":
            assert len(e.data["content"])<=MAX_TOOL_OBSERVATION_CHARS


def test_index_is_source_anchored_without_fabricating_relation():
    state=EvidenceState("Where was Leon Martin born?")
    state.observe("read",{"source":"passages/p1.txt"},ok=True,
                  observation="Alice Chen taught Leon Martin.")
    state.observe("read",{"source":"passages/p2.txt"},ok=True,
                  observation="Leon Martin was born in Vienna.")
    summary=state.summary()
    assert "[S1 p1.txt@0]" in summary
    assert "[S2 p2.txt@0]" in summary
    assert "Leon Martin" in summary
    assert "Possible shared names" in summary
    assert "teacher" not in summary.casefold()
    assert state.safe_metrics()["successful_reads_seen"]==2


def test_fails_closed_on_failed_read_and_does_not_copy_secret_gold():
    s=EvidenceState("Who is the teacher of Alice?")
    s.observe("read",{"source":"file.txt"},ok=False,observation="secret test answer")
    s.observe("search",{"source":"file.txt"},ok=True,observation="secret test answer")
    assert s.summary()==""
    assert s.safe_metrics()["successful_reads_seen"]==0


def test_state_quotes_are_from_visible_excerpt_only_and_bounded():
    s=EvidenceState("Where was the teacher born?",max_chars=350)
    s.observe("read",{"source":"a.txt"},ok=True,
              observation="Professor Eve Stone taught Nora Lee.\n" + "x"*1400)
    text=s.summary()
    assert "Eve Stone" in text
    assert len(text)<=350
    assert "x"*500 not in text


def test_duplicate_source_does_not_prove_multihop_bridge():
    state=EvidenceState("Find a person")
    state.observe("read",{"source":"p1"},ok=True,observation="Leon Martin is here.")
    state.observe("read",{"source":"p1"},ok=True,observation="Leon Martin is still here.")
    assert "Possible shared names" not in state.summary()
