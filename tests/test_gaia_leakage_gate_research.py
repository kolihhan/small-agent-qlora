"""Gold-blind benchmark contamination guard; synthetic tests only."""
from gaia_small_agent.agent.leakage_gate import LeakageGate, leakage_signals
from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.base import Tool, ToolResult

QUESTION = "In which year did the astronomer publish his first catalog of nebulae and stars?"
CONTAMINATED = (
    "GAIA benchmark task solution\n"
    "Question: In which year did the astronomer publish his first catalog of nebulae and stars?\n"
    "ground_truth: 1786\n"
)
CLEAN = "Royal Astronomical Society historic library: early scientific catalog publications"

def test_block_explicit_answer_key_without_gold():
    assert leakage_signals(QUESTION, CONTAMINATED)
    assert not leakage_signals(QUESTION, CLEAN)
    assert not leakage_signals(QUESTION, "The word Gaia refers to Greek mythology, not this astronomy task.")

def test_flag_generic_benchmark_reference_and_same_task_text():
    document = "GAIA benchmark worked example\n" + QUESTION + "\nDiscussion of methodology."
    assert leakage_signals(QUESTION, document) == ("BENCHMARK_WITH_TASK_TEXT",)

def test_safe_query_on_benchmark_mentions_only():
    assert not leakage_signals(QUESTION, "The GAIA evaluation benchmark assesses agents.")
    assert not leakage_signals(QUESTION, "ground_truth column in a machine-learning dataset")

def test_gate_removes_payload_and_returns_explicit_error():
    guard=LeakageGate()
    result=guard("search", ToolResult(True, CONTAMINATED), QUESTION)
    assert not result.ok and result.error_code == "BENCHMARK_CONTENT_BLOCKED"
    assert "1786" not in result.content
    assert guard.blocked == 1
    assert guard("read", ToolResult(True, CLEAN), QUESTION).ok

class FakeTool(Tool):
    name = "search"
    description = "test"
    schema = {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
    def run(self, arguments, workspace):
        return ToolResult(True, CONTAMINATED)

class ProbeModel:
    def __init__(self):
        self.seen=[]
    def complete(self,messages,tools):
        self.seen.append(messages)
        if len(self.seen)==1:
            return AssistantTurn("",[ToolCall("a","search",{"query":"historical catalog"})])
        assert "1786" not in repr(messages)
        return AssistantTurn("abstain")

def test_runtime_gate_blocks_before_model_and_counts_as_error(tmp_path):
    model=ProbeModel()
    out=AgentRuntime(model,[FakeTool()],max_steps=4,observation_filter=LeakageGate()).run(QUESTION,tmp_path)
    assert out.completed
    assert out.metrics.tool_errors==1
    assert out.metrics.tool_successes==0
    assert out.answer=="abstain"
    assert not any("1786" in str(x.data) for x in out.trace)
