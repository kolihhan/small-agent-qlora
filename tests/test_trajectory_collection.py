import json

from gaia_small_agent.agent.types import AgentResult, RunMetrics, TraceEvent
from gaia_small_agent.training.trajectories import trajectory_from_result, collect_verified_trajectories


def test_trajectory_from_result_preserves_tool_observation_and_final_answer():
    result = AgentResult(
        answer="646",
        completed=True,
        stop_reason="final",
        trace=[
            TraceEvent("tool_call", 1, {"id": "c1", "name": "python", "arguments": {"code": "print(17*38)"}}),
            TraceEvent("tool_result", 1, {"id": "c1", "name": "python", "ok": True, "error_code": None, "content": "646"}),
            TraceEvent("final", 2, {"answer": "646"}),
        ],
        metrics=RunMetrics(steps=2, tool_calls=1, tool_successes=1),
    )
    row = trajectory_from_result(
        task_id="calc-1",
        source="synthetic-arithmetic",
        question="Compute 17*38",
        expected_answer="646",
        result=result,
        tools=[{"type": "function", "function": {"name": "python"}}],
        provenance={"license":"CC0-1.0","generator":"unit","generator_version":"1","oracle_type":"exact","oracle_version":"1"},
    )
    assert row["verified"] is True
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert "compact autonomous tool-using assistant" in row["messages"][0]["content"]
    assert row["messages"][2]["tool_calls"][0]["function"]["name"] == "python"
    assert row["messages"][3]["content"] == "646"
    assert row["messages"][-1]["content"] == "646"


def test_collector_writes_only_verified_non_gaia_rows(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join([
            json.dumps({"id": "ok", "source": "synthetic", "license": "CC0-1.0", "generator": "unit", "generator_version": "1", "oracle_type": "exact", "oracle_version": "1", "required_tools": [], "question": "q1", "expected_answer": "yes"}),
            json.dumps({"id": "bad", "source": "synthetic", "license": "CC0-1.0", "generator": "unit", "generator_version": "1", "oracle_type": "exact", "oracle_version": "1", "required_tools": [], "question": "q2", "expected_answer": "no"}),
            json.dumps({"id": "gaia", "source": "GAIA-validation", "license": "CC0-1.0", "generator": "unit", "generator_version": "1", "oracle_type": "exact", "oracle_version": "1", "required_tools": [], "question": "q3", "expected_answer": "yes"}),
        ]) + "\n",
        encoding="utf-8",
    )

    class Runtime:
        tools = {}
        def run(self, question, workspace):
            answer = "yes"
            return AgentResult(answer, True, "final", [TraceEvent("final", 1, {"answer": answer})], RunMetrics(steps=1))

    output = tmp_path / "verified.jsonl"
    summary = collect_verified_trajectories(tasks, output, Runtime, tmp_path / "runs")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in rows] == ["ok"]
    assert summary == {"total": 3, "verified": 1, "failed": 1, "rejected_gaia": 1}


def test_correct_final_answer_is_not_verified_when_policy_trace_is_dirty():
    tools = [{"type": "function", "function": {"name": "python"}}]
    base = [
        TraceEvent("tool_call", 1, {"id": "c1", "name": "python", "arguments": {"code": "print(2+2)"}}),
        TraceEvent("tool_result", 1, {"id": "c1", "name": "python", "ok": False, "error_code": "DUPLICATE_CALL", "content": "blocked"}),
        TraceEvent("final", 2, {"answer": "4"}),
    ]
    result = AgentResult("4", True, "final", base, RunMetrics(steps=2, tool_calls=1, tool_errors=1, duplicate_calls_blocked=1))
    row = trajectory_from_result(
        task_id="dirty", source="synthetic", question="Compute 2+2", expected_answer="4",
        result=result, tools=tools, required_tools=["python"],
        provenance={"license":"CC0-1.0","generator":"unit","generator_version":"1","oracle_type":"exact","oracle_version":"1"},
    )
    assert row["verified"] is False
    assert row["verification"]["final_answer_correct"] is True
    assert row["verification"]["zero_tool_errors"] is False
    assert row["verification"]["zero_duplicate_blocks"] is False


def test_required_tool_must_actually_be_used_for_training_gold():
    result = AgentResult("4", True, "final", [TraceEvent("final", 1, {"answer": "4"})], RunMetrics(steps=1))
    row = trajectory_from_result(
        task_id="missing-tool", source="synthetic", question="Compute 2+2", expected_answer="4",
        result=result, tools=[], required_tools=["python"],
        provenance={"license":"CC0-1.0","generator":"unit","generator_version":"1","oracle_type":"exact","oracle_version":"1"},
    )
    assert row["verified"] is False
    assert row["verification"]["required_tools_satisfied"] is False


def test_trajectory_is_not_verified_without_complete_provenance():
    result = AgentResult("4", True, "final", [TraceEvent("final", 1, {"answer": "4"})], RunMetrics(steps=1))
    row = trajectory_from_result(
        task_id="missing-provenance", source="synthetic", question="2+2", expected_answer="4",
        result=result, tools=[], required_tools=[], provenance={},
    )
    assert row["verified"] is False
    assert row["verification"]["provenance_complete"] is False
