from __future__ import annotations

import hashlib

import pytest

from gaia_small_agent.agent.loop import AgentRuntime, FINALIZATION_NUDGE
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED
from gaia_small_agent.tools.base import Tool, ToolResult

from scripts.gaia_fast_pair import (
    ARMS, EXPECTED_IDS_SHA256, N_SHARDS, aggregate, markdown_report,
)


class EchoTool(Tool):
    name = "echo"
    description = "Return a string"
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"]}

    def run(self, arguments, workspace):
        return ToolResult(ok=True, content=arguments["text"])


class RecordingModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(("normal", messages[0]["content"], len(tools)))
        step = len(self.calls)
        if step <= 2:
            return AssistantTurn("", [ToolCall(str(step), "echo", {"text": str(step)})])
        return AssistantTurn("correct answer")

    def complete_final(self, messages, tools):
        self.calls.append(("final", messages[0]["content"], len(tools)))
        return AssistantTurn("correct answer")


@pytest.mark.parametrize("nudge", [False, True])
def test_late_final_nudge_is_opt_in_and_final_call_exists(tmp_path, nudge):
    model = RecordingModel()
    result = AgentRuntime(
        model, [EchoTool()], max_steps=3, finalization_nudge=nudge
    ).run("give answer", tmp_path)
    assert result.completed and result.answer == "correct answer"
    assert result.metrics.steps == 3
    assert [entry[0] for entry in model.calls] == ["normal", "normal", "final"]
    assert model.calls[0][2] == 1
    assert model.calls[1][2] == 1
    assert model.calls[2][0] == "final"
    assert (FINALIZATION_NUDGE in model.calls[0][1]) is False
    assert (FINALIZATION_NUDGE in model.calls[1][1]) is nudge
    assert (FINALIZATION_NUDGE in model.calls[2][1]) is nudge


def test_early_correct_final_untouched(tmp_path):
    class Immediate:
        def complete(self, messages, tools):
            assert FINALIZATION_NUDGE not in messages[0]["content"]
            return AssistantTurn("short")

    result = AgentRuntime(
        Immediate(), [], max_steps=12, finalization_nudge=True
    ).run("question", tmp_path)
    assert result.completed and result.metrics.steps == 1


def row(index, *, correct, stop="final"):
    return {
        "task_hash": hashlib.sha256(f"test-task-{index}".encode()).hexdigest(),
        "level": 1, "correct": correct, "completed": stop == "final",
        "stop_reason": stop, "invalid_final": stop == "invalid_final",
        "max_steps_stop": False, "steps": 12, "tool_calls": 1,
        "tool_errors": 0, "duplicate_calls_blocked": 0, "latency_ms": 12,
    }


def fake_shards():
    shards = []
    for shard in range(N_SHARDS):
        baseline, candidate = [], []
        for offset in range(5):
            i = shard * 5 + offset
            baseline.append(row(i, correct=i in {1, 2}))
            candidate.append(row(i, correct=i in {0, 2}))
        shards.append({
            "schema": "gaia-fast-v3b-paired/v1", "shard": shard,
            "source_revision": GAIA_REVISION,
            "selection": "gaia-evaluation100-seen-dev",
            "full_ids_sha256": EXPECTED_IDS_SHA256,
            "seed": LOCAL_PROTOCOL_SEED, "max_steps": 12,
            "same_v3b_adapter": True, "no_question_answer_trace_export": True,
            "arms": {"baseline": baseline, "candidate": candidate},
        })
    return shards


def test_aggregate_100_paired_rows_and_win_loss_table():
    summary = aggregate(fake_shards())
    assert summary["n"] == 100
    assert summary["arms"]["baseline"]["correct"] == 2
    assert summary["arms"]["candidate"]["correct"] == 2
    assert len(summary["wins"]) == 1
    assert len(summary["losses"]) == 1
    assert len(summary["case_table"]) == 100
    report = markdown_report(summary)
    assert "Candidate wins (1)" in report
    assert "Candidate losses (1)" in report


def test_aggregate_fails_closed_on_missing_shard_or_mismatched_pair():
    shards = fake_shards()
    with pytest.raises(ValueError, match="20-shard"):
        aggregate(shards[:-1])
    shards[0]["arms"]["candidate"][0]["task_hash"] = "bad"
    with pytest.raises(ValueError, match="mismatch"):
        aggregate(shards)
