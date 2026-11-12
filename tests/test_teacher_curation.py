from __future__ import annotations

from copy import deepcopy
import json

import pytest

from gaia_small_agent.training.teacher_curation import curate_teacher_rows, _check_row
from gaia_small_agent.training.qlora import _render_turn_examples


class NativeShapeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        def dump(v):
            return json.dumps(v, sort_keys=True, separators=(",", ":"))
        if kwargs.get("add_generation_prompt"):
            return dump(messages) + "<assistant>"
        if messages and messages[-1].get("role") == "assistant":
            return dump(messages[:-1]) + "<assistant>" + dump(messages[-1])
        return dump(messages)


def make_row(task_id="task-a::sample-1", ntools=3):
    messages = [
        {"role": "system", "content": "Use trustworthy observations only"},
        {"role": "user", "content": "Find the current archive token through linked documents."},
    ]
    for step in range(ntools):
        cid = f"t{step}"
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "read", "arguments": {"source": f"file-{step}.txt"}}}
            ]},
            {"role": "tool", "tool_call_id": cid, "name": "read", "content": f"file-{step + 1}.txt; verified token CANON-997"},
        ]
    messages.append({"role": "assistant", "content": "CANON-997"})
    return {
        "task_id": task_id, "source": "independent-offline-tool-rollout",
        "expected_answer": "CANON-997", "verified": True,
        "provenance": {"license": "CC0-1.0", "generator": "test-teacher", "generator_version": "1",
                       "oracle_type": "exact", "oracle_version": "1"},
        "verification": {"completed": True, "final_answer_correct": True, "zero_duplicate_blocks": True,
                         "required_tools_satisfied": True, "provenance_complete": True,
                         "zero_tool_errors": True},
        "metrics": {"tool_calls": ntools, "tool_errors": 0, "duplicate_calls_blocked": 0},
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "messages": messages,
    }


def test_rejection_sampling_picks_shortest_clean_teacher_with_same_task():
    longer = make_row("task-a::sample-1", ntools=4)
    shorter = make_row("task-a::sample-2", ntools=3)
    result, stats = curate_teacher_rows(
        [longer, shorter], protected_hashes={"nonempty-protected-hash"},
        holdout_hashes=set(), min_actions=3,
    )
    assert len(result) == 1
    assert result[0]["task_id"] == "task-a"
    assert result[0]["selection_provenance"]["sample_id"] == "task-a::sample-2"
    assert result[0]["selection_provenance"]["candidate_count"] == 2
    assert result[0]["training_action_indices"] == [2, 4, 6, 8]
    assert stats["input_rollouts"] == 2
    assert stats["accepted_tasks"] == 1
    assert stats["trainable_actions"] == 4


def test_curated_rows_train_only_valid_assistant_indices(tmp_path):
    row, _ = curate_teacher_rows(
        [make_row()], protected_hashes={"protect"}, holdout_hashes=set(),
    )
    examples = _render_turn_examples(NativeShapeTokenizer(), row)
    assert len(examples) == 4
    assert [r["metadata"]["assistant_message_index"] for r in examples] == [2, 4, 6, 8]
    row[0]["training_action_indices"] = [2, 4, 6]
    assert len(_render_turn_examples(NativeShapeTokenizer(), row)) == 3
    row[0]["training_action_indices"] = [3]
    with pytest.raises(ValueError, match="training_action_indices"):
        _render_turn_examples(NativeShapeTokenizer(), row)


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.update({"source": "GAIA-validation"}), "forbidden_source"),
    (lambda r: r["metrics"].update({"duplicate_calls_blocked": 1}), "duplicate_tool_calls"),
    (lambda r: r["messages"].__setitem__(-1, {"role": "assistant", "content": "WRONG"}), "wrong_final"),
    (lambda r: r["messages"].__setitem__(3, {"role": "tool", "tool_call_id": "fake", "name": "read", "content": "x"}), "mismatched_tool_result"),
    (lambda r: r["messages"][2]["tool_calls"][0]["function"].update({"name": "python"}), "unexpected_tool"),
    (lambda r: r["messages"][3].update({"content": "ToolError[READ_ERROR]: fail"}), "failed_tool_action"),
    (lambda r: r["messages"][4]["tool_calls"][0]["function"]["arguments"].update({"source": "file-0.txt"}), "duplicate_tool_call"),
])
def test_rejects_bad_trajectory_even_if_verified_flag_true(mutation, reason):
    row = make_row()
    mutation(row)
    with pytest.raises(ValueError, match=reason):
        _check_row(row, {"protected"}, set(), min_actions=3)


def test_training_protection_and_holdout_are_enforced():
    from gaia_small_agent.training.protection import question_hash
    row = make_row()
    question = row["messages"][1]["content"]
    with pytest.raises(ValueError, match="protected_or_holdout_question"):
        _check_row(row, {question_hash(question)}, set(), 3)
    with pytest.raises(ValueError, match="protected_or_holdout_question"):
        _check_row(row, {"other"}, {question_hash(question)}, 3)
    with pytest.raises(ValueError, match="GAIA-protected"):
        curate_teacher_rows([row], protected_hashes=set(), holdout_hashes=set())


def test_min_depth_gate_fails_and_only_one_unique_question_can_enter_training():
    shallow = make_row(ntools=2)
    with pytest.raises(ValueError, match="insufficient verified long trajectories"):
        curate_teacher_rows([shallow], protected_hashes={"notempty"}, holdout_hashes=set(), min_actions=3)
    same_question = make_row("another-task::sample-1")
    selected, stats = curate_teacher_rows(
        [make_row(), same_question], protected_hashes={"notempty"}, holdout_hashes=set()
    )
    assert len(selected) == 1
    assert stats["rejected_reasons"]["duplicate_question_across_tasks"] == 1
