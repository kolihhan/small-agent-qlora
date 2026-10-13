import json

from gaia_small_agent.agent.types import AssistantTurn, ToolCall


def _write_rows(path):
    rows = [
        {
            "task_id": "tool-case",
            "capability": "tool_selection",
            "tools": [{"type": "function", "function": {"name": "python", "description": "calc", "parameters": {"type": "object"}}}],
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "compute 6*7"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "oracle-1", "type": "function", "function": {"name": "python", "arguments": {"code": "print(6 * 7)"}}}]},
                {"role": "tool", "tool_call_id": "oracle-1", "name": "python", "content": "42"},
                {"role": "assistant", "content": "42"},
            ],
        },
        {
            "task_id": "final-case",
            "capability": "evidence_to_final",
            "tools": [],
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "answer from evidence"},
                {"role": "assistant", "content": "done"},
            ],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_next_action_eval_scores_tool_arguments_and_final_semantics(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    rows = tmp_path / "test.jsonl"
    _write_rows(rows)
    turns = [AssistantTurn("", [ToolCall("pred-1", "python", {"code": "print(6 * 7)"})]), AssistantTurn("42"), AssistantTurn("done")]

    class Model:
        def complete(self, messages, tools):
            return turns.pop(0)

    summary = evaluate_next_actions(rows, lambda: Model())
    assert summary["total"] == 3
    assert summary["correct"] == 3
    assert summary["accuracy"] == 100.0
    assert summary["by_action_type"]["tool_call"] == {"total": 1, "correct": 1}
    assert summary["by_action_type"]["final"] == {"total": 2, "correct": 2}
    assert summary["by_capability"]["tool_selection"] == {"total": 2, "correct": 2}
    assert summary["by_capability"]["evidence_to_final"] == {"total": 1, "correct": 1}
    assert len(summary["examples"]) == 3
    assert summary["examples"][0]["example_id"] == "tool-case:2"


def test_next_action_eval_distinguishes_wrong_tool_args_from_wrong_action_type(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    rows = tmp_path / "test.jsonl"
    _write_rows(rows)
    turns = [AssistantTurn("", [ToolCall("pred-1", "python", {"code": "print(6 + 7)"})]), AssistantTurn("42"), AssistantTurn("", [ToolCall("pred-2", "python", {"code": "print(1)"})])]

    class Model:
        def complete(self, messages, tools):
            return turns.pop(0)

    summary = evaluate_next_actions(rows, lambda: Model())
    assert summary["correct"] == 1
    assert summary["error_counts"] == {"wrong_arguments": 1, "wrong_action_type": 1}
    assert summary["examples"][0]["error"] == "wrong_arguments"
    assert summary["examples"][2]["error"] == "wrong_action_type"


def test_next_action_eval_can_score_a_frozen_subset_of_example_ids(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    rows = tmp_path / "test.jsonl"
    _write_rows(rows)
    turns = [AssistantTurn("42")]

    class Model:
        def complete(self, messages, tools):
            return turns.pop(0)

    summary = evaluate_next_actions(rows, lambda: Model(), example_ids={"tool-case:4"})
    assert summary["total"] == 1
    assert summary["correct"] == 1
    assert [item["example_id"] for item in summary["examples"]] == ["tool-case:4"]


def test_next_action_eval_rejects_rows_without_assistant_targets(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"task_id": "bad", "capability": "x", "tools": [], "messages": [{"role": "user", "content": "q"}]}) + "\n", encoding="utf-8")
    try:
        evaluate_next_actions(path, lambda: object())
    except ValueError as exc:
        assert "assistant target" in str(exc)
    else:
        raise AssertionError("expected ValueError")
