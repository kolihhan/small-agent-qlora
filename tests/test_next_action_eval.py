import json

from gaia_small_agent.agent.types import AssistantTurn, ToolCall


def _row(messages, *, task_id="task-1", capability="multi_step"):
    return {
        "task_id": task_id,
        "capability": capability,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}},
                        "required": ["source"],
                    },
                },
            }
        ],
        "messages": messages,
    }


class QueueModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return self.turns.pop(0)


def test_next_action_eval_uses_teacher_forced_prefixes_and_scores_tool_then_final(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_action_tasks

    row = _row(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "oracle-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": {"source": "fact.txt"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "oracle-1", "name": "read", "content": "VALUE=42"},
            {"role": "assistant", "content": "42"},
        ]
    )
    tasks = tmp_path / "test.jsonl"
    tasks.write_text(json.dumps(row) + "\n", encoding="utf-8")
    model = QueueModel(
        [
            AssistantTurn("", [ToolCall("pred-1", "read", {"source": "fact.txt"})]),
            AssistantTurn("42", []),
        ]
    )

    result = evaluate_next_action_tasks(tasks, model)

    assert result["summary"]["total_decisions"] == 2
    assert result["summary"]["exact_decisions"] == 2
    assert result["summary"]["exact_accuracy"] == 1.0
    assert result["summary"]["action_type_accuracy"] == 1.0
    assert result["summary"]["tool_name_accuracy"] == 1.0
    assert result["summary"]["tool_arguments_accuracy"] == 1.0
    assert result["summary"]["final_accuracy"] == 1.0
    assert result["summary"]["by_capability"]["multi_step"]["exact_decisions"] == 2
    assert len(result["decisions"]) == 2

    assert model.calls[0][0] == row["messages"][:2]
    assert model.calls[1][0] == row["messages"][:4]
    assert model.calls[0][1] == row["tools"]


def test_next_action_eval_separates_tool_name_from_argument_correctness(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_action_tasks

    row = _row(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "read the authoritative file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "oracle-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": {"source": "current.txt"}},
                    }
                ],
            },
        ],
        capability="argument_grounding",
    )
    tasks = tmp_path / "test.jsonl"
    tasks.write_text(json.dumps(row) + "\n", encoding="utf-8")
    model = QueueModel([AssistantTurn("", [ToolCall("pred-1", "read", {"source": "old.txt"})])])

    result = evaluate_next_action_tasks(tasks, model)
    summary = result["summary"]

    assert summary["total_decisions"] == 1
    assert summary["action_type_correct"] == 1
    assert summary["tool_name_correct"] == 1
    assert summary["tool_arguments_correct"] == 0
    assert summary["exact_decisions"] == 0
    assert result["decisions"][0]["expected_tool_names"] == ["read"]
    assert result["decisions"][0]["predicted_tool_names"] == ["read"]


def test_next_action_comparison_reports_paired_transitions_and_exact_mcnemar():
    from gaia_small_agent.training.next_action_eval import compare_next_action_results

    base = {
        "decisions": [
            {"decision_id": "a:2", "exact": True},
            {"decision_id": "b:2", "exact": True},
            {"decision_id": "c:2", "exact": False},
            {"decision_id": "d:2", "exact": False},
        ]
    }
    candidate = {
        "decisions": [
            {"decision_id": "a:2", "exact": True},
            {"decision_id": "b:2", "exact": False},
            {"decision_id": "c:2", "exact": True},
            {"decision_id": "d:2", "exact": True},
        ]
    }

    result = compare_next_action_results(base, candidate)

    assert result["total_decisions"] == 4
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 0
    assert result["base_only"] == 1
    assert result["candidate_only"] == 2
    assert result["net_exact_change"] == 1
    assert result["mcnemar_exact_p"] == 1.0
