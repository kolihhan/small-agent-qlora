import json

from gaia_small_agent.agent.types import AssistantTurn, ToolCall


def test_next_action_examples_include_exact_forensic_payloads(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    path = tmp_path / "rows.jsonl"
    row = {
        "task_id": "recovery",
        "capability": "failure_recovery",
        "tools": [],
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "x",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "MIRROR current release code", "max_results": 5},
                    },
                }],
            },
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    class Model:
        def complete(self, messages, tools):
            return AssistantTurn(
                "",
                [ToolCall("p", "search", {"query": "MIRROR release code", "max_results": 5})],
            )

    summary = evaluate_next_actions(path, Model)
    example = summary["examples"][0]
    assert example["error"] == "wrong_arguments"
    assert example["message_index"] == 1
    assert example["expected_tool_calls_full"] == [{
        "name": "search",
        "arguments": {"max_results": 5, "query": "MIRROR current release code"},
    }]
    assert example["predicted_tool_calls_full"] == [{
        "name": "search",
        "arguments": {"max_results": 5, "query": "MIRROR release code"},
    }]
    assert example["expected_final"] is None
    assert example["predicted_final"] == ""


def test_next_action_forensics_preserve_final_text(tmp_path):
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions

    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({
        "task_id": "final",
        "capability": "evidence_to_final",
        "tools": [],
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "CODE-777"},
        ],
    }) + "\n", encoding="utf-8")

    class Model:
        def complete(self, messages, tools):
            return AssistantTurn("wrong")

    example = evaluate_next_actions(path, Model)["examples"][0]
    assert example["error"] == "wrong_final"
    assert example["expected_final"] == "CODE-777"
    assert example["predicted_final"] == "wrong"
    assert example["expected_tool_calls_full"] == []
    assert example["predicted_tool_calls_full"] == []
