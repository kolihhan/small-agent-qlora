"""Synthetic only: prove 2048-token filtering never makes partial tool traces."""
from copy import deepcopy

import pytest

from scripts.prepare_real_2048_complete_tasks import filter_whole_tasks


class CharTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(text)}


def example(tid, index, kind, prompt_chars, completion_chars):
    return {
        "prompt": "p" * prompt_chars,
        "completion": "c" * completion_chars,
        "metadata": {
            "task_id": tid, "assistant_message_index": index,
            "action_type": kind,
        },
    }


def synthetic():
    rows = [
        {"task_id": "a", "source": "hotpotqa-distractor-train",
         "training_action_indices": [2, 4]},
        {"task_id": "b", "source": "2wikimultihopqa-train",
         "training_action_indices": [2, 4]},
    ]
    examples = [
        example("a", 2, "tool_call", 40, 10),
        example("a", 4, "final", 75, 15),
        example("b", 2, "tool_call", 40, 10),
        example("b", 4, "final", 94, 10),
    ]
    return rows, examples


def test_complete_parent_filter_preserves_all_action_and_final():
    rows, examples = synthetic()
    kept, report = filter_whole_tasks(rows, examples, CharTokenizer(), max_length=100)
    assert [r["task_id"] for r in kept] == ["a"]
    assert report["retained_unique_tasks"] == 1
    assert report["retained_native_actions"] == 2
    assert report["retained_tool_calls"] == 1
    assert report["retained_final_answers"] == 1
    assert report["rejected_actions_with_parent"] == 2
    assert report["excluded_unique_tasks"] == 1
    assert report["training_ran"] is False
    assert report["gaia_evaluation_ran"] is False


def test_wider_context_retains_both_whole_tasks_no_relabel():
    rows, examples = synthetic()
    kept, report = filter_whole_tasks(rows, examples, CharTokenizer(), max_length=2048)
    assert kept == rows
    assert report["retained_native_actions"] == 4
    assert report["retained_final_answers"] == 2
    assert report["source_task_counts"] == {
        "2wikimultihopqa-train": 1, "hotpotqa-distractor-train": 1,
    }


@pytest.mark.parametrize("defect", [
    "duplicate_task", "missing_action", "missing_final", "duplicate_final",
    "wrong_index", "unknown_task", "non_train_source", "empty_completion",
])
def test_corrupt_task_structure_rejected_before_sft(defect):
    rows, examples = synthetic()
    if defect == "duplicate_task":
        rows[1]["task_id"] = "a"
    elif defect == "missing_action":
        examples.pop(1)
    elif defect == "missing_final":
        examples[1]["metadata"]["action_type"] = "tool_call"
    elif defect == "duplicate_final":
        examples[0]["metadata"]["action_type"] = "final"
    elif defect == "wrong_index":
        examples[1]["metadata"]["assistant_message_index"] = 6
    elif defect == "unknown_task":
        examples[0]["metadata"]["task_id"] = "invented"
    elif defect == "non_train_source":
        rows[0]["source"] = "hotpotqa-distractor-validation"
    elif defect == "empty_completion":
        examples[0]["completion"] = ""
    with pytest.raises(ValueError):
        filter_whole_tasks(rows, examples, CharTokenizer(), max_length=2048)


def test_invalid_or_empty_budget_refused():
    rows, examples = synthetic()
    for budget in (0, -1):
        with pytest.raises(ValueError):
            filter_whole_tasks(rows, examples, CharTokenizer(), max_length=budget)
    with pytest.raises(ValueError):
        filter_whole_tasks([], [], CharTokenizer(), max_length=100)
