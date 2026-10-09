from gaia_small_agent.training.cpu_policy_pilot import (
    TRAIN_MAX_LENGTH,
    build_frozen_task_sets,
    oracle_trajectory,
    summarize_pair,
)


def test_frozen_pilot_splits_are_disjoint_and_fixed_size(tmp_path):
    train, evaluation = build_frozen_task_sets(tmp_path)

    assert len(train) == 16
    assert len(evaluation) == 12
    assert [row["id"] for row in train] == [f"policy-{i:03d}" for i in range(1, 17)]
    assert [row["id"] for row in evaluation] == [f"policy-{i:03d}" for i in range(17, 29)]
    assert {row["question"] for row in train}.isdisjoint({row["question"] for row in evaluation})


def test_frozen_pilot_uses_measured_minimum_safe_training_budget():
    assert TRAIN_MAX_LENGTH == 768


def test_oracle_trajectory_is_verified_and_uses_required_tool(tmp_path):
    train, _ = build_frozen_task_sets(tmp_path / "sets")

    for task in train[:4]:
        row = oracle_trajectory(task, tmp_path / task["id"])
        assert row["verified"] is True
        assert row["required_tools"] == task["required_tools"]
        assistant_calls = [
            call["function"]["name"]
            for message in row["messages"]
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
        ]
        assert assistant_calls == task["required_tools"]
        assert row["messages"][-1]["content"] == task["expected_answer"]


def test_pair_summary_reports_accuracy_tool_use_and_transitions():
    base = [
        {"task_id": "a", "correct": False, "completed": True, "required_tool_used": True},
        {"task_id": "b", "correct": True, "completed": True, "required_tool_used": False},
        {"task_id": "c", "correct": True, "completed": True, "required_tool_used": True},
        {"task_id": "d", "correct": False, "completed": False, "required_tool_used": False},
    ]
    lora = [
        {"task_id": "a", "correct": True, "completed": True, "required_tool_used": True},
        {"task_id": "b", "correct": False, "completed": True, "required_tool_used": True},
        {"task_id": "c", "correct": True, "completed": True, "required_tool_used": True},
        {"task_id": "d", "correct": False, "completed": True, "required_tool_used": True},
    ]

    summary = summarize_pair(base, lora)

    assert summary["base"]["exact_correct"] == 2
    assert summary["lora"]["exact_correct"] == 2
    assert summary["base"]["required_tool_used"] == 2
    assert summary["lora"]["required_tool_used"] == 4
    assert summary["transitions"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "correct_to_correct": 1,
        "wrong_to_wrong": 1,
    }
    assert summary["verdict"] == "PILOT_NO_GAIN"
