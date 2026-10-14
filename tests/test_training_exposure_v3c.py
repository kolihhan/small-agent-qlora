import json
from collections import Counter


def test_v3c_selects_complete_oracle_trajectories_deterministically(tmp_path):
    from gaia_small_agent.training.exposure_v3c import (
        V3C_TRAJECTORY_QUOTAS,
        select_v3c_trajectories,
        summarize_v3c_selection,
    )
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset

    paths = generate_oracle_policy_v3_dataset(tmp_path / "oracle", seed="small-agent-policy-v3")
    rows = [json.loads(line) for line in paths["train"].read_text(encoding="utf-8").splitlines() if line.strip()]

    selected = select_v3c_trajectories(rows, seed="qlora-v3c-full-trajectory")
    repeated = select_v3c_trajectories(rows, seed="qlora-v3c-full-trajectory")

    assert [row["task_id"] for row in selected] == [row["task_id"] for row in repeated]
    assert len(selected) == sum(V3C_TRAJECTORY_QUOTAS.values()) == 54
    assert len({row["task_id"] for row in selected}) == 54
    assert Counter(row["capability"] for row in selected) == Counter(V3C_TRAJECTORY_QUOTAS)

    report = summarize_v3c_selection(selected)
    assert report["selected_task_count"] == 54
    assert report["selected_example_count"] == 128
    assert report["action_type_counts"] == {"final": 54, "tool_call": 74}
    assert report["capability_example_counts"] == {
        "argument_grounding": 12,
        "duplicate_avoidance": 4,
        "evidence_to_final": 20,
        "failure_recovery": 36,
        "multi_step": 24,
        "no_tool_stop": 6,
        "strategy_switch": 18,
        "tool_selection": 8,
    }

    selected_by_id = {row["task_id"]: row for row in selected}
    for task_id, row in selected_by_id.items():
        expected_turns = sum(1 for message in row["messages"] if message.get("role") == "assistant")
        assert expected_turns in {1, 2, 3}, task_id
