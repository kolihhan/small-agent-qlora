"""A fixed same-Qwen holdout protocol; NEVER a GAIA replacement."""
from gaia_small_agent.training.v8_holdout import select_v8_targets, paired_v8_actions


def rows():
    output = []
    for i in range(35):
        output.append({
            "task_id": f"task-{i}", "source": "musique-train" if i < 24 else "hotpotqa-distractor-train",
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": {"source": "a"}}}]},
                {"role": "tool", "content": "evidence"},
                {"role": "assistant", "content": f"answer {i}"},
            ],
            "training_action_indices": [1, 3],
        })
    return output


def test_target_selector_is_35_parent_70_action_source_preserving():
    first = select_v8_targets(rows())
    assert len(first) == 70 and len(set(first)) == 70
    assert sum(t["action_type"] == "final" for t in first.values()) == 35
    assert sum(t["action_type"] == "tool_call" for t in first.values()) == 35
    assert len({t["task_id"] for t in first.values()}) == 35
    assert first == select_v8_targets(rows())


def test_target_selector_fails_closed_on_source_or_bad_history():
    import pytest
    with pytest.raises(ValueError):
        select_v8_targets(rows()[:34])
    copied = rows()
    copied[0]["messages"][1]["tool_calls"] = []
    with pytest.raises(ValueError):
        select_v8_targets(copied)


def test_paired_strict_scoring_tracks_rescues_regressions_and_action_groups():
    selected = select_v8_targets(rows())
    base = {"examples": [{"example_id": i, "correct": i.endswith(":3")}
                         for i in selected]}
    adapter = {"examples": [{"example_id": i, "correct": i.endswith(":3") or i == "task-0:1"}
                            for i in selected]}
    result = paired_v8_actions(selected, base, adapter)
    assert result["total"] == 70
    assert result["base_correct"] == 35
    assert result["adapter_correct"] == 36
    assert result["rescues"] == 1 and result["regressions"] == 0
    assert result["per_action_type"]["final"]["total"] == 35
    assert result["per_action_type"]["tool_call"]["total"] == 35
    assert result["promotion_allowed"] is False


def test_paired_scorer_rejects_missing_cases_duplicate_ids_and_modified_source():
    import pytest
    selected = select_v8_targets(rows())
    base = {"examples": [{"example_id": i, "correct": False} for i in selected]}
    adapter = {"examples": [{"example_id": i, "correct": True} for i in selected]}
    with pytest.raises(ValueError, match="coverage"):
        paired_v8_actions(selected, base, {"examples": adapter["examples"][:-1]})
    with pytest.raises(ValueError, match="duplicate"):
        paired_v8_actions(selected, base, {"examples": adapter["examples"] + adapter["examples"][:1]})
