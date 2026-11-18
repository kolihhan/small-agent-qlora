from __future__ import annotations

import pytest
from scripts.freeze_native_gaia_actions import eligible_actions, freeze


def row(source, i):
    call = {"id": f"call-{i}", "type": "function",
            "function": {"name": "read", "arguments": {"source": "docs/a.txt"}}}
    return {
        "task_id": f"{source}-{i}", "source": source, "verified": True,
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
        "messages": [
            {"role": "system", "content": "Tool user"},
            {"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "read", "content": "Real text."},
            {"role": "assistant", "content": "Final answer"},
        ],
        "training_action_indices": [2, 4],
    }


def test_balanced_actions_keep_one_source_parent_each():
    rows = [row(source, i) for source in ("2wikimultihopqa-train", "hotpotqa-distractor-train")
            for i in range(8)]
    result, report = freeze(rows, per_kind=4)
    assert len(result) == 8
    assert len({r["task_id"] for r in result}) == 8
    assert report["selected_actions"] == {"tool_call": 4, "final": 4}
    assert report["GAIA_supervision"] == 0
    assert all(len(r["training_action_indices"]) == 1 for r in result)


def test_fails_on_missing_tool_observation_and_non_native_action():
    broken = row("2wikimultihopqa-train", 1)
    broken["messages"][3]["tool_call_id"] = "fake-call"
    assert not eligible_actions([broken])["final"]
    nonnative = row("2wikimultihopqa-train", 2)
    nonnative["messages"][2]["tool_calls"][0]["function"]["name"] = "fetch"
    assert not eligible_actions([nonnative])["tool_call"]


def test_gaia_excluded_and_duplicate_parents_rejected():
    assert not any(eligible_actions([row("gaia-eval", 3)]).values())
    duplicate = row("2wikimultihopqa-train", 4)
    with pytest.raises(ValueError, match="duplicate"):
        eligible_actions([duplicate, duplicate])
