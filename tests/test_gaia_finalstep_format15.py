import pytest

from scripts.gaia_finalstep_format15 import aggregate, final_only_messages


def test_format_prompt_only_modifies_first_system_message():
    rows = [
        {"role": "system", "content": "You are Small Agent."},
        {"role": "user", "content": "Task"},
        {"role": "tool", "tool_call_id": "id", "content": "Observed data"},
    ]
    changed = final_only_messages(rows)
    assert changed[1:] == rows[1:]
    assert changed[0]["content"].startswith("You are Small Agent.")
    assert "output format exactly" in changed[0]["content"]
    assert rows[0]["content"] == "You are Small Agent."


def _fake(shard):
    cases = []
    for i in range(3):
        fields = {
            "correct": False, "completed": False,
            "tool_calls": 3, "tool_errors": 0,
            "duplicate_calls_blocked": 0, "steps": 12,
            "stop_reason": "invalid_final",
            "latency_ms": 2000.0, "terminal_ms": 1000.0,
        }
        cases.append({
            "task_hash": f"id-{shard}-{i}",
            "level": 1,
            "legacy": dict(fields),
            "format_final_only": {**fields, "correct": True,
                                  "completed": True, "stop_reason": "final"},
        })
    return {"shard": shard, "cases": cases}


def test_full_aggregate_and_causal_pairing():
    out = aggregate([_fake(i) for i in range(5)])
    assert out["cases"] == 15
    assert out["paired_rescues"] == 15
    assert out["paired_regressions"] == 0
    assert out["arms"]["format_final_only"]["correct"] == 15
    assert out["arms"]["legacy"]["tool_calls"] == 45
    assert out["arms"]["format_final_only"]["tool_calls"] == 45


def test_fails_closed_when_tool_behavior_differs_or_shard_missing():
    with pytest.raises(ValueError, match="five"):
        aggregate([_fake(0)])
    cases = [_fake(i) for i in range(5)]
    cases[0]["cases"][0]["format_final_only"]["tool_calls"] += 1
    with pytest.raises(ValueError, match="pre-terminal"):
        aggregate(cases)
