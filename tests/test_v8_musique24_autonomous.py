import pytest
from scripts.eval_v8_musique24_autonomous import paired_results


def _case(i, correct, hop=4):
    return {
        "task_id": f"held-{i}", "hop_count": hop, "correct": bool(correct),
        "completed": True, "stop_reason": "final",
        "tool_calls": i+1, "duplicate_calls_blocked": i % 2,
    }


def test_autonomous_paired_rescues_regressions_and_costs():
    base = [_case(i, i < 10) for i in range(24)]
    new = [_case(i, i < 10 or i in (10, 11)) for i in range(24)]
    new[0]["correct"] = False
    report = paired_results(base, new)
    assert report["total"] == 24
    assert (report["base_correct"], report["v8_correct"]) == (10, 11)
    assert (report["rescues"], report["regressions"]) == (2, 1)
    assert report["base_calls"] == report["v8_calls"] == sum(range(1, 25))
    assert report["promotion_allowed"] is False


def test_autonomous_paired_missing_tasks_and_horizon_drift_fail_closed():
    base = [_case(i, True) for i in range(3)]
    with pytest.raises(ValueError, match="coverage"):
        paired_results(base, base[:2])
    changed = [_case(i, True, hop=3) for i in range(3)]
    with pytest.raises(ValueError, match="hop/source"):
        paired_results(base, changed)
