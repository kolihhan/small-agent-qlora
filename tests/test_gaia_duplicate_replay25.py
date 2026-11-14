import pytest

from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION
from scripts.gaia_duplicate_replay25 import aggregate


def shard(i):
    original = []
    replay = []
    for j in range(5):
        task = {"task_hash": f"task-{i}-{j}", "correct": False,
                "completed": False, "stop_reason": "invalid_final",
                "duplicates": 1, "tool_calls": 4, "tool_errors": 0,
                "latency_ms": 2000.0}
        original.append(dict(task))
        replay.append({**task, "correct": True, "completed": True,
                       "stop_reason": "final"})
    return {"shard": i, "cohort_sha256": "frozen",
            "gaia_revision": GAIA_REVISION,
            "arms": {"original": original, "replay_success": replay}}


def test_aggregate_requires_real_paired_25_not_partial():
    out = aggregate([shard(i) for i in range(5)])
    assert out["n"] == 25
    assert out["rescues"] == 25 and out["regressions"] == 0
    assert out["arms"]["original"]["tool_calls"] == 100
    with pytest.raises(ValueError, match="five unique"):
        aggregate([shard(0)])


def test_aggregate_rejects_pair_mismatch():
    inputs = [shard(i) for i in range(5)]
    inputs[0]["arms"]["replay_success"][0]["task_hash"] = "different"
    with pytest.raises(ValueError, match="matched cases"):
        aggregate(inputs)
