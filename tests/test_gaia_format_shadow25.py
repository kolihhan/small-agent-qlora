import pytest

from gaia_small_agent.benchmark.gaia100 import select_gaia_partition
from scripts.gaia_format_shadow25 import aggregate, select_shadow


def row(index, level):
    return {"task_id": f"case-{level}-{index}", "Level": level}


def test_shadow25_does_not_overlap_diagnostic_or_evaluation():
    population = ([row(i, 1) for i in range(60)]
                  + [row(i, 2) for i in range(90)]
                  + [row(i, 3) for i in range(30)])
    selected = select_shadow(population)
    excluded = {
        r["task_id"] for part in ("diagnostic", "evaluation")
        for r in select_gaia_partition(population, part)
    }
    assert len(selected) == 25
    assert not excluded.intersection(r["task_id"] for r in selected)


def fake_shard(shard):
    arms = {}
    for arm in ("original", "format_reminder"):
        arms[arm] = []
        for i in range(5):
            arms[arm].append({
                "task_hash": str(shard * 5 + i),
                "level": 1, "correct": arm == "format_reminder", "completed": True,
                "tool_calls": 1, "tool_errors": 0, "duplicate_calls_blocked": 0,
                "premature_final_proxy": False, "capability_gap": False,
                "latency_ms": 1000.0, "stop_reason": "final",
            })
    return {"shard": shard, "gaia_revision": "682dd723ee1e1697e00360edccf2366dc8418dd9",
            "selection": "shadow25", "selected_identity_sha256": "fixed",
            "arms": arms}


def test_aggregate_paired_without_content_or_answers():
    out = aggregate([fake_shard(i) for i in range(5)])
    assert out["n"] == 25
    assert out["arms"]["original"]["correct"] == 0
    assert out["arms"]["format_reminder"]["correct"] == 25
    assert out["rescues"] == 25
    assert "cases" not in out


def test_aggregate_rejects_missing_and_mismatched_cases():
    with pytest.raises(ValueError, match="five unique"):
        aggregate([fake_shard(0)])
    shards = [fake_shard(i) for i in range(5)]
    shards[1]["arms"]["format_reminder"][0]["task_hash"] = "other"
    with pytest.raises(ValueError, match="mismatched"):
        aggregate(shards)
