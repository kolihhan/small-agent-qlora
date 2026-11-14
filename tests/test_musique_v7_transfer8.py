import pytest

from scripts.musique_v7_transfer8 import aggregate


def part(arm, shard, *, good=False):
    return {
        "arm": arm, "shard": shard, "source_sha": "frozen",
        "cohort_sha": "same",
        "rows": [
            {"id_hash": f"task-{shard}-3", "hop_count": 3, "correct": good,
             "completed": True, "tool_calls": 3, "tool_errors": 0,
             "unique_passage_reads": 2, "stop": "final"},
            {"id_hash": f"task-{shard}-4", "hop_count": 4, "correct": False,
             "completed": False, "tool_calls": 6, "tool_errors": 0,
             "unique_passage_reads": 3, "stop": "max_steps"},
        ],
    }


def test_pairing_same_eight_tasks_and_report_success_only_when_observed():
    parts = [part("base", i) for i in range(4)] + [part("v7", i, good=True) for i in range(4)]
    report = aggregate(parts)
    assert report["n"] == 8
    assert report["arms"]["base"]["correct"] == 0
    assert report["arms"]["v7"]["correct"] == 4
    assert report["rescues"] == 4
    assert report["regressions"] == 0


def test_duplicate_or_mismatched_task_ids_are_rejected():
    parts = [part("base", i) for i in range(4)] + [part("v7", i) for i in range(4)]
    parts[-1]["rows"][0]["id_hash"] = "different-task"
    with pytest.raises(ValueError, match="identity"):
        aggregate(parts)
