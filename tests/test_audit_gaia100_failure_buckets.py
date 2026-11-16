from __future__ import annotations

import json

import pytest

from scripts.audit_gaia100_failure_buckets import summarize,EXPECTED_REVISION,EXPECTED_TASK_IDS_SHA


def make_artifacts(tmp_path, *,repeat=False,invalid_correct=False):
    paths=[]
    for shard in range(20):
        one=[];two=[]
        for i in range(5):
            hashed=f"{shard:03d}-{i:03d}" if not repeat else "same-id"
            row={"task_hash":hashed,"level":1,"correct":False,
                 "completed":False,"stop_reason":"invalid_final",
                 "tool_calls":2,"tool_errors":0,"duplicates":1}
            one.append({**row})
            two.append({**row})
        if invalid_correct and shard==0:
            one[0]["correct"]=True
        path=tmp_path/f"shard-{shard}.json"
        path.write_text(json.dumps({
            "shard":shard,"source_revision":EXPECTED_REVISION,
            "selection":"gaia-evaluation100-seen-dev",
            "full_hash":EXPECTED_TASK_IDS_SHA,
            "arms":{"original":one,"format_reminder":two}
        }))
        paths.append(path)
    return paths


def test_all_100_paired_exactly_once_with_accounting(tmp_path):
    r=summarize(make_artifacts(tmp_path))
    assert r["n"]==100
    assert r["paired"]["both_wrong"]==100
    assert r["arms"]["original"]["not_completed"]==100


def test_reject_duplicate_source_ids(tmp_path):
    with pytest.raises(ValueError,match="reused"):
        summarize(make_artifacts(tmp_path,repeat=True))


def test_reject_impossible_correct_without_final(tmp_path):
    with pytest.raises(ValueError,match="Impossible"):
        summarize(make_artifacts(tmp_path,invalid_correct=True))
