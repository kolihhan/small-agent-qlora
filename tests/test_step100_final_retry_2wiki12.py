"""Synthetic assertions for the new 2Wiki source-family evaluation, no real gold/task IDs."""
import json

import pytest

from scripts.eval_step100_final_retry_2wiki12 import (
    ARMS, N, PROTOCOL, SOURCE_SHA, SOURCE_REV, _valid, materialize, score,
)


def test_2wiki_structural_validation_and_passage_workspace():
    row={
        "_id":"fictional-id","question":"Where was Actor X born?",
        "answer":"Fictional City",
        "context":[[f"Fictional Article {i}",["A sourced sentence about Actor X from the provided fake file."]]
                   for i in range(10)],
    }
    assert _valid(row)
    task=materialize(row)
    assert len(task["files"])==11 and "passages/index.txt" in task["files"]
    assert "Fictional City" not in task["question"]
    assert task["files"]["passages/p0.txt"].startswith("Title: Fictional Article 0")
    assert "question_decomposition" not in json.dumps(task["files"])


def _cases(shard,arm,correct,invalid,trigger):
    return [{
        "id_sha256":f"not_a_real_id_{shard*4+k}",
        "correct":correct[shard*4+k],
        "completed":not invalid[shard*4+k],
        "stop_reason":"invalid_final" if invalid[shard*4+k] else "final",
        "steps":12, "tool_calls":6, "tool_errors":0,
        "duplicate_calls_blocked":0,
        "final_text_retries":1 if arm=="retry" and trigger[shard*4+k] else 0,
        "elapsed_seconds":11.0 if arm=="control" else 11.5
    } for k in range(4)]


def _populate(tmp, *, control=None, retry=None, inv0=None, inv1=None, triggers=None):
    control=control or [False]*N
    retry=retry or [False]*N
    inv0=inv0 or [False]*N
    inv1=inv1 or [False]*N
    triggers=triggers or [False]*N
    for shard in range(3):
        for arm in ARMS:
            item={
                "protocol":PROTOCOL,"arm":arm,"shard":shard,"n":4,
                "source_sha":SOURCE_SHA,"source_rev":SOURCE_REV,
                "selection_manifest":"fictional-bounded-selection",
                "base_adapter":"physically_audited_qwen3.5_9b_v8_step100",
                "only_variable":"final_text_retry_opt_in",
                "max_steps":12,"max_new_tokens":512,
                "tool_names":["search","read","inspect","python"],
                "no_train_or_gold_in_decisions":True,
                "no_task_or_answer_content_exported":True,
                "cases":_cases(shard,arm,control if arm=="control" else retry,
                               inv0 if arm=="control" else inv1,triggers),
            }
            (tmp/f"shard-{shard}-{arm}.json").write_text(json.dumps(item))


def test_predeclared_gate_requires_real_correctness_and_cost(tmp_path):
    candidate=[False]*12
    candidate[1]=candidate[9]=True
    inv0=[False]*12
    inv0[1]=inv0[9]=True
    triggers=[False]*12
    triggers[1]=triggers[9]=True
    _populate(tmp_path,retry=candidate,inv0=inv0,triggers=triggers)
    report=score(tmp_path)
    assert report["control_correct"]==0 and report["retry_correct"]==2
    assert report["rescues"]==2 and report["regressions"]==0
    assert report["final_retry_events"]==2
    assert report["gate_pass"] is True


def test_no_improvement_cannot_pass_and_refuse_protocol_drift(tmp_path):
    _populate(tmp_path)
    assert score(tmp_path)["gate_pass"] is False
    target=tmp_path/"shard-1-control.json"
    payload=json.loads(target.read_text())
    payload["tool_names"] += ["extra_bad_tool"]
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protocol drift"):
        score(tmp_path)


def test_refuse_control_intervention(tmp_path):
    _populate(tmp_path)
    target=tmp_path/"shard-0-control.json"
    payload=json.loads(target.read_text())
    payload["cases"][0]["final_text_retries"]=1
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Control used intervention"):
        score(tmp_path)
