from types import SimpleNamespace

import pytest

import scripts.eval_qwen9b_musique_other8_readpath as experiment
from gaia_small_agent.agent.types import TraceEvent


def test_read_paths_show_observed_evidence_and_no_ground_truth_hints():
    task={"files":{"passages/index.txt":"passages/p0.txt | A\npassages/p1.txt | B",
                   "passages/p0.txt":"The answer is Geneva.", "passages/p1.txt":"A distraction."},
          "expected_answer":"Geneva"}
    trace=[
        TraceEvent("tool_call",1,{"name":"read","arguments":{"source":"passages/index.txt"}}),
        TraceEvent("tool_call",2,{"name":"read","arguments":{"source":"passages/p0.txt"}}),
        TraceEvent("final",3,{"answer":"incorrect"}),
    ]
    s=experiment._path_summary(SimpleNamespace(trace=trace),task)
    assert s["first_read_path"]=="passages/index.txt"
    assert s["unique_passage_count"]==1
    assert s["reference_literal_in_read_passages"] is True
    assert s["read_titles"]==["[unrecognized path]","A"]
    assert s["tool_names"]==["read","read"]


def test_next_eight_cannot_reuse_prior_cases(monkeypatch):
    def fake_freeze(rows,report):
        return [x for hop in (3,4) for x in rows if x["hop_count"]==hop][:8]
    monkeypatch.setattr(experiment,"freeze_previous",fake_freeze)
    rows=[{"hop_count":hop,"task_id":f"{hop}-task-{i}"} for hop in (3,4) for i in range(12)]
    selected=experiment.freeze_other(rows,{})
    old={x["task_id"] for x in fake_freeze(rows,{})}
    assert len(selected)==8
    assert not old.intersection(x["task_id"] for x in selected)
    assert {3:4,4:4}=={hop:sum(x["hop_count"]==hop for x in selected) for hop in (3,4)}


def test_fail_closed_aggregate_and_true_autonomous_outcome():
    shards=[]
    for i in range(4):
        cases=[{"task_id":f"unseen-{2*i+j}","hop_count":3 if i<2 else 4,
                "correct":(i,j)==(0,0),"completed":True,"tool_calls":3,
                "tool_errors":0,"index_read":True,"unique_passage_count":2,
                "reference_literal_in_read_passages":False,"stop_reason":"final",
                "answer":"real model answer","expected_answer":"expected",
                "read_paths":["passages/index.txt"]} for j in range(2)]
        shards.append({"shard":i,"protocol":experiment.PROTOCOL,
                       "model_digest":experiment.EXPECTED_MODEL_DIGEST,
                       "physical_source_manifest":experiment.EXPECTED_MANIFEST,
                       "task_selection_hash":"cohort","cases":cases})
    r=experiment.aggregate(shards)
    assert (r["total"],r["correct"],r["completed"],r["read_index"])==(8,1,8,8)
    assert r["total_tools"]==24
    assert r["promote"] is False and r["model_improvement_claim"] is False
    with pytest.raises(ValueError,match="incomplete"):
        experiment.aggregate(shards[:3])
