"""Synthetic tests for disjoint selection and 24-case paired prereg scoring."""
import hashlib
import json

import pytest
from scripts.eval_evidence_state_dev24b import (
    SEED, N, SOURCE_SHA256, PROTOCOL, select_rows, score,
)


def _devrows():
    rows=[]
    for hop in (3,4):
        for k in range(31):
            ident=f"{hop}hop123__fictional{k:02}"
            rows.append({
                "id":ident, "answerable":True, "question":f"Question {ident}?",
                "answer":f"secret{k}",
                "question_decomposition":[{"id":f"part{n}"} for n in range(hop)],
                "paragraphs":[{"idx":j, "title":f"Title{j}",
                  "paragraph_text":"Fictional passage to test structural source validation."} for j in range(20)],
            })
    return rows


def test_second_rank12_disjoint_from_original_first_rank12():
    rows=_devrows()
    second=select_rows(rows)
    assert len(second)==N
    assert [r["id"] for r in second]==[r["id"] for r in select_rows(reversed(rows))]
    for hop in (3,4):
        ranked=sorted(
            (hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest(),r['id'])
            for r in rows if r["id"][0]==str(hop)
        )
        original_first12={x[1] for x in ranked[:12]}
        second_ids={r["id"] for r in second if r["id"][0]==str(hop)}
        assert not original_first12.intersection(second_ids)
        assert second_ids=={x[1] for x in ranked[12:24]}


def _case(shard, local, arm, correct):
    ident=f"task-{shard}-{local}"
    return {
        "id_sha256":hashlib.sha256(ident.encode()).hexdigest(),
        "hop_count":3 if shard<2 else 4,
        "correct":correct, "completed":True,
        "stop_reason":"final", "tool_calls":4,
        "steps":5, "tool_errors":0, "duplicate_calls":0,
        "elapsed_s":1.1, "evidence_state_messages":0 if arm=="control" else 2,
    }


def _fixture(tmp_path, truth):
    for shard in range(4):
        for arm in ("control","state"):
            payload={
                "protocol":PROTOCOL,"arm":arm,"shard":shard,
                "n":6,"manifest_sha256":"fixed",
                "source_sha256":SOURCE_SHA256,
                "adapter":"verified_qwen3.5_9b_v8_step100",
                "max_steps":12,"max_new_tokens":512,
                "only_variable":"opt_in_source_anchored_evidence_state",
                "same_four_tools":True,
                "answer_label_used_in_decisions":False,
                "no_task_or_answers_exported":True,
                "cases":[_case(shard,i,arm,truth[arm][shard*6+i]) for i in range(6)],
            }
            (tmp_path/f"shard-{shard}-{arm}.json").write_text(json.dumps(payload))


def test_preregistered_gate_requires_cross_hop_rescues_low_regression_and_call_cost(tmp_path):
    control=[False]*24
    for i in (2,13):
        control[i]=True
    state=list(control)
    for i in (0,4,14,18):
        state[i]=True
    _fixture(tmp_path, {"control":control,"state":state})
    result=score(tmp_path)
    assert result["base_correct"]==2
    assert result["state_correct"]==6
    assert result["rescues"]==4 and result["regressions"]==0
    assert result["by_hop"]["3"]["state_correct"]==3
    assert result["by_hop"]["4"]["state_correct"]==3
    assert result["gate_pass"] is True
    assert result["control_index_messages"]==0
    assert result["state_index_messages"]==48


def test_missing_or_protocol_drift_rejected(tmp_path):
    _fixture(tmp_path,{"control":[False]*24,"state":[False]*24})
    f=tmp_path/"shard-1-state.json"
    obj=json.loads(f.read_text())
    obj["same_four_tools"]=False
    f.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="protocol drift"):
        score(tmp_path)
    f.unlink()
    with pytest.raises(FileNotFoundError):
        score(tmp_path)


def test_regressions_cancel_rescues_and_are_not_hidden(tmp_path):
    control=[False]*24
    control[0]=control[15]=True
    state=list(control)
    state[0]=False
    state[2]=state[16]=True
    _fixture(tmp_path,{"control":control,"state":state})
    result=score(tmp_path)
    assert result["rescues"]==2
    assert result["regressions"]==1
    assert result["state_correct"]-result["base_correct"]==1
    assert result["gate_pass"] is False
