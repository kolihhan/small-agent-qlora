import json

from scripts.prepare_nextsearch_bounded_sft import export,prepare
from gaia_small_agent.training.qlora import load_verified_rows


def source_task(answer_visible=True):
    target=[]
    for i in range(8):
        name="search" if i%2==0 else "fetch"
        ident=f"call-{i}"
        target.extend([
            {"role":"assistant","content":"","reasoning_content":"secret hidden text",
             "tool_calls":[{"id":ident,"type":"function","function":{
                 "name":name,"arguments":json.dumps({"query":"original source","index":i})
             }}]},
            {"role":"tool","tool_call_id":ident,"content":
             ("Observed original web evidence Blue" if i==7 and answer_visible
              else f"Original real web observation {i}.")}
        ])
    target.append({"role":"assistant","content":"Blue",
                   "reasoning_content":"private teacher thinking"})
    return {
        "id":"taskcraft/unique-source-42","source":"taskcraft",
        "license":"apache-2.0","gold":{"answer":"Blue"},
        "messages":[
            {"role":"system","content":"research helper"},
            {"role":"user","content":"What is the answer to an independent question?"}
        ],
        "tools":[
            {"name":"search","description":"Search public web",
             "parameters":{"type":"object","properties":{"query":{"type":"string"}}}},
            {"name":"fetch","description":"Fetch public source",
             "parameters":{"type":"object","properties":{"query":{"type":"string"}}}}
        ],
        "target":target
    }


def test_all_labels_use_original_actions_and_actual_bounded_web_observations(tmp_path):
    row,reason=export(source_task())
    assert reason=="source_verified_candidate",reason
    assert row["verified"]
    assert row["metrics"]["tool_calls"]==8
    assert row["expected_answer"]=="Blue"
    assert row["tools"][0]["function"]["name"]=="search"
    assert row["tools"][1]["function"]["name"]=="fetch"
    assert all(x["role"] in ("system","user","assistant","tool")
               for x in row["messages"])
    assert "secret hidden text" not in json.dumps(row)
    assert "private teacher thinking" not in json.dumps(row)
    assert "Blue" in row["messages"][-2]["content"]
    assert len(row["training_action_indices"])==5
    assert all(row["messages"][i]["role"]=="assistant"
               for i in row["training_action_indices"])
    assert not row["source_success_independently_replayed"]
    path=tmp_path/"verified.jsonl"
    path.write_text(json.dumps(row)+"\n")
    assert len(load_verified_rows(path))==1


def test_missing_answer_evidence_never_becomes_source_teacher():
    assert export(source_task(answer_visible=False))[1] == "final_reference_not_in_bounded_original_observations"


def test_duplicate_parent_is_deduped_from_training_corpus(tmp_path):
    t=source_task()
    out=tmp_path/"sft.jsonl"
    summary=prepare([t,t],out)
    assert summary["task_disjoint_unique_parents"]==1
    assert summary["rejections"]["duplicate_qualified_source_task"]==1
    assert len(out.read_text().splitlines())==1


def test_answer_in_question_and_reference_change_are_rejected():
    x=source_task()
    x["messages"][1]["content"]+=" Blue"
    assert export(x)[1]=="reference_already_in_question"
    y=source_task()
    y["target"][-1]["content"]="Red"
    assert export(y)[1]=="not_original_executed_long_gold_exact"
