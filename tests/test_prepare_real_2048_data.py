"""No GPU or private datasets needed: generic whole-task 2048 gate tests."""
from copy import deepcopy

import pytest

from scripts.prepare_real_2048_data import select_whole_task_rows


class StubTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids":list(text)}


def source_row(tid,source):
    return {
        "task_id":tid,"source":source,"verified":True,
        "training_action_indices":[2,4,6,8],
        "messages":[{"role":"system","content":"synthetic tool trace"}],
    }


def actions(tid, *, final_size=700):
    a=[]
    for n in range(4):
        a.append({
            "prompt":"p"*(500 if n!=3 else final_size),
            "completion":"c"*80,
            "metadata":{
                "task_id":tid,
                "action_type":"tool_call" if n!=3 else "final",
                "assistant_message_index":[2,4,6,8][n],
            },
        })
    return a


def _fixtures():
    rows=[
        source_row("hot-1","hotpotqa-distractor-train"),
        source_row("hot-2","hotpotqa-distractor-train"),
        source_row("wiki-1","2wikimultihopqa-train"),
        source_row("hot-long","hotpotqa-distractor-train"),
    ]
    acts=[a for r in rows for a in actions(
        r["task_id"],final_size=2300 if r["task_id"]=="hot-long" else 700)]
    return rows,acts


def test_complete_tool_and_final_supervision_preserved_when_filtering():
    rows,acts=_fixtures()
    selected,report=select_whole_task_rows(
        rows=rows,examples=acts,tokenizer=StubTokenizer(),
        min_tasks=3,min_second_source=1)
    assert [r["task_id"] for r in selected]==["hot-1","hot-2","wiki-1"]
    assert report["original_unique_tasks"]==4
    assert report["retained_unique_tasks"]==3
    assert report["retained_native_actions"]==12
    assert report["retained_final_answers"]==3
    assert report["rejected_tasks"]=={"whole_task_exceeds_context":1}
    assert report["source_counts"]=={
        "2wikimultihopqa-train":1,"hotpotqa-distractor-train":2,
    }
    assert report["maximum_kept_action_tokens"]==780
    assert report["require_full_prompt"] is True
    assert report["new_model_training_ran"] is False
    assert report["gaia_evaluation_ran"] is False


def test_never_include_a_partial_trace_even_when_tool_calls_fit():
    rows,acts=_fixtures()
    selected,_=select_whole_task_rows(
        rows=rows,examples=acts,tokenizer=StubTokenizer(),
        min_tasks=3,min_second_source=1)
    assert "hot-long" not in [row["task_id"] for row in selected]
    with pytest.raises(ValueError,match="insufficient untruncated"):
        select_whole_task_rows(
            rows=rows,examples=acts,tokenizer=StubTokenizer(),
            min_tasks=4,min_second_source=1)


def test_unknown_source_missing_final_and_duplicate_ids_fail_closed():
    rows,acts=_fixtures()
    broken=deepcopy(rows)
    broken[2]["source"]="gaia-validation"
    with pytest.raises(ValueError,match="source-attributed"):
        select_whole_task_rows(
            rows=broken,examples=acts,tokenizer=StubTokenizer(),
            min_tasks=3,min_second_source=1)
    broken=deepcopy(rows)
    broken[1]["task_id"]="hot-1"
    with pytest.raises(ValueError,match="duplicate"):
        select_whole_task_rows(
            rows=broken,examples=acts,tokenizer=StubTokenizer(),
            min_tasks=3,min_second_source=1)
    broken=deepcopy(acts)
    broken[3]["metadata"]["action_type"]="tool_call"
    with pytest.raises(ValueError,match="final answer"):
        select_whole_task_rows(
            rows=rows,examples=broken,tokenizer=StubTokenizer(),
            min_tasks=3,min_second_source=1)
    with pytest.raises(ValueError,match="incomplete or foreign"):
        select_whole_task_rows(
            rows=rows,examples=acts[4:],tokenizer=StubTokenizer(),
            min_tasks=3,min_second_source=1)


def test_limit_is_exactly_2048_no_fitting_via_truncation():
    rows,acts=_fixtures()
    row=source_row("wiki-2048","2wikimultihopqa-train")
    ex=actions("wiki-2048",final_size=1968) # 1968+80=2048 exactly
    selected,report=select_whole_task_rows(
        rows=[row],examples=ex,tokenizer=StubTokenizer(),
        max_length=2048,min_tasks=1,min_second_source=1)
    assert len(selected)==1 and report["maximum_kept_action_tokens"]==2048
    ex[-1]["prompt"]+="x"
    with pytest.raises(ValueError,match="insufficient untruncated"):
        select_whole_task_rows(
            rows=[row],examples=ex,tokenizer=StubTokenizer(),
            max_length=2048,min_tasks=1,min_second_source=1)
