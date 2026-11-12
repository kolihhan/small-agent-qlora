"""Zero-GPU whole-task filtering invariants; only synthetic observed trace shapes."""
import pytest

from scripts.filter_real_2048_tasks import filter_full_tasks


class Tokenizer:
    def __call__(self,text,*,add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids":list(text)}


def _task(tid,source):
    return {"task_id":tid,"source":source,"verified":True,
            "training_action_indices":[2,4,6,8]}


def _examples(tid,*,long_final=False,long_tool=False):
    rows=[]
    for i in range(4):
        rows.append({
            "prompt":"x"*(1900 if (long_final and i==3) or (long_tool and i==1) else 250),
            "completion":"y"*100,
            "metadata":{"task_id":tid,"action_type":"final" if i==3 else "tool_call"},
        })
    return rows


def _data():
    rows=[
        _task("hot-short","hotpotqa-distractor-train"),
        _task("wiki-short","2wikimultihopqa-train"),
        _task("hot-long","hotpotqa-distractor-train"),
    ]
    actions=(_examples("hot-short")+_examples("wiki-short")
             +_examples("hot-long",long_final=True))
    return rows,actions


def test_one_overbudget_final_rejects_whole_task_not_just_answer():
    rows,ex=_data()
    kept,report=filter_full_tasks(rows,ex,Tokenizer(),budget=1024)
    assert {x["task_id"] for x in kept}=={"hot-short","wiki-short"}
    assert report["input_native_actions"]==12
    assert report["retained_native_actions"]==8
    assert report["rejected_complete_tasks"]==1
    assert report["retained_complete_tasks"]==2
    assert report["retained_source_counts"]=={
        "2wikimultihopqa-train":1, "hotpotqa-distractor-train":1
    }
    assert report["truncated_prompt_or_completion_count"]==0
    assert all("question" not in obj for obj in report["rejection_diagnostics_no_questions"])


def test_extra_long_tool_rejects_entire_task_even_if_final_is_short():
    rows,ex=_data()
    ex[-4:]=_examples("hot-long",long_tool=True)
    kept,report=filter_full_tasks(rows,ex,Tokenizer(),budget=1024)
    assert report["retained_complete_tasks"]==2
    assert "hot-long" not in {x["task_id"] for x in kept}


def test_source_diversity_loss_is_rejected():
    rows,ex=_data()
    # Reject the *only* Wiki source even though the Hotpot task still fits.
    ex[4:8]=_examples("wiki-short",long_final=True)
    with pytest.raises(ValueError,match="multi-source"):
        filter_full_tasks(rows,ex,Tokenizer(),budget=1024)


def test_fails_closed_when_missing_final_or_dropped_tool():
    rows,ex=_data()
    ex[3]["metadata"]["action_type"]="tool_call"
    with pytest.raises(ValueError,match="exactly one final"):
        filter_full_tasks(rows,ex,Tokenizer(),budget=1024)
    rows,ex=_data()
    ex.pop(0)
    with pytest.raises(ValueError,match="curated indices"):
        filter_full_tasks(rows,ex,Tokenizer(),budget=1024)


def test_incomplete_source_or_duplicate_parent_id_is_rejected():
    rows,ex=_data()
    rows[2]["task_id"]=rows[0]["task_id"]
    with pytest.raises(ValueError,match="duplicate"):
        filter_full_tasks(rows,ex,Tokenizer(),budget=1024)
    rows,ex=_data()
    rows[2]["source"]="gaia-validation"
    with pytest.raises(ValueError,match="not permitted"):
        filter_full_tasks(rows,ex,Tokenizer(),budget=1024)


def test_all_actions_overbudget_and_invalid_context_fail():
    rows,ex=_data()
    with pytest.raises(ValueError,match="budget"):
        filter_full_tasks(rows,ex,Tokenizer(),budget=32)
    big=[{**obj,"prompt":"q"*3000} for obj in ex]
    with pytest.raises(ValueError,match="all real tasks rejected"):
        filter_full_tasks(rows,big,Tokenizer(),budget=2048)
