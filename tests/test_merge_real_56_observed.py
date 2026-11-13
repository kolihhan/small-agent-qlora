"""Contracts for non-GAIA observed 56-task QLoRA mix."""
import pytest
from scripts.merge_real_56_observed import combine_strict_rows


def make_row(i, source, n_calls=3):
    msg=[{"role":"system","content":"test"},{"role":"user","content":f"Question: independent task {i}"}]
    for j in range(n_calls):
        msg.extend((
            {"role":"assistant","content":"","tool_calls":[{"id":f"{i}-{j}","function":{"name":"read","arguments":{"source":f"article_{j}.txt"}}}]},
            {"role":"tool","name":"read","tool_call_id":f"{i}-{j}","content":f"observed {j}"}
        ))
    msg.append({"role":"assistant","content":"verified answer"})
    return {
        "task_id":f"task-{i}","source":source,"verified":True,
        "messages":msg,"training_action_indices":list(range(2,len(msg),2)),
        "verification":{key:True for key in (
            "completed","final_answer_correct","zero_tool_errors",
            "zero_duplicate_blocks","required_tools_satisfied","provenance_complete"
        )}
    }


def rows():
    return [
        [make_row(i,"hotpotqa-distractor-train" if i<36 else "2wikimultihopqa-train",
                  n_calls=4 if i<4 else 3)
         for i in range(40)],
        [make_row(i,"2wikimultihopqa-train") for i in range(40,44)],
        [make_row(i,"2wikimultihopqa-train") for i in range(44,48)],
        [make_row(i,"2wikimultihopqa-train") for i in range(48,50)],
        [make_row(i,"2wikimultihopqa-train") for i in range(50,56)],
    ]


def test_merge_source_balanced_including_full_action_indices():
    combined,report=combine_strict_rows(rows())
    assert len(combined)==56
    assert report["input_assistant_actions"]==228
    assert report["source_task_counts"]=={
        "hotpotqa-distractor-train":36,"2wikimultihopqa-train":20,
    }


def test_reject_repeat_parent_and_question():
    source=rows()
    source[1][0]["task_id"]=source[0][0]["task_id"]
    with pytest.raises(ValueError,match="duplicate"):
        combine_strict_rows(source)
    source=rows()
    source[1][0]["messages"][1]["content"]=source[0][0]["messages"][1]["content"]
    with pytest.raises(ValueError,match="duplicate question"):
        combine_strict_rows(source)


def test_fail_closed_wrong_answer_and_missing_final():
    source=rows()
    source[2][0]["verification"]["final_answer_correct"]=False
    with pytest.raises(ValueError,match="unclean"):
        combine_strict_rows(source)
    source=rows()
    source[2][0]["messages"][-1]["tool_calls"]=[{"id":"bad"}]
    with pytest.raises(ValueError,match="final"):
        combine_strict_rows(source)
