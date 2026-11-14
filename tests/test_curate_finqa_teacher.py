import pytest

from scripts.curate_finqa_teacher import audit_verified, curate


def row(sample="parent::sample-1", *, answer="24.0", verified=True):
    return {
        "task_id": sample, "source": "finqa-train", "verified": verified,
        "expected_answer": "24.0",
        "required_tools": ["read", "python"],
        "verification": {
            "completed": True, "final_answer_correct": True,
            "zero_tool_errors": True, "zero_duplicate_blocks": True,
            "required_tools_satisfied": True, "provenance_complete": True,
        },
        "metrics": {"tool_calls": 2, "tool_errors": 0},
        "messages": [
            {"role": "system", "content": "Use tools"},
            {"role": "user", "content": "Find the change in revenue"},
            {"role": "assistant", "tool_calls": [{"id":"one","function":{
                "name": "read", "arguments": {"source": "report.txt"}
            }}]},
            {"role": "tool", "tool_call_id": "one", "name": "read", "content": "100 and 76"},
            {"role": "assistant", "tool_calls": [{"id":"two","function":{
                "name": "python", "arguments": {"code": "print(100-76)"}
            }}]},
            {"role": "tool", "tool_call_id": "two", "name": "python", "content": "24"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_accept_only_real_tool_and_numeric_final_and_dedup_trials():
    assert audit_verified(row()) == [2,4,6]
    r1=row()
    r2=row("parent::sample-2")
    out, report=curate([r1,r2])
    assert report["accepted_unique_parents"] == 1
    assert report["trainable_assistant_actions"] == 3
    assert out[0]["task_id"] == "parent"
    assert out[0]["training_action_indices"] == [2,4,6]


def test_reject_wrong_answer_and_non_source_read():
    with pytest.raises(ValueError, match="final_answer"):
        audit_verified(row(answer="25"))
    bad=row()
    bad["messages"][2]["tool_calls"][0]["function"]["arguments"]["source"] = "external.txt"
    with pytest.raises(ValueError, match="non_source_document_read"):
        audit_verified(bad)


def test_reject_dirty_recovery_for_automatic_sft():
    bad=row()
    bad["verification"]["zero_tool_errors"]=False
    with pytest.raises(ValueError,match="incomplete_verification"):
        audit_verified(bad)
    bad=row(verified=False)
    results, report=curate([bad])
    assert not results and report["accepted_unique_parents"] == 0


def test_no_duplicate_samples_silently_accepted():
    with pytest.raises(ValueError,match="duplicate_observed_sample"):
        curate([row(),row()])
