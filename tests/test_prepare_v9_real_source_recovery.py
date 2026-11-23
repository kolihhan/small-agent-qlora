"""Synthetic tests: v9 changes context, never supervised labels or target answers."""
from scripts.prepare_v9_real_source_recovery import augment


def _row(*, visible: bool):
    filename="memo-314159.txt"
    return {
        "task_id":"synthetic-test-row",
        "source":"2wikimultihopqa-train",
        "verified":True,
        "verification":{"completed":True,"final_answer_correct":True,
                        "zero_duplicate_blocks":True,"required_tools_satisfied":True,
                        "provenance_complete":True,"zero_tool_errors":True},
        "provenance":{"license":"CC0-1.0","generator":"test","generator_version":"v1",
                      "oracle_type":"program","oracle_version":"1"},
        "metrics":{"tool_calls":1,"tool_errors":0,"tool_successes":1},
        "messages":[
            {"role":"system","content":"Follow supplied sources"},
            {"role":"user","content":"Please read memo-314159.txt and answer." if visible else "Find the answer."},
            {"role":"assistant","content":"","tool_calls":[
                {"id":"read1","type":"function","function":{
                    "name":"read","arguments":{"source":filename}}}]},
            {"role":"tool","tool_call_id":"read1","name":"read",
             "content":"The fictional value is blue."},
            {"role":"assistant","content":"blue"},
        ],
        "training_action_indices":[2,4],
        "expected_answer":"blue",
        "eval_task":{"question":"Find the answer.","expected_answer":"blue",
                     "files":{filename:"The fictional value is blue."},
                     "required_tools":["read"]},
        "tools":[],
    }


def test_executed_native_failure_preserves_supervised_actions():
    old=_row(visible=True)
    result=augment(old)
    assert result is not None
    new, group=result
    assert group=="2wikimultihopqa-train"
    assert new["training_action_indices"]==[4,6]
    assert len(new["messages"])==len(old["messages"])+2
    assert new["messages"][3]["content"].startswith("ToolError[NOT_FOUND]:")
    assert "File not found:" in new["messages"][3]["content"]
    assert [new["messages"][i] for i in new["training_action_indices"]]==[
        old["messages"][i] for i in old["training_action_indices"]]
    assert new["counterfactual_tool_failure"]["not_observed_agent_recovery"]
    assert new["verification"]["expected_tool_error_codes"]==["NOT_FOUND"]
    assert new["metrics"]["tool_errors"]==1
    assert new["expected_answer"]==old["expected_answer"]


def test_refuse_oracle_hidden_filename():
    assert augment(_row(visible=False)) is None


def test_do_not_augment_generative_synthetic_training():
    row=_row(visible=True)
    row["source"]="synthetic-small-agent-policy-v3/train"
    assert augment(row) is None
