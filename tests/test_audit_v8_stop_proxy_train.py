import json
from scripts.audit_v8_stop_proxy_train import summarize
import pytest


def test_continuation_after_correct_answer_visible_disproves_naive_stop(tmp_path):
    # At first read result, the final answer string is visible, but the
    # verified trace continues to read a second source before finalizing.
    m=[
        {"role":"user","content":"Find both sources"},
        {"role":"assistant","tool_calls":[{"name":"read"}]},
        {"role":"tool","content":"Alpha"},
        {"role":"assistant","tool_calls":[{"name":"read"}]},
        {"role":"tool","content":"Alpha confirmed by independent evidence"},
        {"role":"assistant","content":"Alpha"},
    ]
    row={"task_id":"safe1","source":"synthetic-small-agent-policy-v3/train",
         "verified":True,"messages":m,"training_action_indices":[1,3,5]}
    p=tmp_path/"train.jsonl"
    p.write_text(json.dumps(row)+"\n")
    r=summarize(p,expected_n=1)
    assert r["overall"]["answer_already_visible_while_agent_continued"]==1
    assert r["overall"]["extra_tool_actions_after_answer_visible"]==1
    assert r["overall"]["answer_verbatim_at_final_prefix"]==1
    assert "Alpha" not in json.dumps(r)
    assert r["promotion_allowed"] is False


def test_fails_on_incorrect_trace(tmp_path):
    p=tmp_path/"bad.jsonl"
    p.write_text(json.dumps({"task_id":"x","verified":False})+"\n")
    with pytest.raises(ValueError,match="Frozen verified"):
        summarize(p,expected_n=1)
