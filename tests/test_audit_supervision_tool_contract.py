from __future__ import annotations

import pytest

from scripts.audit_supervision_tool_contract import audit_rows, _target_actions

def _row(task_id, action, *, observations=None, chosen=2):
    messages=[{"role":"system","content":"agent"},
              {"role":"user","content":"task"},
              action]
    if observations:
        messages += observations
    return {
        "task_id":task_id,"messages":messages,
        "training_action_indices":[chosen],
        "verification":{"zero_tool_errors":True},
        "metrics":{"tool_errors":0},
    }

def _call(name, args):
    return {"role":"assistant","content":"","tool_calls":[
        {"id":"call-1","type":"function","function":{"name":name,"arguments":args}}
    ]}

def test_native_read_target_is_valid_and_final_is_not_tool_action():
    rows=[_row("native-read",_call("read",{"source":"index.txt"})),
          _row("native-final",{"role":"assistant","content":"correct answer"})]
    report=audit_rows(rows,source_label="native")
    assert report["valid_native_tool_calls"]==1
    assert report["selected_final_actions"]==1
    assert report["decision"]=="PASS_NATIVE_SCHEMA_ONLY_NOT_TASK_CORRECTNESS"

def test_nextsearch_tool_surface_is_incompatible_and_metadata_error_contradicted():
    row=_row("next",_call("search",{"objective":"Find evidence","search_queries":["a","b","c"]}),
        observations=[{"role":"tool","content":"tool error: HTTPStatusError: 429 Too Many Requests"}])
    row["source_parent_task_id"]="parent"
    other=_row("next2",_call("fetch",{"objective":"Read URL","urls":["https://example.org"]}))
    report=audit_rows([row,other],source_label="nextsearch")
    assert report["target_tool_calls"]==2
    assert report["valid_native_tool_calls"]==0
    assert report["invalid_native_arguments"]==1
    assert report["unknown_native_tool_calls"]==1
    assert report["observed_tool_error_windows"]==1
    assert report["metadata_zero_error_contradictions"]==1
    assert report["decision"]=="FAIL_NATIVE_TOOL_CONTRACT"

def test_selected_target_may_be_before_final_and_validation_is_fail_closed():
    row=_row("a",_call("read",{"source":"notes.txt"}),
        observations=[{"role":"tool","content":"read observation"},{"role":"assistant","content":"answer"}])
    assert len(_target_actions(row))==1
    row["training_action_indices"]=[3]
    with pytest.raises(ValueError,match="assistant"):
        _target_actions(row)
    row["training_action_indices"]=[999]
    with pytest.raises(ValueError,match="index"):
        _target_actions(row)
