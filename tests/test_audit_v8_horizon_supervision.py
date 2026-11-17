"""CPU tests of verified selected decision horizon accounting, not agent score."""
from hashlib import sha256
import json
import pytest
from scripts import audit_v8_horizon_supervision as a


def _row(task_id="x",source="synthetic-small-agent-policy-v3/train"):
    return {
      "task_id":task_id,"verified":True,"source":source,
      "messages":[
        {"role":"system","content":"do work"},
        {"role":"user","content":"question"},
        {"role":"assistant","tool_calls":[{"function":{"name":"read","arguments":{}}}]},
        {"role":"tool","content":"observation 1"},
        {"role":"assistant","tool_calls":[{"function":{"name":"read","arguments":{}}}]},
        {"role":"tool","content":"observation 2"},
        {"role":"assistant","tool_calls":[{"function":{"name":"read","arguments":{}}}]},
        {"role":"tool","content":"observation 3"},
        {"role":"assistant","tool_calls":[{"function":{"name":"read","arguments":{}}}]},
        {"role":"tool","content":"observation 4"},
        {"role":"assistant","tool_calls":[{"function":{"name":"read","arguments":{}}}]},
        {"role":"tool","content":"observation 5"},
        {"role":"assistant","content":"answer"}],
      "training_action_indices":[2,4,6,8,10,12]
    }


def test_selected_horizon_is_prior_tool_turns_not_future_tool_lookahead():
    states=a.selected_action_states(_row())
    assert states==[("tool",0),("tool",1),("tool",2),("tool",3),("tool",4),("final",5)]
    report=a.summarize([_row()])
    assert report["actions"]==6 and report["late_tool"]==1 and report["late_final"]==1
    assert report["late_continue_fraction"]==0.5
    assert report["max_prefix_tool_turns"]==5
    assert report["selected_actions_at_prefix_6_or_more"]==0


def test_multi_tool_single_assistant_turn_counts_as_one_decision_prefix():
    row=_row()
    row["messages"][2]["tool_calls"].append({"function":{"name":"search","arguments":{}}})
    assert a.selected_action_states(row)[1]==("tool",1)


def test_invalid_supervision_is_rejected():
    row=_row()
    row["training_action_indices"]=[2,3]
    with pytest.raises(ValueError,match="not assistant"):
        a.selected_action_states(row)
    row["training_action_indices"]=[2,2]
    with pytest.raises(ValueError,match="indices"):
        a.selected_action_states(row)


def test_pinned_source_bytes_and_gaia_train_exclusion(tmp_path):
    rows=[_row()]
    source=tmp_path/"source.jsonl"
    source.write_text(json.dumps(rows[0])+"\n")
    digest=sha256(source.read_bytes()).hexdigest()
    assert len(a.load_verified(source,digest))==1
    with pytest.raises(ValueError,match="SHA"):
        a.load_verified(source,"0"*64)
    rows[0]["source"]="GAIA-2023"
    source.write_text(json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError,match="GAIA"):
        a.load_verified(source,sha256(source.read_bytes()).hexdigest())


def test_both_cohorts_source_counts_preregistered():
    assert a.EXPECTED["train"]=={
       "parents":304,"actions":798,"late_tool":11,"late_final":68,
       "max_prefix_tool_turns":5}
    assert a.EXPECTED["holdout"]=={
       "parents":35,"actions":176,"late_tool":12,"late_final":24,
       "max_prefix_tool_turns":5}
    assert len(a.TRAIN_SHA)==len(a.HOLDOUT_SHA)==64
