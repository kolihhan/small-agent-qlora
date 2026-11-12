"""Frozen v5 real/synthetic mixture: no repeated task or extra GPU."""
from copy import deepcopy
import pytest

from scripts.prepare_qlora_v5_real_mix import merge_rows


def _synth():
    return [
        {"task_id":f"synth-{i:02d}",
         "source":"synthetic-small-agent-policy-v3/train",
         "verified":True,
         "capability":("tool_selection" if i%2==0 else "multi_step"),
         "training_action_indices":[2,4]}
        for i in range(40)
    ]


def _real():
    return [
        {"task_id":"hotpot-independent","source":"hotpotqa-distractor-train",
         "verified":True,"training_action_indices":[2,4]},
        {"task_id":"2wiki-independent","source":"2wikimultihopqa-train",
         "verified":True,"training_action_indices":[2,4]},
    ]


def test_exact_action_budget_without_repeating_real_tasks():
    syn=_synth()
    real=_real()
    mixed,report=merge_rows(syn,real,target_actions=80,max_real_fraction=.05)
    assert len(mixed)==40 and report["unique_mixed_tasks"]==40
    assert report["mixed_actions"]==80
    assert report["removed_synthetic_tasks"]==2
    assert report["removed_synthetic_actions"]==4
    assert report["retained_synthetic_tasks"]==38
    assert report["retained_synthetic_actions"]==76
    assert report["real_action_fraction"]==.05
    assert report["real_task_oversampling_factor"]==1
    assert sum(x["source"]=="hotpotqa-distractor-train" for x in mixed)==1
    assert sum(x["source"]=="2wikimultihopqa-train" for x in mixed)==1
    assert report["training_ran"] is False and report["gaia_evaluated"] is False
    assert report["evaluation_ready"] is False
    assert mixed==merge_rows(syn,real,target_actions=80,max_real_fraction=.05)[0]


def test_stratified_removal_across_synthetic_capabilities():
    _,report=merge_rows(_synth(),_real(),target_actions=80,max_real_fraction=.05)
    assert report["omitted_synthetic_capabilities"]=={
        "multi_step":1,"tool_selection":1}
    assert report["synthetic_capabilities_after"]=={
        "multi_step":19,"tool_selection":19}


def test_no_source_leakage_repeat_or_broken_provenance():
    syn=_synth()
    real=_real()
    real[0]["task_id"]=syn[0]["task_id"]
    with pytest.raises(ValueError,match="duplicate"):
        merge_rows(syn,real,target_actions=80)
    real=_real()
    real[0]["source"]="gaia-validation"
    with pytest.raises(ValueError,match="real source"):
        merge_rows(syn,real,target_actions=80)
    real=_real()
    real[0]["verified"]=False
    with pytest.raises(ValueError,match="real source"):
        merge_rows(syn,real,target_actions=80)
    syn=deepcopy(_synth())
    syn[0]["source"]="synthetic-small-agent-policy-v3/dev"
    with pytest.raises(ValueError,match="synthetic policy"):
        merge_rows(syn,_real(),target_actions=80)
    syn=_synth()
    syn[0]["training_action_indices"]=[2,2]
    with pytest.raises(ValueError,match="native"):
        merge_rows(syn,_real(),target_actions=80)


def test_real_percentage_never_exceeds_frozen_gate():
    with pytest.raises(ValueError,match="dominate"):
        merge_rows(_synth(),_real(),target_actions=80,max_real_fraction=.04)
    with pytest.raises(ValueError,match="budget"):
        merge_rows(_synth(),_real(),target_actions=100)


def test_full_task_budget_not_met_with_insufficient_two_action_synthetic():
    syn=[{"task_id":"s1","source":"synthetic-small-agent-policy-v3/train",
          "verified":True,"capability":"no_tool_stop","training_action_indices":[2]},
         {"task_id":"s2","source":"synthetic-small-agent-policy-v3/train",
          "verified":True,"capability":"multi_step","training_action_indices":[2,4,6]},
         {"task_id":"s3","source":"synthetic-small-agent-policy-v3/train",
          "verified":True,"capability":"argument_grounding","training_action_indices":[2,4]}]
    # Source target=6, real data would exceed 5% even with generous limit;
    # use 10% test cap and verify percentage gate is reached first.
    with pytest.raises(ValueError,match="dominate"):
        merge_rows(syn,_real(),target_actions=6,max_real_fraction=.1)
