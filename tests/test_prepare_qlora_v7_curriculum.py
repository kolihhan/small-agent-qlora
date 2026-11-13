"""v7 regression contracts: real multi-source whole-task SFT, no GAIA or replays."""
from copy import deepcopy
import pytest

from scripts.prepare_qlora_v7_curriculum import build_curriculum


def _row(idx, source, actions, capability=None):
    return {"task_id":f"{source}:{idx}", "source":source, "verified":True,
            "capability":capability or "unspecified",
            "training_action_indices":list(range(2,2+2*actions,2))}


def _data():
    synthetic=[]
    capacities = {
        "no_tool_stop": 1, "evidence_to_final": 2,
        "tool_selection": 2, "argument_grounding": 2,
        "multi_step": 2, "failure_recovery": 2,
        "strategy_switch": 2, "duplicate_avoidance": 2,
    }
    for cap,n in capacities.items():
        synthetic += [_row(f"{cap}-{i}","synthetic-small-agent-policy-v3/train",n,cap)
                      for i in range(40)]
    # every source family is represented by independently observed actions
    real=([_row(i,"2wikimultihopqa-train",4) for i in range(20)]
        + [_row(i,"hotpotqa-distractor-train",4) for i in range(35)])
    return synthetic, real


def test_deterministic_parent_disjoint_holdout_and_real_ratio():
    synth,real=_data()
    train,val,report=build_curriculum(synth,real)
    assert train==build_curriculum(synth,real)[0]
    tids=[r["task_id"] for r in train+val]
    assert len(tids)==len(set(tids))
    assert {r["source"] for r in val}=={
        "hotpotqa-distractor-train","2wikimultihopqa-train"}
    assert len(val)==11  # groupwise 20% of 35 Hotpot and 20 2Wiki
    assert report["source_train_real_tasks"]==44
    assert report["source_holdout_real_tasks"]==11
    assert 0.30 <= report["train_real_action_fraction"] <= 0.40
    assert report["synthetic_capabilities_covered"]==8
    assert report["no_tool_stop_tasks"]>0
    assert report["training_ran"] is False
    assert report["gaia_evaluated"] is False


def test_no_repeat_failure_and_no_gaia_or_dev_leakage():
    synth,real=_data()
    bad=deepcopy(real)
    bad[0]["task_id"]=synth[0]["task_id"]
    with pytest.raises(ValueError,match="duplicate"):
        build_curriculum(synth,bad)
    bad=deepcopy(real)
    bad[0]["source"]="gaia-evaluation"
    with pytest.raises(ValueError,match="source"):
        build_curriculum(synth,bad)
    bad=deepcopy(real)
    bad[0]["source"]="hotpotqa-distractor-validation"
    with pytest.raises(ValueError,match="source"):
        build_curriculum(synth,bad)
    bad=deepcopy(synth)
    bad[0]["training_action_indices"]=[2,2]
    with pytest.raises(ValueError,match="action"):
        build_curriculum(bad,real)


def test_source_shift_fail_closed_if_only_one_real_family():
    synth,real=_data()
    with pytest.raises(ValueError,match="source"):
        build_curriculum(synth,[r for r in real if r["source"].startswith("hotpotqa")])


def test_complete_real_task_not_sliced_when_budget_is_tight():
    synth,real=_data()
    train,val,summary=build_curriculum(synth,real,synthetic_to_real_ratio=1.0)
    parents={r["task_id"] for r in train}
    assert all(len(r["training_action_indices"])==4
               for r in train if r["source"]!="synthetic-small-agent-policy-v3/train")
    assert not parents.intersection({r["task_id"] for r in val})
    assert 0.45 <= summary["train_real_action_fraction"] <= 0.55
