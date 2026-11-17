"""CPU-only source selection, strict 4-task paired result fail-closed tests."""
import json
from hashlib import sha256
import pytest
from scripts import eval_appworld_qwen9b_dev4 as m

def ids():
    return [f"scene{i}_{j}" for i in range(15) for j in range(2)]

def test_source_frozen_before_any_model_and_no_lookahead():
    sample=ids()
    a=m.freeze_ids(sample)
    b=m.freeze_ids(list(reversed(sample)))
    assert a==b
    assert a["n"]==4 and a["distinct_scenarios"]==4
    assert a["source_selection"]=="sha_task_id_only_not_gt"
    assert m.validate(a,sample)==m.choose_dev_ids(sample)
    a["max_steps"]=123
    with pytest.raises(ValueError):
        m.validate(a,sample)

def test_only_unique_source_scenarios():
    assert len({x.split("_")[0] for x in m.choose_dev_ids(ids())})==4
    with pytest.raises(ValueError):
        m.choose_dev_ids(["x_1"]*30)

def _part(i):
    return {"status":"REAL_QWEN_DEV_ONE_TASK_COMPLETE","protocol":m.PROTOCOL,
      "source_sha256":sha256(b"frozen").hexdigest(),
      "original_model_gguf_sha256":m.MODEL_SHA,
      "shard":i,"n":1,"cases":[{
        "task_sha256":sha256(str(i).encode()).hexdigest(),
        "state_success":i==2,"state_tests_pass":i,"state_tests_total":7,
        "student_completed_text":i!=3,"steps":5,"tool_calls":4,
        "world_api_calls":2,"invalid_tool_calls":1,"seconds":20.0}],
      "no_training":True,"no_QLoRA":True,"no_dev_groundtruth_in_model":True}

def test_aggregate_requires_all_four_verified_original_model_source():
    parts=[_part(i) for i in range(4)]
    out=m.aggregate(parts)
    assert out["state_successes"]==1 and out["world_api_calls"]==8
    assert not out["promotion_allowed"] and out["no_GAIA"]
    with pytest.raises(ValueError):
        m.aggregate(parts[:3])
    parts[2]["no_dev_groundtruth_in_model"]=False
    with pytest.raises(ValueError):
        m.aggregate(parts)

def test_fixed_original_model_baseline():
    assert m.MAX_STEPS==20 and m.MAX_API_CALLS==48 and m.SHARDS==4
    assert len(m.MODEL_SHA)==64
    assert m.PROTOCOL.endswith("CLOSEDLOOP-V1")
