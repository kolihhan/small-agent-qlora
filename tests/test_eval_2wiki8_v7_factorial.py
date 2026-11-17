"""CPU-only invariant tests, NOT Qwen accuracy or an unseen result."""
import copy
from hashlib import sha256
import pytest
from scripts import eval_2wiki8_v7_factorial as m

def sha(x):
    return sha256(x.encode()).hexdigest()

def shard(shard,mode):
    rows=[]
    for i in range(2):
        task=sha(str(2*shard+i))
        arms={}
        for suffix in ("native","serializer"):
            name=f"{mode}_{suffix}"
            correct = (2*shard+i)<(2 if mode=="base" else 3)
            if suffix=="serializer":
                correct = correct or (2*shard+i)==5
            arms[name]={"correct":correct,"model_calls":1+(suffix=="serializer"),
                        "elapsed_s":1.5+(suffix=="serializer"),
                        "completed":True,"tool_calls":2,"tool_errors":0}
        rows.append({"task_sha256":task,"source_sha":sha("locked-source"),"arms":arms})
    return {"status":"COMPLETE_PHYSICAL_ONE_MODEL_MODE","protocol":m.PROTOCOL,
            "mode":mode,"shard":shard,"n":2,"source_sha":sha("locked-source"),
            "base_gguf_sha256":m.BASE_SHA,"adapter_gguf_sha256":m.V7_SHA,
            "serializer_sha256":m.SERIALIZER_SHA,"cases":rows}

def all_shards():
    return [shard(s,mode) for mode in ("base","adapter") for s in range(4)]

def test_complete_metric_contract():
    x=m.aggregate(all_shards())
    assert x["n"]==8 and x["physical_shards"]==8
    assert x["totals"]["base_native"]["correct"]==2
    assert x["totals"]["adapter_native"]["correct"]==3
    assert x["totals"]["base_serializer"]["correct"]==3
    assert x["totals"]["adapter_serializer"]["correct"]==4
    assert x["paired"]["base_native->adapter_native"]["rescues"]==1
    assert x["paired"]["base_native->adapter_native"]["regressions"]==0
    assert x["interaction_net"]==0
    assert x["promotion_allowed"] is False
    assert x["aa_variance_not_yet_evaluated"] is True

def test_fail_closed_missing_shards_and_duplicate_tasks():
    with pytest.raises(ValueError):
        m.aggregate(all_shards()[:-1])
    rows=all_shards()
    rows[2]["cases"][0]["task_sha256"]=rows[0]["cases"][0]["task_sha256"]
    with pytest.raises(ValueError):
        m.aggregate(rows)

def test_freeze_rejects_fabricated_prior48():
    with pytest.raises((RuntimeError,ValueError)):
        m.freeze_source([],{"cases":[]},None,None,None)

def test_exact_p_value():
    assert m._p(6,0)==0.03125
