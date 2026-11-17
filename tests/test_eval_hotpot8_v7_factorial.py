"""CPU contracts, not Hotpot model inference or accuracy results."""
from hashlib import sha256
import pytest

from scripts import eval_hotpot8_v7_factorial as f
from scripts import eval_2wiki8_v7_factorial as old
def all_shards():
    shards=[]
    for mode in ('base','adapter'):
        for shard in range(4):
            cases=[]
            for i in range(2):
                n=2*shard+i
                arms={}
                for suffix in ('native','serializer'):
                    score=(n<2 if mode=='base' else n<3) or (suffix=='serializer' and n==5)
                    arms[f'{mode}_{suffix}']={
                        'correct':score,'model_calls':1+(suffix=='serializer'),
                        'elapsed_s':1.5,'completed':True,'tool_calls':2}
                cases.append({'task_sha256':sha256(str(n).encode()).hexdigest(),
                              'source_sha':sha256(b'new').hexdigest(),'arms':arms})
            shards.append({
                'status':'COMPLETE_PHYSICAL_ONE_MODEL_MODE','protocol':f.PROTOCOL,
                'mode':mode,'shard':shard,'n':2,'source_sha':sha256(b'new').hexdigest(),
                'base_gguf_sha256':old.BASE_SHA,'adapter_gguf_sha256':old.V7_SHA,
                'serializer_sha256':f.old.SERIALIZER_SHA,'cases':cases})
    return shards


def sample_source():
    cases=[]
    for k in range(8):
        t="bridge" if k<4 else "comparison"
        cases.append({"id":f"new_{k}","type":t,"question":"Which?",
                      "answer":"Name","support":{f"pages/x{k}.txt":["a"],f"pages/y{k}.txt":["b"]},
                      "files":{f"pages/x{k}.txt":"a",f"pages/y{k}.txt":"b"},
                      "index_titles":{"x":"a"}})
    return {
      "status":"PREINFERENCE_FROZEN_PUBLIC_HOTPOT8","protocol":f.PROTOCOL,
      "n":8,"shards":4,"source_revision":f.REV,
      "task_types":{"bridge":4,"comparison":4},
      "excluded_prior_task_ids":24,
      "base_gguf_sha256":old.BASE_SHA,
      "adapter_gguf_sha256":old.V7_SHA,
      "serializer_sha256":f.old.SERIALIZER_SHA,
      "cases":cases,"source_sha":old.stable_sha(cases)
    }


def test_source_hash_and_balanced_support_gate():
    sample=sample_source()
    f.validate(sample)
    broken=dict(sample)
    broken["cases"]=sample["cases"][:-1]
    with pytest.raises(ValueError):f.validate(broken)
    broken=dict(sample)
    broken["source_sha"]="0"*64
    with pytest.raises(ValueError):f.validate(broken)
    broken=dict(sample)
    broken["task_types"]={"bridge":8}
    with pytest.raises(ValueError):f.validate(broken)


def test_fails_closed_without_prior_exclusion_sources():
    with pytest.raises(ValueError):
        f.freeze([],{"cases":[]},{"cases":[]})


def test_aggregate_four_arms_with_strict_matched_ids():
    shards=all_shards()
    for s in shards:
        s["protocol"]=f.PROTOCOL
        for r in s["cases"]:
            task_index=int(next(k for k in range(8) if sha256(str(k).encode()).hexdigest()==r["task_sha256"]))
            r["type"]="bridge" if task_index<4 else "comparison"
    result=f.aggregate(shards)
    assert result["n"]==8
    assert result["status"]=="COMPLETE_HOTPOT_DIAGNOSTIC_2X2_NO_PROMOTION"
    assert result["case_types"]=={"bridge":4,"comparison":4}
    assert result["totals"]["base_native"]["correct"]==2
    assert result["totals"]["base_serializer"]["correct"]==3
    assert result["promotion_allowed"] is False
