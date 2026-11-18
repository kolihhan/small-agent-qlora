import pytest

from scripts.eval_nextsearch_qwen_source_val import freeze,aggregate,PROTOCOL,DIGEST


def source_row(i, source="asearcher", split="validation"):
    return {
        "id":source+"/"+split+"-"+str(i),
        "source":source,"license":"apache-2.0",
        "band":"hard","gold":{"answer":str(10000+i)},
        "messages":[
            {"role":"system","content":"Please research"},
            {"role":"user","content":f"What is the verifiable value for test item {i}?"},
        ],
    }


def test_heldout_source_freeze_excludes_train_ids_and_uses_three_sources():
    train=[source_row(i,"asearcher",split="train") for i in range(12)]
    val=[source_row(i,source)
         for source in ("asearcher","taskcraft","quest") for i in range(5)]
    # Deliberately expose one source ID in training. Must not choose it.
    train.append(val[0])
    frozen=freeze(train,val)
    assert len(frozen["cases"])==8
    assert val[0]["id"] not in {c["id"] for c in frozen["cases"]}
    assert len({c["source"] for c in frozen["cases"]})==3
    assert frozen["no_gold_to_model"]
    assert frozen["heldout_by_task_id"]


def test_aggregate_never_confuses_gaia_and_source_baseline():
    ids=[f"task-{i}" for i in range(8)]
    shards=[]
    for shard in range(4):
        shards.append({
            "protocol":PROTOCOL,"digest":DIGEST,
            "cohort_sha":"frozen",
            "shard":shard,
            "cases":[{
                "id":ident, "source":"taskcraft",
                "exact":i%3==0, "completed":True, "tool_calls":4,
                "tool_errors":0, "actions_by_name":{"search":2,"fetch":2},
                "seconds":1.5
            } for i,ident in [(j,ids[j]) for j in range(2*shard,2*shard+2)]],
        })
    report=aggregate(shards)
    assert report["tasks"]==8
    assert report["exact"]==3
    assert report["tool_calls"]==32
    assert report["search_calls"]==16
    assert report["fetch_calls"]==16
    assert report["not_GAIA"] and not report["promotion_allowed"]
    with pytest.raises(ValueError,match="incomplete"):
        aggregate(shards[:3])
