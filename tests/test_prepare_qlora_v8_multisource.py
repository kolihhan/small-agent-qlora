"""Multi-source QLoRA supervision must not learn GAIA or replay heldout parents."""
import pytest
from hashlib import sha256

from scripts.prepare_qlora_v8_multisource import merge_supervision


def _r(source, id, *, user=None, annotated=False):
    text=user or ("Task "+id)
    obj={
        "task_id":id,"source":source,"verified":True,
        "training_action_indices":[2],"messages":[
            {"role":"system","content":"policy"},
            {"role":"user","content":text},
            {"role":"assistant","content":"done"},
        ],
        "provenance":{
            "license":"CC-BY-4.0","generator":"g","generator_version":"v1",
            "oracle_type":"human","oracle_version":"1",
        },
    }
    if annotated:
        obj["derived_from_annotation_not_observed_model"]=True
        obj["source_id_sha256"]=sha256(id.encode()).hexdigest()
        obj["supporting_paragraph_hashes"]=[
            sha256((id+":p:"+str(n)).encode()).hexdigest() for n in range(3)
        ]
        obj["selection_method"]="human_annotated_support_paths_not_model_generated"
        obj["composition_family"]="3hop1"
        obj["hop_count"]=3
    return obj


def test_combines_all_families_whole_parent_without_mislabeling():
    base=[_r("synthetic-small-agent-policy-v3/train","synthetic-1")]
    finqa=[_r("finqa-train","finqa-1")]
    mu=[_r("musique-train","musique-1",annotated=True)]
    held=[_r("musique-train","other-hold",annotated=True)]
    train,hold,report=merge_supervision(base,finqa,mu,held,set())
    assert len(train)==3 and len(hold)==1
    assert report["train_actions"]==3
    assert report["annotation_oracle_parent_count"]==1
    assert report["observed_finqa_parent_count"]==1
    assert report["no_gaia_training"] is True
    assert {r["task_id"] for r in train}.isdisjoint({r["task_id"] for r in hold})


def test_rejects_duplicate_parent_or_protected_question():
    base=[_r("synthetic-small-agent-policy-v3/train","dup")]
    with pytest.raises(ValueError,match="duplicate"):
        merge_supervision(base,[_r("finqa-train","dup")],[],[],set())
    from gaia_small_agent.training.protection import question_hash
    with pytest.raises(ValueError,match="protected"):
        merge_supervision(base,[],[],[],{question_hash("Task dup")})


def test_rejects_annotation_without_origin_or_evidence_holdout_overlap():
    base=[_r("synthetic-small-agent-policy-v3/train","base")]
    m=_r("musique-train","musique",annotated=True)
    del m["derived_from_annotation_not_observed_model"]
    with pytest.raises(ValueError,match="source annotation"):
        merge_supervision(base,[],[m],[],set())
    a=_r("musique-train","musique",annotated=True)
    b=_r("musique-train","holdout",annotated=True)
    b["supporting_paragraph_hashes"]=a["supporting_paragraph_hashes"]
    with pytest.raises(ValueError,match="support"):
        merge_supervision(base,[],[a],[b],set())


def test_prevent_finqa_dev_or_any_gaia_train():
    with pytest.raises(ValueError,match="forbidden"):
        merge_supervision([_r("gaia-val","g")],[],[],[],set())
    with pytest.raises(ValueError,match="forbidden"):
        merge_supervision([],[_r("finqa-dev","bad")],[],[],set())
