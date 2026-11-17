"""CPU-only confirmation contract: 48 public 2Wiki cases and zero known source reuse."""
from __future__ import annotations

from hashlib import sha256
import inspect
import json

import pytest

import scripts.eval_2wiki_serializer_confirm48 as m
def _row(i:int)->dict:
    titles=[f"2Wiki synthetic case {i} title {j}" for j in range(7)]
    original={t:[f"Answer_{i} on {t}",f"Second sentence {i} {j}"]
              for j,t in enumerate(titles)}
    return {
        "id":f"fake-2wiki-id-{i}","question":f"Question about synthetic case {i}?",
        "answer":f"Answer_{i}",
        "context":{"title":titles,"sentences":list(original.values())},
        "supporting_facts":{"title":titles[:4],"sent_id":[0,0,0,0]},
    }


def _linked_prior()->dict:
    cases=[{"id":f"old-{i}","support":{m.page_path(f"old support {i} {j}"):["x"]
           for j in range(4)}} for i in range(8)]
    return {
        "protocol":"2wiki-validation-qwen9b-visible-local-links-pair-v1",
        "source_revision":m.REV,"cases":cases,
        "cases_sha":sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
    }


def _novel_prior(old:dict)->dict:
    cases=[{"id":f"novel-{i}","support":{
        m.page_path(f"novel support {i} {j}"):["source"] for j in range(4)}}
        for i in range(8)]
    return {
        "protocol":"qwen35-9b-v3b-vs-base-2wiki-novel-validation8-v1",
        "source_revision":m.REV,"source8_sha":old["cases_sha"],
        "n":8,"cases":cases,
        "new8_sha":sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
    }




def prior_positive16() -> dict:
    cases = []
    for i in range(16):
        titles = [f"published positive16 case {i} doc {j}" for j in range(4)]
        cases.append({
            "id":f"prior-positive16-id-{i}",
            "support":{m.page_path(t):[f"old sentence {i}"] for t in titles},
        })
    return {
        "status":"FROZEN_BEFORE_NEW_QWEN_MODEL_INFERENCE",
        "protocol":"2WIKI-NEW16-ORIGINAL-QWEN9B-QUESTION-CONDITIONED-SERIALIZER-V1",
        "source_revision":m.REV,
        "model_digest":m.DIGEST,
        "n":16,
        "serializer_sha256":m.SERIALIZER_SHA,
        "frozen_source_sha256":sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
        "cases":cases,
    }


def lock_prior(monkeypatch):
    linked=_linked_prior()
    novel=_novel_prior(linked)
    old16=prior_positive16()
    monkeypatch.setattr(m,"LINKED_PREVIOUS_SHA",linked["cases_sha"])
    monkeypatch.setattr(m,"NOVEL_PREVIOUS_SHA",novel["new8_sha"])
    monkeypatch.setattr(m,"PRIOR_POSITIVE16_SOURCE_SHA",old16["frozen_source_sha256"])
    return linked,novel,old16


def test_confirmation_is_sha_locked_and_disjoint_40_prior_cases(monkeypatch):
    linked,novel,old16=lock_prior(monkeypatch)
    frozen=m.freeze([_row(i) for i in range(100)],linked,novel,old16)
    assert frozen["n"]==48
    assert frozen["shards"]==8
    assert frozen["known_prior_ids_excluded"]==40
    assert frozen["known_previous_document_hashes_excluded"]==160
    assert len({x["id"] for x in frozen["cases"]})==48
    assert frozen["serializer_sha256"]==m.SERIALIZER_LOCKED_SHA
    assert frozen["gate"]==m.MIN_GATE
    m.validate_frozen(frozen)

    old_ids={x["id"] for x in old16["cases"]}
    new_ids={x["id"] for x in frozen["cases"]}
    assert not new_ids & old_ids
    known_doc_ids={path.rsplit("/",1)[-1] for row in old16["cases"]
                   for path in row["support"]}
    new_doc_ids={path.rsplit("/",1)[-1] for row in frozen["cases"]
                 for path in row["support"]}
    assert not new_doc_ids & known_doc_ids
    bad=json.loads(json.dumps(frozen))
    bad["cases"][0]["answer"]="gold injection"
    with pytest.raises(RuntimeError,match="drift"):
        m.validate_frozen(bad)


def test_confirmation_previous_success16_identity_cannot_reappear(monkeypatch):
    linked,novel,old16=lock_prior(monkeypatch)
    reuse=_row(999)
    reuse["id"]=old16["cases"][0]["id"]
    with pytest.raises(RuntimeError,match="Insufficient"):
        m.freeze([reuse]+[_row(i) for i in range(47)],linked,novel,old16)


def test_confirmation_previous_success16_support_cannot_reappear(monkeypatch):
    linked,novel,old16=lock_prior(monkeypatch)
    reuse=_row(999)
    old_support=next(iter(old16["cases"][0]["support"]))
    # Source reader derives support file path from article title. Make a
    # synthetic prospective title collide with a prior source document.
    target_title=next(x for x in reuse["context"]["title"][:4])
    old_title="published positive16 case 0 doc 0"
    assert m.page_path(old_title)==old_support
    reuse["context"]["title"][0]=old_title
    reuse["supporting_facts"]["title"][0]=old_title
    with pytest.raises(RuntimeError,match="Insufficient"):
        m.freeze([reuse]+[_row(i) for i in range(47)],linked,novel,old16)


def paired_shards():
    rows=[]
    for i in range(48):
        rescue=i in range(9)
        right=i in range(14,26)
        rows.append({
            "task_hash":sha256(f"new48-case-{i}".encode()).hexdigest(),
            "original_exact":right,
            "serialized_exact":right or rescue,
            "original_completed":True,
            "stop_reason":"final",
            "raw_sha":sha256(f"raw {i}".encode()).hexdigest(),
            "serialized_sha":sha256(f"formatted {i}".encode()).hexdigest(),
            "formatter_status":"accepted_verbatim",
            "extra_formatter_call":True,
            "tool_calls":2,"tool_errors":0,
            "original_agent_time_seconds":1.0,
            "not_new_reasoning":True,
        })
    return [{
        "status":"complete","protocol":m.PROTOCOL,
        "shard":s,"n":6,"source_sha":"fixed-source",
        "serializer_sha":m.SERIALIZER_SHA,"model_digest":m.DIGEST,
        "cases":rows[s*6:(s+1)*6],
    } for s in range(8)]


def test_paired_48_exact_mcnemar_gate_and_nonpromotion():
    shards=paired_shards()
    summary=m.aggregate(shards)
    assert summary["n"]==48
    assert (summary["original_exact"],summary["serialized_exact"])==(12,21)
    assert (summary["rescues"],summary["regressions"])==(9,0)
    assert summary["exact_mcnemar_two_sided_p"]<0.05
    assert summary["positive_screen_only"] is True
    assert summary["promotion_allowed"] is False
    shards[5]["cases"][2]["task_hash"]=shards[1]["cases"][0]["task_hash"]
    with pytest.raises(RuntimeError,match="duplicate"):
        m.aggregate(shards)


def test_unchanged_verbatim_original_qwen_rule_and_model():
    from scripts.eval_hotpot_serializer_new16 import serialize
    assert m.SERIALIZER_SHA==m.SERIALIZER_LOCKED_SHA
    assert m.DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    src=inspect.getsource(m.one_case)
    assert "serialize(question,original" in src
    assert "gaia_score(original,case[\"answer\"])" in src
    assert "case[\"answer\"]" in src
    assert "result=AgentRuntime(model" in src
