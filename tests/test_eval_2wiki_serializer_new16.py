"""Source-lock and paired strict-EX tests; never call model or use GAIA data."""
from __future__ import annotations
from hashlib import sha256
import json
import pytest

import scripts.eval_2wiki_serializer_new16 as m


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


def test_source_freeze_selects_16_disjoint_and_locks_same_serializer(monkeypatch):
    old=_linked_prior()
    third=_novel_prior(old)
    monkeypatch.setattr(m,"LINKED_PREVIOUS_SHA",old["cases_sha"])
    monkeypatch.setattr(m,"NOVEL_PREVIOUS_SHA",third["new8_sha"])
    cases=[_row(i) for i in range(28)]
    frozen=m.freeze(cases,old,third)
    assert frozen["n"]==16
    assert len({c["id"] for c in frozen["cases"]})==16
    assert frozen["known_prior_ids_excluded"]==24
    assert frozen["serializer_sha256"]==m.SERIALIZER_LOCKED_SHA
    assert frozen["model_digest"]==m.DIGEST
    m.validate_frozen(frozen)
    bad=json.loads(json.dumps(frozen))
    bad["cases"][0]["answer"]="secret injected from gold"
    with pytest.raises(RuntimeError,match="drift"):
        m.validate_frozen(bad)


def test_source_freeze_fails_closed_if_one_old_case_used(monkeypatch):
    old=_linked_prior()
    third=_novel_prior(old)
    monkeypatch.setattr(m,"LINKED_PREVIOUS_SHA",old["cases_sha"])
    monkeypatch.setattr(m,"NOVEL_PREVIOUS_SHA",third["new8_sha"])
    bad=_row(999)
    bad["id"]=old["cases"][0]["id"]
    with pytest.raises(RuntimeError,match="Insufficient"):
        m.freeze([bad]+[_row(i) for i in range(15)],old,third)


def test_aggregate_requires_complete_unique_16_and_paired_stats():
    cases=[
        {
            "task_hash":sha256(f"synthetic-{i}".encode()).hexdigest(),
            "original_exact":i in (6,7,8),
            "serialized_exact":i in (0,1,2,3,6,7,8),
            "original_completed":True,"stop_reason":"final",
            "raw_sha":sha256(f"r{i}".encode()).hexdigest(),
            "serialized_sha":sha256(f"s{i}".encode()).hexdigest(),
            "formatter_status":"accepted_verbatim",
            "extra_formatter_call":True,"tool_calls":2,"tool_errors":0,
            "original_agent_time_seconds":1.0,"not_new_reasoning":True,
        } for i in range(16)]
    shards=[{"status":"complete","protocol":m.PROTOCOL,"shard":i,"n":4,
             "source_sha":"synthetic-locked",
             "serializer_sha":m.SERIALIZER_SHA,"model_digest":m.DIGEST,
             "cases":cases[i*4:(i+1)*4]} for i in range(4)]
    got=m.aggregate(shards)
    assert (got["original_exact"],got["serialized_exact"])==(3,7)
    assert (got["rescues"],got["regressions"])==(4,0)
    assert got["positive_screen_only"] is True
    assert got["promotion_allowed"] is False
    shards[2]["cases"][0]["task_hash"]=shards[0]["cases"][0]["task_hash"]
    with pytest.raises(RuntimeError,match="duplicate"):
        m.aggregate(shards)


def test_serializer_is_same_frozen_verbatim_rule():
    import inspect
    from scripts.eval_hotpot_serializer_new16 import serialize,SERIALIZER_SHA
    assert SERIALIZER_SHA==m.SERIALIZER_LOCKED_SHA
    source=inspect.getsource(serialize)
    assert "re.search(re.escape(candidate),answer" in source
    assert 'tools=[]' in source
    run_source=inspect.getsource(m.one_case)
    assert "serialize(question,original," in run_source
    assert "case[\"answer\"]" in run_source
    assert "serialize(question,original" in run_source
