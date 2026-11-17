"""Fail-closed same-model grammar-final recovery tests (NO GAIA labels/model calls)."""
from __future__ import annotations

import json
from hashlib import sha256
from unittest.mock import Mock

import pytest

import scripts.eval_gaia_terminal_grammar15 as m
from gaia_small_agent.agent.types import AssistantTurn,ModelRuntimeError
from gaia_small_agent.model.openai_compat import OpenAICompatModel


def test_original_gaia_unchanged_and_exposed_15_is_explicit(monkeypatch):
    fake=[{"task_id":str(i),"Level":1 if i<60 else 2 if i<130 else 3}
          for i in range(165)]
    old={
        "task_id_sequence_sha256":m.SOURCE_SHA,"n":15,
        "hashed_tasks":[sha256(str(i).encode()).hexdigest() for i in range(15)],
    }
    monkeypatch.setattr(m,"old_freeze",lambda rows:old)
    f=m.freeze(fake)
    assert f["source_seen_before"].startswith("GAIA terminal")
    assert f["cannot_promote_or_claim_hidden_generalization"] is True
    assert f["fallback_only_after_failed_terminal"] is True
    assert f["preregistered_gate"]==m.GATE
    old["task_id_sequence_sha256"]="wrong"
    with pytest.raises(RuntimeError,match="source"):
        m.freeze(fake)


def test_schema_parse_is_strict_even_when_server_claims_json(monkeypatch):
    class Resp:
        def __init__(self,content,tool_calls=None):
            self.content=content
            self.tool_calls=tool_calls
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices":[{"message":{"content":self.content,
                                            "tool_calls":self.tool_calls}}]}
    model=OpenAICompatModel(model="test")
    seen=[]
    def post(url,json,timeout):
        seen.append(json)
        return Resp('{"answer":"42"}')
    monkeypatch.setattr(m.requests,"post",post)
    ans,status=m._request_json_final(model,[{"role":"user","content":"Question"}],
                                     [{"type":"function","function":{"name":"search"}}])
    assert (ans,status)==("42","accepted_schema")
    assert seen[0]["response_format"]["schema"]==m.JSON_SCHEMA
    assert seen[0]["tool_choice"]=="none"
    assert seen[0]["tools"]
    for answer in ('42','{"answer":42}','{"answer":"<tool_call> fake"}',
                   '{"answer":"x","reason":"extra"}','{"answer":""}'):
        monkeypatch.setattr(m.requests,"post",lambda *args,**kwargs:Resp(answer))
        got,status=m._request_json_final(model,[],[])
        assert status!="accepted_schema"


def test_keeps_all_valid_legacy_finals_identical(monkeypatch):
    monkeypatch.setattr(OpenAICompatModel,"complete_final",
                        lambda self,messages,tools:AssistantTurn(content="March 2020",tool_calls=[]))
    fake=lambda *args,**kwargs:pytest.fail("fallback must NOT run on valid original")
    monkeypatch.setattr(m,"_request_json_final",fake)
    model=m.FailedFinalGrammarModel()
    ans=model.complete_final([{"role":"user","content":"when?"}],[])
    assert ans.content=="March 2020"
    assert model.terminals[-1]["fallback_attempted"] is False
    assert model.terminals[-1]["original_completed"] is True


def test_on_legacy_invalid_final_uses_same_context_one_extra_call(monkeypatch):
    monkeypatch.setattr(OpenAICompatModel,"complete_final",
        lambda self,messages,tools:AssistantTurn(content='<tool_call>{"name":"search"}</tool_call>',tool_calls=[]))
    observed=[]
    def fallback(model,messages,tools):
        observed.append((model,messages,tools))
        return "2017","accepted_schema"
    monkeypatch.setattr(m,"_request_json_final",fallback)
    mdl=m.FailedFinalGrammarModel()
    messages=[{"role":"system","content":"original"},{"role":"user","content":"question"}]
    tools=[{"type":"function","function":{"name":"search"}}]
    got=mdl.complete_final(messages,tools)
    assert got.content=="2017"
    assert len(observed)==1 and observed[0][1] is messages and observed[0][2] is tools
    assert mdl.terminals[-1]["original_completed"] is False
    assert mdl.terminals[-1]["fallback_attempted"] is True


def test_grammar_failure_preserves_invalid_or_error_original(monkeypatch):
    monkeypatch.setattr(OpenAICompatModel,"complete_final",
        lambda self,messages,tools:AssistantTurn(content="<tool_call>bad</tool_call>",tool_calls=[]))
    monkeypatch.setattr(m,"_request_json_final",lambda *a:("","structured_error"))
    mdl=m.FailedFinalGrammarModel()
    got=mdl.complete_final([],[])
    assert got.content.startswith("<tool_call>")
    assert mdl.terminals[-1]["fallback_status"]=="structured_error"


def _paired_shards():
    rows=[]
    for i in range(15):
        rescued=i in (0,1)
        rows.append({
            "task_hash":sha256(f"source-{i}".encode()).hexdigest(),
            "level":1 if i==0 else 2 if i==1 else 3,
            "original_correct":False if rescued else i%3==0,
            "grammar_correct":True if rescued else i%3==0,
            "original_completed":False if rescued else True,
            "grammar_completed":True,
            "original_status":"invalid_final" if rescued else "valid_final",
            "grammar_status":"final",
            "fallback_attempted":rescued,
            "fallback_status":"accepted_schema" if rescued else "not_needed",
            "fallback_s":0.2 if rescued else 0.0,
            "tool_calls":2,"tool_errors":0,"same_tools_and_history":True,
        })
    return [{
        "status":"complete","protocol":m.PROTOCOL,"source_sha":m.SOURCE_SHA,
        "grammar_sha":m.GRAMMAR_SHA,"base_sha":m.BASE_SHA,
        "adapter_sha":m.V7_SHA,"shard":i,"n":5,
        "cases":rows[i*5:(i+1)*5],
    } for i in range(3)]


def test_exact_15_paired_aggregate_and_no_ood_promotion():
    shards=_paired_shards()
    got=m.aggregate(shards)
    assert got["rescues"]==2 and got["regressions"]==0
    assert got["positive_diagnostic_only"] is True
    assert got["promotion_allowed"] is False
    assert got["EXPOSED_prior_GAIA15"] is True
    shards[2]["cases"][1]["task_hash"]=shards[0]["cases"][1]["task_hash"]
    with pytest.raises(RuntimeError,match="duplicate"):
        m.aggregate(shards)
