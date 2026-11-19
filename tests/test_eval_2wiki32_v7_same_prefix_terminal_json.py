"""CPU-only protocol unit tests for new disjoint 2Wiki32 terminal JSON fork."""
from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from scripts.eval_2wiki32_v7_same_prefix_terminal_json import (
    ANSWER_SCHEMA, BASE_SHA, V7_SHA, PairedLastStepModel,
    PROTOCOL, GATE, N, SHARDS, h, check, aggregate, json_terminal_request
)


def test_no_background_stronger_model_and_same_prefix_last_step():
    class Dummy:
        def __init__(self): self.calls=0
        def complete_final(self,messages,tools):
            self.calls+=1
            return AssistantTurn(content="<tool_call>still searching</tool_call>")
    origin=Dummy()
    seen=[]
    def structured(model,messages):
        seen.append(messages)
        return "Correct answer","valid_json_answer"
    fork=PairedLastStepModel(origin,structured)
    messages=[{"role":"system","content":"same"},{"role":"user","content":"hard task"}]
    original=fork.complete_final(messages,[{"name":"search"}])
    assert original.content.startswith("<tool_call>")
    assert fork.structured_answer=="Correct answer"
    assert fork.terminal_fork_count==1
    assert origin.calls==1
    assert seen==[messages]
    assert GATE["min_terminal_forks"]==3
    assert N==32 and SHARDS==8 and len(BASE_SHA)==len(V7_SHA)==64


def test_strict_json_api_contract_and_json_rejection(monkeypatch):
    from scripts import eval_2wiki32_v7_same_prefix_terminal_json as m
    calls=[]
    class Response:
        def __init__(self,content): self.content=content
        def raise_for_status(self): pass
        def json(self): return {"choices":[{"message":{"content":self.content}}]}
    def post(url, json, timeout):
        calls.append((url,json,timeout))
        return Response('{"answer":"Sunday On The Rocks"}')
    monkeypatch.setattr(m.requests,"post",post)
    class Model:
        base_url="http://127.0.0.1:8080"
        model="qwen3.5-9b"
        max_new_tokens=512
        timeout_s=300
    messages=[{"role":"user","content":"fixed input"}]
    out,status=json_terminal_request(Model(),messages)
    assert (out,status)==("Sunday On The Rocks","valid_json_answer")
    payload=calls[0][1]
    assert payload["messages"] is messages
    assert "tools" not in payload
    assert payload["response_format"]["type"]=="json_schema"
    assert payload["response_format"]["json_schema"]["schema"]==ANSWER_SCHEMA


def test_gold_scoring_is_separate_and_unscorable_json_cannot_regress_other_30():
    cs=[{"id":str(i),"question":"q","files":{},"index_titles":{}} for i in range(32)]
    source={"protocol":PROTOCOL,"status":"NEW32_SOURCE_FROZEN_BEFORE_MODEL",
            "n":N,"shards":SHARDS,"source_sha":h(cs),
            "model_base_sha":BASE_SHA,"v7_adapter_sha":V7_SHA,
            "gate":GATE,"no_publisher_gold_for_model":True,"cases":cs}
    check(source)
    labels={"protocol":PROTOCOL,"source_sha":source["source_sha"],
            "references":[{"id":str(i),"answer":"Correct","support_paths":[]} for i in range(32)]}
    parts=[]
    for shard in range(8):
        rows=[]
        for i in range(shard*4,(shard+1)*4):
            forced=i in (2,4,6)
            rows.append({"id":str(i),
                         "original_answer":"" if forced else "Correct",
                         "original_completed":not forced,
                         "original_stop":"invalid_final" if forced else "final",
                         "structured_answer":"Correct" if forced else "",
                         "structured_status":"valid_json_answer" if forced else "not_reached",
                         "final_forks":1 if forced else 0,"model_calls_added":1 if forced else 0,
                         "tool_calls":8,"duplicate_blocks":0,
                         "successful_read_paths":[]})
        parts.append({"protocol":PROTOCOL,"status":"GOLD_BLIND_ALL_ORIGINAL_AND_JSON_FINALS_SEALED",
                      "source_sha":source["source_sha"],"shard":shard,"n":4,
                      "model_base_sha":BASE_SHA,"adapter_sha":V7_SHA,
                      "gold_not_loaded":True,"cases":rows})
    r=aggregate(source,labels,parts)
    assert r["counts"]["terminal_forks"]==3
    assert r["counts"]["extra_calls"]==3
    assert r["counts"]["rescues"]==3 and r["counts"].get("regressions",0)==0
    assert r["counts"]["baseline_correct"]==29
    assert r["counts"]["structured_correct"]==32
    assert r["net_correct"]==3 and r["positive_screen_only"]
    assert not r["promotion_allowed"] and r["no_GAIA"]
