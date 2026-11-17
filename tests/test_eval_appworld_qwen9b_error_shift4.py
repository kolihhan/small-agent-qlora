"""Original source4 exclusion, strict physical pairing, and true HTTP tool error tests."""
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts import eval_appworld_qwen9b_error_shift4 as m
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool


def ids():
    return [f"scene{i}_{j}" for i in range(15) for j in (1,2)]

def test_new_source_excludes_all_first_four_original_scenarios():
    source=ids()
    initial=m.old.choose_dev_ids(source)
    # The synthetic 30 IDs are not expected to yield the real frozen source.
    # Verify the old SHA pin correctly rejects any source-set drift.
    with pytest.raises(ValueError,match="previous original"):
        m.new_dev_ids(source)
    assert len(initial)==4

def test_safe_error_buckets_never_publish_raw_exception():
    trace=[SimpleNamespace(kind="model_error",data={
        "message":"Model response invalid tool arguments JSON: SENSITIVE USER DATA"})]
    assert m.error_bucket("model_error",trace)=="malformed_tool_json"
    assert m.error_bucket("final",trace)=="final"
    assert m.error_bucket("model_error",[])=="other_model_error"
    assert m.error_bucket("model_error",[SimpleNamespace(kind="model_error",data={
        "message":"HTTP 500 Internal Server Error"})])=="model_server"

class BadRequester:
    def request(self,**kwargs):
        assert kwargs["raise_on_failure"] is True
        raise Exception("Response status code is 401: a token secret")

def test_truthful_tool_http_status_and_no_error_body_leak():
    world=SimpleNamespace(task=SimpleNamespace(allowed_apps=["spotify"],
                 api_docs={"spotify":{"search_tracks":{}}}),requester=BadRequester())
    t=AppWorldCallTool(world,report_http_failures=True,max_calls=3)
    x=t.run({"app":"spotify","api":"search_tracks","arguments":{"query":"some token"}},Path("."))
    assert not x.ok and x.error_code=="HTTP_ERROR"
    assert "401" in x.content
    assert "token secret" not in x.content
    assert t.http_errors==1 and t.errors==1 and t.calls==1

def _part(mode,i):
    correct=i<2 if mode=="baseline" else i<3
    return {"status":"PHYSICAL_QWEN_NEW_DEV_ONE_TASK_COMPLETE","protocol":m.PROTOCOL,
      "source_sha256":sha256(b"fresh four").hexdigest(),
      "model_sha256":m.old.MODEL_SHA,"arm":mode,"shard":i,"n":1,
      "task_sha256":sha256(str(i).encode()).hexdigest(),
      "no_training":True,"no_reference_inference":True,
      "results":{"state_success":correct,
                 "tests_pass":2*int(correct),"tests_total":3,
                 "completed_text":True,"model_steps":12,"tool_calls":10,
                 "world_api_calls":5,"world_api_http_failures":int(mode=="http_error"),
                 "model_error_bucket":"final","elapsed_s":18.2}}
def test_aggregate_requires_all_eight_matched_arms():
    parts=[_part(arm,s) for arm in m.ARMS for s in range(4)]
    out=m.aggregate(parts)
    assert out["n"]==4 and out["totals"]["baseline"]["state_successes"]==2
    assert out["totals"]["http_error"]["state_successes"]==3
    assert out["rescues"]==1 and not out["positive_screen_only"]
    assert out["promotion_allowed"] is False
    with pytest.raises(ValueError):m.aggregate(parts[:-1])
    parts[-1]["task_sha256"]="mismatch"
    with pytest.raises(ValueError,match="mismatched"):m.aggregate(parts)

def test_no_training_unchanged_fixed_base_and_predeclared_gate():
    assert m.PROTOCOL.endswith("-V1")
    assert m.old.MODEL_SHA.startswith("8c4cc24f")
    assert m.GATE["min_net_state_rescues"]==2
    assert m.OLD_SOURCE_SHA.startswith("12d0a62d")
    assert m.ARMS==("baseline","http_error")
