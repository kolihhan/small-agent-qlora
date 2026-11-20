"""Synthetic first-401 action chain and model-error-category privacy tests."""
from types import SimpleNamespace

import pytest
import requests

from gaia_small_agent.agent.types import ModelRuntimeError,TraceEvent
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools.appworld_auth_chain_audit import first_401_action_chain


def fake_world():
    protected={"parameters":[{"name":"access_token","type":"string","required":True}]}
    return SimpleNamespace(task=SimpleNamespace(api_docs={
        "calendar":{"protected":protected,"login":{"parameters":[]}}}))


def call(step,id,tool,kwargs):
    return TraceEvent("tool_call",step,{"id":id,"name":tool,"arguments":kwargs})


def result(step,id,tool,ok,content="",code=None):
    return TraceEvent("tool_result",step,{"id":id,"name":tool,"ok":ok,
                                        "content":content,"error_code":code})


def test_401_chain_before_failure_and_no_credentials_export():
    token="SECRET_PW_NEVER_LEAK"
    trace=[
        call(1,"d1","appworld_docs",{"app":"calendar","api":"protected"}),
        result(1,"d1","appworld_docs",True,
               '{"parameters":[{"name": "access_token","required":true}]}'),
        call(2,"l1","appworld_docs",{"app":"calendar","api":"login"}),
        result(2,"l1","appworld_docs",True,"public login docs"),
        call(3,"l2","appworld_call",{"app":"calendar","api":"login","arguments":{"password":token}}),
        result(3,"l2","appworld_call",True,'{"access_token":"'+token+'"}'),
        call(4,"x1","appworld_call",{"app":"calendar","api":"protected","arguments":{}}),
        result(4,"x1","appworld_call",False,
               "ToolError[HTTP_ERROR]: AppWorld API rejected this request (HTTP 401).",
               "HTTP_ERROR"),
        TraceEvent("model_error",5,{"diagnostic_code":"MODEL_TOOL_ARGUMENTS_JSON_MALFORMED",
                                   "message":"secret "+token}),
    ]
    d=first_401_action_chain(trace,fake_world(),
        [{"http_status":401,"model_step":4,"api_call_index":2}],20,"model_error")
    assert d["first401_observed"] and d["first401_eligible_for_root_review"]
    assert d["first401_step"]==4 and d["remaining_decision_steps"]==16
    assert d["relevant_api_docs_actually_viewed"]
    assert d["access_token_requirement_visible_in_actual_view"]
    assert d["login_api_docs_actually_viewed"] and d["login_api_attempted"]
    assert d["login_api_succeeded"] and not d["request_supplied_nonempty_token"]
    assert d["public_auth_explanation_class"]=="REQUIRED_ACCESS_TOKEN_ABSENT"
    assert d["model_error_category"]=="MODEL_TOOL_ARGUMENTS_JSON_MALFORMED"
    assert token not in repr(d) and "calendar" not in repr(d) and "protected" not in repr(d)


def test_401_with_no_prior_docs_and_present_token():
    trace=[
        call(2,"x","appworld_call",
             {"app":"calendar","api":"protected","arguments":{"access_token":"SECRET"}}),
        result(2,"x","appworld_call",False,"HTTP 401","HTTP_ERROR")
    ]
    d=first_401_action_chain(trace,fake_world(),
        [{"http_status":401,"model_step":2,"api_call_index":1}],20,"final")
    assert not d["relevant_api_docs_actually_viewed"]
    assert not d["login_api_attempted"]
    assert d["request_supplied_nonempty_token"]
    assert d["public_auth_explanation_class"]=="REQUIRED_ACCESS_TOKEN_PRESENT_REJECTED"
    assert "SECRET" not in repr(d)


def test_ambiguous_multiple_401_same_step_fails_closed():
    trace=[
        call(2,"x","appworld_call",{"app":"calendar","api":"protected","arguments":{}}),
        result(2,"x","appworld_call",False,"HTTP 401","HTTP_ERROR"),
        call(2,"y","appworld_call",{"app":"calendar","api":"protected","arguments":{}}),
        result(2,"y","appworld_call",False,"HTTP 401","HTTP_ERROR")]
    out=first_401_action_chain(trace,fake_world(),
        [{"http_status":401,"model_step":2},{"http_status":401,"model_step":2}],20,"final")
    assert not out["first401_eligible_for_root_review"]
    assert out["ambiguous_same_step_401"]


def test_typed_model_errors_preserve_existing_stop_reason(monkeypatch):
    model=OpenAICompatModel("fixed-qwen",base_url="http://localhost:12345")
    class BadResponse:
        status_code=500
        def raise_for_status(self):
            raise requests.HTTPError("PRIVATE_SERVER_LOG",response=self)
    monkeypatch.setattr(requests,"post",lambda *a,**k:BadResponse())
    with pytest.raises(ModelRuntimeError) as info:
        model.complete([{"role":"user","content":"hi"}],[])
    assert info.value.stop_reason=="model_error"
    assert info.value.diagnostic_code=="MODEL_HTTP_5XX"


def test_model_error_diagnostic_code_cannot_be_raw_data():
    with pytest.raises(ValueError):
        ModelRuntimeError("model_error","opaque","MODEL_PRIVATE_USER_TOKEN")
