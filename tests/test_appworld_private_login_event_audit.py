"""Synthetic protected-runner login call-id/status/contract verification."""
from types import SimpleNamespace
from gaia_small_agent.agent.types import TraceEvent
from gaia_small_agent.tools.appworld_private_login_event_audit import source_only_login_status

def setup():
    return SimpleNamespace(task=SimpleNamespace(api_docs={
        "private_app":{"login":{"parameters":[
            {"name":"email","required":True,"type":"string"},
            {"name":"password","required":True,"type":"string"},
            {"name":"tenant","required":False,"type":"string"},
        ]}}}))

def call(step,id,api,args,app="private_app"):
    return TraceEvent("tool_call",step,{"id":id,"name":"appworld_call",
        "arguments":{"app":app,"api":api,"arguments":args}})

def result(step,id,ok,code="HTTP_ERROR"):
    return TraceEvent("tool_result",step,{"id":id,"name":"appworld_call",
        "ok":ok,"error_code":code,"content":"HTTP 401"})

def test_login_status_422_exact_and_public_missing_fields():
    token="SECRET_PASSWORD_ABCDEFG"
    trace=[
        call(1,"first","private",{}),
        result(1,"first",False),
        call(3,"second","login",{"email":"user@example.com"}),
        result(3,"second",False),
        call(4,"third","login",{"email":"user@example.com","password":token}),
        result(4,"third",False),
    ]
    statuses=[{"model_step":1,"http_status":401,"api_call_index":1},
              {"model_step":3,"http_status":422,"api_call_index":2},
              {"model_step":4,"http_status":401,"api_call_index":3}]
    a=source_only_login_status(trace,setup(),statuses,"Task no credentials")
    assert a["first401_step"]==1
    assert a["login_attempt_count"]==2
    assert a["login_http_status_counts"]=={"401":1,"422":1}
    assert a["per_login_safe"][0]["missing_or_empty_required_field_count"]==1
    assert a["per_login_safe"][1]["missing_or_empty_required_field_count"]==0
    assert a["ambiguous_login_http_link_count"]==0
    assert token not in repr(a) and "password" not in repr(a)
    assert "private_app" not in repr(a)


def test_multiple_same_step_http_events_must_abstain_from_exact_attribution():
    trace=[call(2,"x","login",{}),result(2,"x",False),
           call(2,"y","private",{}),result(2,"y",False)]
    errors=[{"model_step":2,"http_status":401,"api_call_index":1},
            {"model_step":2,"http_status":422,"api_call_index":2}]
    a=source_only_login_status(trace,setup(),errors,"task")
    assert a["login_attempt_count"]==1
    assert a["ambiguous_login_http_link_count"]==1
    assert a["per_login_safe"][0]["http_status"] is None


def test_single_status_two_tool_errors_also_fails_closed():
    trace=[call(2,"x","login",{}),result(2,"x",False),
           call(2,"y","private",{}),result(2,"y",False)]
    errors=[{"model_step":2,"http_status":401,"api_call_index":1}]
    a=source_only_login_status(trace,setup(),errors,"task")
    assert a["ambiguous_login_http_link_count"]==1


def test_type_mismatch_is_public_contract_not_credentials_validity():
    trace=[call(1,"a","login",{"email":123,"password":"unknown secret"}),
           result(1,"a",False)]
    errors=[{"model_step":1,"http_status":422,"api_call_index":1}]
    a=source_only_login_status(trace,setup(),errors,"task")
    assert a["per_login_safe"][0]["required_field_public_type_mismatch_count"]==1
    assert a["per_login_safe"][0]["missing_or_empty_required_field_count"]==0
    assert a["per_login_safe"][0]["credential_validity_unverified"]


def test_model_task_exact_occurrence_only_boolean():
    secret="PRIVATE_USER_TASK_CRED"
    trace=[call(1,"a","login",{"email":"mail@example.com","password":secret}),
           result(1,"a",False)]
    errors=[{"model_step":1,"http_status":401,"api_call_index":1}]
    a=source_only_login_status(trace,setup(),errors,secret)
    assert a["per_login_safe"][0]["supplied_required_values_matched_task_text_count"]==1
    assert secret not in repr(a)
