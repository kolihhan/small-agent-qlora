"""Synthetic first401 token source and recovery privacy regression tests."""
import json
from gaia_small_agent.agent.types import TraceEvent
from gaia_small_agent.tools.appworld_token_provenance_audit import (
    _json_token_values,token_origin_after_first_401,
)


def call(s,id,tool,app,api,arguments=None):
    return TraceEvent("tool_call",s,{
        "id":id,"name":tool,
        "arguments":{"app":app,"api":api,"arguments":arguments or {}}})


def response(s,id,tool,ok,content,err=None):
    return TraceEvent("tool_result",s,{
        "id":id,"name":tool,"ok":ok,"error_code":err,"content":content})


def run(token,task="",before=(),after=(),step=4):
    trace=[*before,
        call(step,"target","appworld_call","calendar","edit",
             {"access_token":token} if token is not None else {}),
        response(step,"target","appworld_call",False,"HTTP 401","HTTP_ERROR"),
        *after]
    result=token_origin_after_first_401(
        trace,task,[{"http_status":401,"model_step":step,"api_call_index":2}],max_steps=20)
    assert token not in json.dumps(result) if token else True
    assert "calendar" not in json.dumps(result) and "edit" not in json.dumps(result)
    return result


def test_task_visible_token_exact_occurrence_is_not_validity():
    secret="PRIVATE_USER_TASK_TOKEN_123"
    a=run(secret,"Please use token="+secret)
    assert a["source_class"]=="MODEL_VISIBLE_TASK_EXACT_OCCURRENCE"
    assert a["exact_token_in_model_visible_task"]
    assert not a["exact_token_in_prior_successful_API_result"]
    assert not a["post401_login_attempted"]


def test_successful_prior_login_token_field_exact_match():
    secret="PRIVATE_TOOL_TOKEN_456"
    before=[call(2,"l","appworld_call","calendar","login",{"username":"hidden"}),
            response(2,"l","appworld_call",True,json.dumps({"result":{"access_token":secret}}))]
    a=run(secret,before=before)
    assert a["source_class"]=="PRIOR_SUCCESSFUL_API_TOKEN_FIELD"
    assert a["exact_token_in_prior_login_result"]


def test_successful_opaque_response_and_public_docs_do_not_authenticate():
    secret="PRIVATE_TOKEN_FROM_NO_SOURCE_789"
    before=[call(2,"l","appworld_docs","calendar","edit"),
            response(2,"l","appworld_docs",True,json.dumps({"access_token":secret})),
            call(3,"s","appworld_call","calendar","status"),
            response(3,"s","appworld_call",False,json.dumps({"access_token":secret}),"HTTP_ERROR")]
    a=run(secret,before=before)
    assert a["source_class"]=="NO_IDENTIFIABLE_MODEL_VISIBLE_SOURCE"


def test_unknown_source_does_not_mean_fabricated():
    a=run("A_LONG_UNKNOWN_TOKEN_X")
    assert a["source_class"]=="NO_IDENTIFIABLE_MODEL_VISIBLE_SOURCE"


def test_after_first401_login_attempt_and_success_are_accounted():
    secret="PRIVATE_TOKEN_FROM_TASK"
    later=[call(5,"l","appworld_call","calendar","login",{"password":secret}),
           response(5,"l","appworld_call",True,json.dumps({"access_token":"DIFFERENT_PRIVATE_TOKEN"}))]
    a=run(secret,after=later)
    assert a["post401_login_attempted"] and a["post401_login_succeeded"]
    assert a["post401_login_attempt_count"]==1


def test_after_first401_login_failure_not_marked_success():
    secret="LONG_TOKEN_XYZ"
    later=[call(5,"l","appworld_call","calendar","login"),
           response(5,"l","appworld_call",False,"HTTP 401","HTTP_ERROR")]
    a=run(secret,after=later)
    assert a["post401_login_attempted"] and not a["post401_login_succeeded"]


def test_short_token_and_missing_token_are_not_provenance_claims():
    assert run("ab")["source_class"]=="SHORT_TOKEN_PROVENANCE_UNDETERMINED"
    assert run(None)["source_class"]=="NO_TOKEN_SUPPLIED"


def test_json_value_parser_ignores_payload_free_text_and_other_fields():
    token="SECRET_VALUE_ABC"
    assert _json_token_values(json.dumps({"description":token,"token":token}))==[]
    assert _json_token_values(json.dumps({"result":{"access_token":token}}))==[token]
    assert _json_token_values("API response "+token)==[]
