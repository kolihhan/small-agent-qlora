"""Public-only HTTP401 source-contract feedback; fail-closed with synthetic data."""
from pathlib import Path
from types import SimpleNamespace

from gaia_small_agent.tools.appworld_public_auth_observation import documented_auth_observation
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool


DOC={"parameters":[{"name":"access_token","required":True,"description":"Real bearer token"}]}
ERROR='Response status code is 401:\n{"message":"PRIVATE_ACCESS_TOKEN_AND_PASSWORD"}'


class FailRequester:
    def request(self,**kwargs):
        raise Exception(ERROR)


def world(doc=DOC):
    return SimpleNamespace(
        task=SimpleNamespace(allowed_apps=["calendar"],
            api_docs={"calendar":{"edit_event":doc,"login":{"parameters":[]}}}),
        requester=FailRequester())


def args(value=None):
    return {"app":"calendar","api":"edit_event",
            "arguments":{} if value is None else {"access_token":value}}


def test_only_public_parameter_contract_can_produce_auth_hint():
    assert documented_auth_observation(403,DOC,{},True) is None
    assert documented_auth_observation(401,{"parameters":[]},{},True) is None
    assert documented_auth_observation(401,DOC,{},True)[0]=="REQUIRED_ACCESS_TOKEN_ABSENT"
    assert documented_auth_observation(401,DOC,{"access_token":"PRIVATE"},True)[0]=="REQUIRED_ACCESS_TOKEN_PRESENT_REJECTED"
    assert documented_auth_observation(401,{"parameters":[{"name":"access_token","required":False}]},
                                       {"access_token":"PRIVATE"},True)[0]=="OPTIONAL_TOKEN_SUPPLIED_REJECTED"
    assert documented_auth_observation(401,{"parameters":[{"name":"access_token","required":False}]},
                                       {},True) is None


def test_current_model_visible_observation_is_unchanged_by_default():
    api=AppWorldCallTool(world(),report_http_failures=True,capture_401_auth_hint=True)
    api.model_step_counter=lambda:2
    out=api.run(args(),Path("."))
    assert out.error_code=="HTTP_ERROR"
    assert out.content=="AppWorld API rejected this request (HTTP 401). Check the API documentation and parameters."
    assert api.http_status_counts=={"401":1}
    assert api.public_401_auth_candidates[0]["model_step"]==2
    assert api.public_401_auth_candidates[0]["safe_category"]=="REQUIRED_ACCESS_TOKEN_ABSENT"
    assert api.public_401_auth_candidates[0]["api_call_index"]==1
    assert "PRIVATE" not in repr(api.public_401_auth_candidates)


def test_one_opt_in_append_is_only_safe_public_auth_hint():
    api=AppWorldCallTool(world(),report_http_failures=True,public_401_auth_hint=True)
    out=api.run(args("PRIVATE_SECRET_TOKEN"),Path("."))
    assert out.content.startswith("AppWorld API rejected this request (HTTP 401).")
    assert "supplied access_token was rejected" in out.content
    assert "PRIVATE_SECRET_TOKEN" not in out.content
    assert "PRIVATE_ACCESS_TOKEN_AND_PASSWORD" not in out.content
    assert api.auth_401_public_category_counts=={"REQUIRED_ACCESS_TOKEN_PRESENT_REJECTED":1}


def test_optional_public_token_contract_only_recommends_public_path_when_supplied():
    doc={"parameters":[{"name":"access_token","required":False}]}
    api=AppWorldCallTool(world(doc),report_http_failures=True,public_401_auth_hint=True)
    out=api.run(args("PRIVATE"),Path("."))
    assert "public version can be accessed without a token" in out.content
