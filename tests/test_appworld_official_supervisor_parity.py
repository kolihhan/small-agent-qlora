"""Official Supervisor Parity: strict opt-in; no real credentials or model calls.

Synthetic credentials remain local to fake requester and the model-facing tool
observation, never in telemetry, CI assertion messages, or evidence artifacts.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gaia_small_agent.tools.appworld_tool import (
    AppWorldCallTool, AppWorldDocsTool, allowed_public_api,
)
from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS

PRIVATE_FAKE_PASSWORD = "SYNTHETIC_NOT_REAL_PASSWORD_TEST_ONLY"
PRIVATE_FAKE_TOKEN = "SYNTHETIC_NOT_REAL_TOKEN_TEST_ONLY"


class FakeRequester:
    def __init__(self):
        self.calls=[]

    def request(self,**kwargs):
        self.calls.append(kwargs)
        target=kwargs["_api_name"]
        if target=="show_profile":
            return {"email":"synthetic@example.invalid","first_name":"Fake","last_name":"Supervisor"}
        if target=="show_account_passwords":
            return {"account_passwords":[{"account_name":"calendar","password":PRIVATE_FAKE_PASSWORD}]}
        if target=="login":
            return {"access_token":PRIVATE_FAKE_TOKEN}
        return {"success":True}


def public_doc(parameters=(),response=None):
    return {
        "description":"Public endpoint documentation",
        "method":"GET","path":"/synthetic",
        "parameters":list(parameters),
        "response_schemas":response or {"200":{"type":"object"}},
    }


def fake_world(*,allowed_supervisor=True,has_passwords=True):
    supervisor={
        "complete_task":public_doc(),
        "show_profile":public_doc(response={"success":{"email":"FAKE_DOC_EMAIL_DO_NOT_TREAT_AS_REAL","first_name":"Fake"}}),
        "show_account_passwords":public_doc(response={"success":[{"account_name":"calendar","password":"FAKE_DOC_PASSWORD_DO_NOT_TREAT_AS_REAL"}]}),
        "unsafe_admin_reset":public_doc(),
    }
    if not has_passwords:
        supervisor.pop("show_account_passwords")
    docs={
        "supervisor":supervisor,
        "calendar":{
            "login":public_doc([
                {"name":"username","type":"string","required":True},
                {"name":"password","type":"string","required":True},
            ]),
            "read_data":public_doc([{"name":"access_token","type":"string","required":True}]),
        },
    }
    return SimpleNamespace(task=SimpleNamespace(
        allowed_apps=(["supervisor"] if allowed_supervisor else [])+["calendar"],
        api_docs=docs,
        instruction="Create one new calendar event.",
        supervisor=SimpleNamespace(email="synthetic@example.invalid"),
    ),requester=FakeRequester())


def invoke(tool,api,**parameters):
    return tool.run({"app":"supervisor","api":api,"arguments":parameters},Path("."))


def test_default_off_original_supervisor_semantics_and_descriptions():
    w=fake_world()
    docs=AppWorldDocsTool(w)
    calls=AppWorldCallTool(w,max_calls=4)
    assert docs.definition()["function"]["description"] == AppWorldDocsTool.description
    assert docs.run({"app":"supervisor"},Path(".")).content==json.dumps({
        "apis":["complete_task"],"note":"Use after task work is complete."
    })
    assert "show_profile" not in docs.run({"app":"supervisor"},Path(".")).content
    assert "show_account_passwords" not in docs.run({"app":"supervisor"},Path(".")).content
    for api in ("show_profile","show_account_passwords","unsafe_admin_reset"):
        assert allowed_public_api(w,"supervisor",api) is False
        assert invoke(calls,api).error_code=="API_DENIED"
    assert calls.definition()["function"]["description"] == AppWorldCallTool.description
    assert "supervisor.show_account_passwords" not in calls.definition()["function"]["description"]
    assert calls.definition()["function"]["parameters"] == AppWorldCallTool.schema
    assert calls.calls==0 and w.requester.calls==[]
    assert allowed_public_api(w,"supervisor","complete_task")


def test_opt_in_parity_exactly_three_documented_supervisor_APIs():
    w=fake_world()
    docs=AppWorldDocsTool(w,official_supervisor_parity=True)
    call=AppWorldCallTool(w,official_supervisor_parity=True)
    listed=json.loads(docs.run({"app":"supervisor"},Path(".")).content)
    assert set(listed["apis"])=={"complete_task","show_profile","show_account_passwords"}
    for api in listed["apis"]:
        assert allowed_public_api(w,"supervisor",api,official_supervisor_parity=True)
        documented=docs.run({"app":"supervisor","api":api},Path("."))
        assert documented.ok
        item=json.loads(documented.content)
        assert item["api"]==api and item["app"]=="supervisor"
        assert "parameters" in item["public_api_doc"]
        assert "response_schemas" not in item["public_api_doc"]
        if api in ("show_profile","show_account_passwords"):
            shape=item["public_api_doc"]["response_shape"]
            assert shape["type"] in ("object","array")
            assert "FAKE_DOC_PASSWORD_DO_NOT_TREAT_AS_REAL" not in documented.content
            assert "FAKE_DOC_EMAIL_DO_NOT_TREAT_AS_REAL" not in documented.content
    password_shape=json.loads(docs.run(
        {"app":"supervisor","api":"show_account_passwords"},Path(".")
    ).content)["public_api_doc"]["response_shape"]
    assert password_shape["type"]=="array"
    assert password_shape["items"]["fields"]["password"]=={"type":"string"}
    call_desc=call.definition()["function"]["description"]
    assert "supervisor.show_profile" in call_desc
    assert "supervisor.show_account_passwords" in call_desc
    assert "appworld_docs" in call_desc
    assert call_desc != AppWorldCallTool.description
    assert call.definition()["function"]["parameters"] == AppWorldCallTool.schema
    assert "Supervisor" in docs.definition()["function"]["description"]
    for api in ("unsafe_admin_reset","__dict__","get_ground_truth","export_database"):
        assert not allowed_public_api(w,"supervisor",api,official_supervisor_parity=True)
        assert docs.run({"app":"supervisor","api":api},Path(".")).error_code=="API_NOT_FOUND"
        assert invoke(call,api).error_code=="API_DENIED"
    assert w.requester.calls==[]
    assert docs.run({"app":"admin","api":"leak"},Path(".")).error_code=="APP_DENIED"
    assert call.run({"app":"admin","api":"leak","arguments":{}},Path(".")).error_code=="API_DENIED"


def test_opt_in_calls_real_original_requester_path_with_only_two_public_lookups():
    w=fake_world()
    tool=AppWorldCallTool(w,max_calls=3,official_supervisor_parity=True)
    p=invoke(tool,"show_profile")
    assert p.ok and json.loads(p.content)["email"]=="synthetic@example.invalid"
    passwords=invoke(tool,"show_account_passwords")
    assert passwords.ok and PRIVATE_FAKE_PASSWORD in passwords.content
    # The private password must be transmitted to the *agent* within its
    # simulator; no telemetry/trace/log/artifact is written by this tool.
    obs,_=passwords.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
    assert PRIVATE_FAKE_PASSWORD in obs
    done=invoke(tool,"complete_task")
    assert done.ok
    assert tool.calls==3 and tool.errors==0 and tool.http_errors==0
    assert [x["_api_name"] for x in w.requester.calls]==[
        "show_profile","show_account_passwords","complete_task"]
    for item in w.requester.calls:
        assert item["_app_name"]=="supervisor"
        assert item["show"] is False


def test_fake_account_passwords_cannot_escape_in_safe_metric_or_schema():
    w=fake_world()
    docs=AppWorldDocsTool(w,official_supervisor_parity=True)
    tool=AppWorldCallTool(w,official_supervisor_parity=True)
    assert invoke(tool,"show_account_passwords").ok
    diagnostic={
        "calls":tool.calls,
        "http_errors":tool.http_errors,
        "http_status_counts":tool.http_status_counts,
        "http_error_events":tool.http_error_events,
        "docs":json.loads(docs.run({"app":"supervisor","api":"show_account_passwords"},Path(".")).content),
    }
    serialized=json.dumps(diagnostic,sort_keys=True)
    assert PRIVATE_FAKE_PASSWORD not in serialized
    assert PRIVATE_FAKE_TOKEN not in serialized
    assert "synthetic@example.invalid" not in serialized
    assert tool.http_error_events==[]


def test_parity_requires_actual_public_supervisor_docs_and_allowed_app():
    missing=fake_world(has_passwords=False)
    docs=AppWorldDocsTool(missing,official_supervisor_parity=True)
    calls=AppWorldCallTool(missing,official_supervisor_parity=True)
    assert "show_account_passwords" not in json.loads(docs.run({"app":"supervisor"},Path(".")).content)["apis"]
    assert invoke(calls,"show_account_passwords").error_code=="API_DENIED"
    assert docs.run({"app":"supervisor","api":"show_account_passwords"},Path(".")).error_code=="API_NOT_FOUND"
    noapp=fake_world(allowed_supervisor=False)
    assert not allowed_public_api(noapp,"supervisor","show_profile",official_supervisor_parity=True)
    assert invoke(AppWorldCallTool(noapp,official_supervisor_parity=True),"show_profile").error_code=="API_DENIED"


def test_only_literal_boolean_true_activates_experiment():
    w=fake_world()
    for truthy in (1,"true","yes",{},None):
        # A typo in research config should not silently enable credentials.
        assert not allowed_public_api(w,"supervisor","show_profile",
                                      official_supervisor_parity=truthy)
        assert invoke(AppWorldCallTool(w,official_supervisor_parity=truthy),
                      "show_account_passwords").error_code=="API_DENIED"
        assert json.loads(AppWorldDocsTool(
            w,official_supervisor_parity=truthy
        ).run({"app":"supervisor"},Path(".")).content)["apis"]==["complete_task"]
