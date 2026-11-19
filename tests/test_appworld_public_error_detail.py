"""Public-feedback hypothesis tests. No AppWorld install or protected data."""
import json
from pathlib import Path
from types import SimpleNamespace

from gaia_small_agent.tools.appworld_public_error_detail import documented_parameter_hint
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool


DOC = {"parameters": [{"name": "start_date", "type": "string", "required": True}]}
PAYLOAD = {"detail": [
    {"loc": ["body", "start_date"], "msg": "Field required",
     "type": "missing", "input": "PRIVATE_SECRET_MUST_NOT_LEAK"}]}
ERR = "Response status code is 422:\n" + json.dumps(PAYLOAD)


class FailRequester:
    def __init__(self, error=ERR):
        self.calls = 0
        self.error = error

    def request(self, **kwargs):
        self.calls += 1
        raise Exception(self.error)


def fake_world(error=ERR):
    docs = {"calendar": {"create_event": DOC}}
    return SimpleNamespace(
        task=SimpleNamespace(allowed_apps=["calendar"], api_docs=docs),
        requester=FailRequester(error))


def args():
    return {"app": "calendar", "api": "create_event", "arguments": {}}


def test_exact_unchanged_baseline_observation_when_opt_in_disabled():
    w = fake_world()
    result = AppWorldCallTool(w, report_http_failures=True).run(args(), Path("."))
    assert not result.ok and result.error_code == "HTTP_ERROR"
    assert result.content == (
        "AppWorld API rejected this request (HTTP 422). "
        "Check the API documentation and parameters.")
    assert w.requester.calls == 1


def test_safe_field_only_is_one_observation_delta_and_one_real_api_call():
    w = fake_world()
    result = AppWorldCallTool(
        w, report_http_failures=True, public_422_field_hint=True
    ).run(args(), Path("."))
    assert w.requester.calls == 1
    assert result.error_code == "HTTP_ERROR"
    assert result.content.startswith("AppWorld API rejected this request (HTTP 422).")
    assert "required documented parameter 'start_date' is missing" in result.content
    for private in ("PRIVATE_SECRET", "input", "Field required", "body"):
        assert private not in result.content
    assert "start_date" in result.bounded_observation(2000)[0]


def test_doc_allowlist_refuses_undocumented_parameters_and_raw_error_text():
    error = "Response status code is 422:\n" + json.dumps({
        "detail": [{"loc": ["body", "admin_secret"], "type": "missing",
                    "msg": "PRIVATE_ORACLE"}]})
    assert documented_parameter_hint(error, DOC) is None
    result = AppWorldCallTool(
        fake_world(error), report_http_failures=True, public_422_field_hint=True
    ).run(args(), Path("."))
    assert "admin_secret" not in result.content
    assert "PRIVATE_ORACLE" not in result.content


def test_non_422_or_unstructured_failure_cannot_leak_untrusted_response():
    for error in (
        'Response status code is 403:\n{"detail":"secret"}',
        "Response status code is 422:\nnot a json object secret",
        'Response status code is 422:\n{"detail":"private value"}',
    ):
        assert documented_parameter_hint(error, DOC) is None
        result = AppWorldCallTool(
            fake_world(error), report_http_failures=True, public_422_field_hint=True
        ).run(args(), Path("."))
        assert "secret" not in result.content
