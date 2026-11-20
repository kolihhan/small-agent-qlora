"""AppWorld tool-bridge observability contract; no real task/source data."""
from pathlib import Path
from types import SimpleNamespace

from gaia_small_agent.tools.appworld_public_contract_audit import (
    public_doc_visibility_audit, public_http_status, public_422_probe_category,
)
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool, AppWorldDocsTool


class FakeRequester:
    def __init__(self, errors):
        self.errors = list(errors)
        self.index = 0

    def request(self, **kwargs):
        e = self.errors[self.index]
        self.index += 1
        raise Exception(e)


def fake_world(errors=None, params=None):
    if params is None:
        params = [{"name": "start_date", "type": "string",
                   "description": "Date", "required": True,
                   "default": None, "constraints": ["ISO date"]}]
    doc = {"description": "Synthetic public API", "method": "POST",
           "path": "/calendar/events", "parameters": params}
    return SimpleNamespace(
        task=SimpleNamespace(allowed_apps=["calendar"],
                             api_docs={"calendar": {"create_event": doc}}),
        requester=FakeRequester(errors or []))


def call_args():
    return {"app": "calendar", "api": "create_event", "arguments": {}}


def test_exact_upstream_http_format_and_fail_closed_categories():
    good = ('Response status code is 422:\n'
            '{"detail":[{"loc":["body","start_date"],"type":"missing",'
            '"input":"DO_NOT_DISCLOSE_PRIVATE_SECRET"}]}')
    assert public_http_status(good) == 422
    assert public_http_status('Response status code is 403:\nprivate') == 403
    assert public_http_status("PRIVATE_BAD_FORMAT 422") is None
    assert public_422_probe_category(good, fake_world().task.api_docs["calendar"]["create_event"]) == "PUBLIC_FIELD_HINT_AVAILABLE"
    assert public_422_probe_category('Response status code is 422:\nother-private',
                                      fake_world().task.api_docs["calendar"]["create_event"]) == "NO_DOCUMENTED_FIELD_HINT"
    assert public_422_probe_category('Response status code is 409:\nprivate',
                                      fake_world().task.api_docs["calendar"]["create_event"]) == "NOT_422"


def test_model_visible_text_unchanged_and_only_status_counters_saved():
    secret = "DO_NOT_DISCLOSE_PRIVATE_SECRET"
    e422 = ('Response status code is 422:\n'
            '{"detail":[{"loc":["body","start_date"],"type":"missing",'
            '"input":"' + secret + '"}]}')
    world = fake_world([e422, "Response status code is 403:\n" + secret])
    tool = AppWorldCallTool(world, max_calls=4, report_http_failures=True)
    first = tool.run(call_args(), Path("."))
    second = tool.run(call_args(), Path("."))
    assert first.content == (
        "AppWorld API rejected this request (HTTP 422). "
        "Check the API documentation and parameters.")
    assert second.content == (
        "AppWorld API rejected this request (HTTP 403). "
        "Check the API documentation and parameters.")
    assert tool.http_status_counts == {"422": 1, "403": 1}
    assert tool.http_422_probe_outcomes == {"PUBLIC_FIELD_HINT_AVAILABLE": 1}
    assert secret not in (first.content + second.content + repr(tool.http_status_counts)
                          + repr(tool.http_422_probe_outcomes))


def test_public_doc_omission_is_counted_but_model_presentation_stays_frozen():
    params = [{"name": "p" + str(i), "type": "string", "required": i == 0,
               "description": "A field", "default": None,
               "constraints": ["synthetic"]} for i in range(25)]
    world = fake_world(params=params)
    audit = public_doc_visibility_audit(world)
    assert audit["public_apis"] == 1
    assert audit["public_parameters"] == 25
    assert audit["parameters_hidden_by_first24_cap"] == 1
    assert audit["parameter_default_field_omitted"] == 24
    assert audit["parameter_constraints_field_omitted"] == 24
    assert audit.get("non_null_default_values_omitted", 0) == 0
    assert audit["nonempty_public_constraints_omitted"] == 24
    world.task.api_docs["calendar"]["create_event"]["response_schemas"] = {"success": "public"}
    assert public_doc_visibility_audit(world)["public_response_schema_examples_omitted"] == 1
    shown = AppWorldDocsTool(world).run(
        {"app": "calendar", "api": "create_event"}, Path("."))
    assert shown.ok
    assert "constraints" not in shown.content and "default" not in shown.content
    assert '"p23"' in shown.content and '"p24"' not in shown.content
