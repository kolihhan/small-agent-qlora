"""Source-only tests; never require protected AppWorld tasks."""
from types import SimpleNamespace
from pathlib import Path

from gaia_small_agent.agent.types import TraceEvent
from scripts.appworld_train_denied_app_list_prefix_fork import (
    OneShotAllowedAppsDocsTool,
    first_denied_docs_error,
    prior_public_app_list_visible,
    safe_hint,
)


def fake_world():
    task = SimpleNamespace(allowed_apps=["email", "calendar", "api_docs", "admin"])
    return SimpleNamespace(task=task)


def test_denied_app_hint_only_on_one_target_error():
    tool = OneShotAllowedAppsDocsTool(fake_world(), target_call_index=1)
    first = tool.run({"app": "not_allowed"}, Path("."))
    assert not first.ok
    assert first.error_code == "APP_DENIED"
    assert first.content == "App unavailable for this task. " + safe_hint(fake_world())
    assert '"calendar"' in first.content and '"email"' in first.content
    assert '"admin"' not in first.content
    next_call = tool.run({"app": "also_denied"}, Path("."))
    assert next_call.content == "App unavailable for this task."
    assert tool.hint_delivered


def test_reject_if_first_obstruction_is_different_error():
    trace = [
        TraceEvent("tool_call", 1, {"id": "1", "name": "appworld_call"}),
        TraceEvent("tool_result", 1, {
            "id": "1", "name": "appworld_call", "ok": False,
            "error_code": "HTTP_ERROR", "content": "HTTP 404",
        }),
    ]
    assert first_denied_docs_error(trace)[1] == "FIRST_OBSTRUCTION_NOT_DOCS_APP_DENIED"


def test_candidate_when_denied_before_any_visible_public_list():
    trace = [
        TraceEvent("tool_call", 1, {"id": "1", "name": "appworld_docs"}),
        TraceEvent("tool_result", 1, {
            "id": "1", "name": "appworld_docs", "ok": False,
            "error_code": "APP_DENIED", "content": "App unavailable",
        }),
    ]
    assert first_denied_docs_error(trace) == (
        {"step": 1, "docs_call_index": 1}, "ELIGIBLE_FIRST_DENIED_APP"
    )


def test_existing_public_list_blocks_treatment_fork():
    trace = [
        TraceEvent("tool_call", 1, {"id": "list", "name": "appworld_docs"}),
        TraceEvent("tool_result", 1, {
            "id": "list", "name": "appworld_docs", "ok": True,
            "error_code": None, "content": '{"task_apps": ["email"]}',
        }),
        TraceEvent("tool_call", 2, {"id": "denied", "name": "appworld_docs"}),
        TraceEvent("tool_result", 2, {
            "id": "denied", "name": "appworld_docs", "ok": False,
            "error_code": "APP_DENIED", "content": "App unavailable",
        }),
    ]
    assert prior_public_app_list_visible(trace, 2, "denied")
    assert first_denied_docs_error(trace)[1] == "PUBLIC_APP_LIST_ALREADY_SEEN_BEFORE_DENIAL"
