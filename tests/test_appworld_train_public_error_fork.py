"""CPU source-only verification; never loads protected AppWorld task bundles."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gaia_small_agent.agent.types import AssistantTurn, ToolCall
from gaia_small_agent.tools.appworld_tool import AppWorldCallTool
from scripts.appworld_train_public_error_fork import (
    PrefixReplayModel, PublicFieldCallTool, RecordingModel,
    documented_missing_field, FEEDBACK_TEMPLATE, sha,
)


class FakeRequester:
    def request(self, **kwargs):
        if not kwargs.get("query"):
            raise RuntimeError("Response status code is 422: missing required field 'query'")
        return {"items": [1], "ok": True}


def world():
    return SimpleNamespace(
        task=SimpleNamespace(
            allowed_apps=["spotify"],
            api_docs={"spotify": {"search_tracks": {
                "description": "Search tracks",
                "parameters": [
                    {"name": "query", "required": True, "type": "string"},
                    {"name": "offset", "required": False, "type": "integer"}
                ]}},
            ),
        requester=FakeRequester())


def test_missing_field_requires_documented_required_and_explicit_error():
    w = world()
    assert documented_missing_field(
        w, "spotify", "search_tracks", {},
        "Response status code is 422: missing required field 'query'") == "query"
    assert documented_missing_field(
        w, "spotify", "search_tracks", {"query": "x"},
        "Response status code is 422: missing required field 'query'") is None
    assert documented_missing_field(
        w, "spotify", "search_tracks", {},
        "Response status code is 422: invalid token") is None
    assert documented_missing_field(
        w, "spotify", "search_tracks", {},
        "Response status code is 422: missing required field 'password'") is None
    assert documented_missing_field(
        w, "spotify", "search_tracks", {},
        "Response status code is 422: missing required field 'offset'") is None


def test_original_tool_feedback_byte_equivalent_until_single_target_field():
    w = world()
    args = {"app": "spotify", "api": "search_tracks", "arguments": {}}
    original = AppWorldCallTool(w, max_calls=48, report_http_failures=True)
    observed = original.run(args, Path("."))
    probed = PublicFieldCallTool(w, lambda: 1)
    before = probed.run(args, Path("."))
    assert before.bounded_observation(2000) == observed.bounded_observation(2000)
    assert len(probed.eligible) == 1
    event = probed.eligible[0]
    assert event["turn"] == 1 and event["ordinal"] == 1
    assert event["field"] == "query"
    fresh = PublicFieldCallTool(world(), lambda: 1, target=event)
    changed = fresh.run(args, Path("."))
    assert fresh.patched_count == 1
    assert changed.error_code == "HTTP_ERROR"
    assert changed.content.startswith(FEEDBACK_TEMPLATE.format(status="422"))
    assert changed.content[len(before.content):] == (
        " Missing required documented API parameter: query.")
    assert len(fresh.observed) == 1
    assert not fresh.recovered_original
    recovered = fresh.run(
        {"app": "spotify", "api": "search_tracks",
         "arguments": {"query": "Pink Floyd"}}, Path("."))
    assert recovered.ok and fresh.recovered_original


class FakeModel:
    def complete(self, messages, tools):
        return AssistantTurn(
            "thinking", [ToolCall("a", "appworld_call", {
                "app": "spotify", "api": "search_tracks", "arguments": {}})])

    def complete_final(self, messages, tools):
        return AssistantTurn("Done")


def test_prefix_replay_checks_real_message_and_tool_identity():
    a = RecordingModel(FakeModel())
    messages = [{"role": "user", "content": "task"}]
    tools = [{"type": "function", "function": {"name": "appworld_call"}}]
    turn = a.complete(messages, tools)
    playback = PrefixReplayModel(a, 1, FakeModel())
    assert playback.complete(messages, tools) == turn
    assert playback.replayed == 1
    assert playback.complete_final(messages, []) == AssistantTurn("Done")
    assert playback.live == 1
    broken = PrefixReplayModel(a, 1, FakeModel())
    with pytest.raises(ValueError, match="prefix drift"):
        broken.complete([{"role": "user", "content": "different task"}], tools)


def test_hash_contains_no_plaintext_and_no_generic_secret_response():
    secret = "private-excluded-test-token"
    assert secret not in sha(secret)
    assert documented_missing_field(
        world(), "supervisor", "secret", {},
        "Response status code is 400: missing required token") is None
