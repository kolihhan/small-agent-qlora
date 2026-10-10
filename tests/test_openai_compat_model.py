import json

import pytest
import requests

from gaia_small_agent.agent.types import ModelRuntimeError


def _response(payload):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    return FakeResponse()


def test_openai_compat_model_posts_tools_and_parses_tool_calls(monkeypatch):
    from gaia_small_agent.model.openai_compat import OpenAICompatModel

    seen = {}

    def fake_post(url, *, json, timeout):
        seen.update(url=url, payload=json, timeout=timeout)
        return _response({
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"qwen"}',
                        },
                    }],
                }
            }]
        })

    monkeypatch.setattr(requests, "post", fake_post)
    model = OpenAICompatModel(model="qwen3.5-9b", base_url="http://127.0.0.1:8080", max_new_tokens=321, timeout_s=17)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    turn = model.complete([{"role": "user", "content": "find it"}], tools)

    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["timeout"] == 17
    assert seen["payload"]["model"] == "qwen3.5-9b"
    assert seen["payload"]["max_tokens"] == 321
    assert seen["payload"]["temperature"] == 0
    assert seen["payload"]["tools"] == tools
    assert seen["payload"]["tool_choice"] == "auto"
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call-1"
    assert turn.tool_calls[0].name == "search"
    assert turn.tool_calls[0].arguments == {"query": "qwen"}


def test_openai_compat_model_accepts_dict_arguments_and_final_text(monkeypatch):
    from gaia_small_agent.model.openai_compat import OpenAICompatModel

    replies = iter([
        _response({
            "choices": [{"message": {"content": "", "tool_calls": [{
                "id": "call-2",
                "function": {"name": "read", "arguments": {"source": "a.txt"}},
            }]}}]
        }),
        _response({"choices": [{"message": {"content": "final answer"}}]}),
    ])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: next(replies))
    model = OpenAICompatModel(model="qwen3.5-9b")

    call_turn = model.complete([{"role": "user", "content": "read"}], [])
    final_turn = model.complete([{"role": "user", "content": "answer"}], [])

    assert call_turn.tool_calls[0].arguments == {"source": "a.txt"}
    assert final_turn.content == "final answer"
    assert final_turn.tool_calls == []


def test_openai_compat_model_maps_timeout(monkeypatch):
    from gaia_small_agent.model.openai_compat import OpenAICompatModel

    def fail(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(ModelRuntimeError) as exc:
        OpenAICompatModel(model="qwen3.5-9b", timeout_s=2).complete([{"role": "user", "content": "x"}], [])
    assert exc.value.stop_reason == "model_timeout"


def test_openai_compat_model_rejects_bad_tool_arguments(monkeypatch):
    from gaia_small_agent.model.openai_compat import OpenAICompatModel

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _response({
        "choices": [{"message": {"content": "", "tool_calls": [{
            "id": "bad",
            "function": {"name": "search", "arguments": "not-json"},
        }]}}]
    }))
    with pytest.raises(ModelRuntimeError) as exc:
        OpenAICompatModel(model="qwen3.5-9b").complete([{"role": "user", "content": "x"}], [])
    assert exc.value.stop_reason == "model_error"
