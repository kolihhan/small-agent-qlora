import pytest
import requests

from gaia_small_agent.agent.types import ModelRuntimeError
from gaia_small_agent.model.ollama import OllamaModel


def test_complete_sends_num_predict(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "done"}}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", fake_post)

    result = OllamaModel(max_new_tokens=128).complete([], [])

    assert result.content == "done"
    assert captured["kwargs"]["json"]["options"] == {"temperature": 0, "num_predict": 128}


def test_complete_disables_thinking_by_default(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "done"}}

    def fake_post(url, **kwargs):
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", fake_post)

    OllamaModel(max_new_tokens=128).complete([], [])

    payload = captured["kwargs"]["json"]
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0, "num_predict": 128}


def test_complete_enables_thinking_when_requested(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "done"}}

    def fake_post(url, **kwargs):
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", fake_post)

    OllamaModel(max_new_tokens=128, enable_thinking=True).complete([], [])

    payload = captured["kwargs"]["json"]
    assert payload["think"] is True
    assert payload["options"] == {"temperature": 0, "num_predict": 128}


def test_complete_maps_timeout_to_model_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", fake_post)

    with pytest.raises(ModelRuntimeError) as exc_info:
        OllamaModel().complete([], [])

    assert exc_info.value.stop_reason == "model_timeout"
    assert "slow" in exc_info.value.message


def test_complete_maps_connection_error_to_model_unavailable(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", fake_post)

    with pytest.raises(ModelRuntimeError) as exc_info:
        OllamaModel().complete([], [])

    assert exc_info.value.stop_reason == "model_unavailable"
    assert "refused" in exc_info.value.message


def test_complete_maps_http_error_to_model_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("500 server error")

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(ModelRuntimeError) as exc_info:
        OllamaModel().complete([], [])

    assert exc_info.value.stop_reason == "model_error"
    assert "500" in exc_info.value.message


def test_complete_rejects_missing_message_payload(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"done": True}

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(ModelRuntimeError) as exc_info:
        OllamaModel().complete([], [])

    assert exc_info.value.stop_reason == "model_error"
    assert "message" in exc_info.value.message.lower()
