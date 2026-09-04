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
