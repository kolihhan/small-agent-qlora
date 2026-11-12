"""Best-of-N tool rollouts require actual stochastic decoding, not repeated greedy."""
from __future__ import annotations

import pytest
from gaia_small_agent.model.ollama import OllamaModel


def test_default_remains_deterministic_and_sampled_temperature_reaches_ollama(monkeypatch):
    observed = []

    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"message": {"role": "assistant", "content": "done"}}

    def fake_post(url, *, json, timeout):
        observed.append(json)
        return Response()

    monkeypatch.setattr("gaia_small_agent.model.ollama.requests.post", fake_post)
    default = OllamaModel(model="qwen3.5:9b-q4_K_M")
    chosen = OllamaModel(model="qwen3.5:9b-q4_K_M", temperature=0.45)
    assert default.complete([{"role": "user", "content": "question"}], []).content == "done"
    assert chosen.complete([{"role": "user", "content": "question"}], []).content == "done"
    assert observed[0]["options"]["temperature"] == 0
    assert observed[1]["options"]["temperature"] == 0.45
    assert observed[1]["think"] is False


@pytest.mark.parametrize("temperature", [-0.01, 2.01, float("nan")])
def test_invalid_training_sampling_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        OllamaModel(temperature=temperature)
