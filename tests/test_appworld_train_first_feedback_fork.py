"""Prefix-fork invariants only. Never loads protected AppWorld data."""
import pytest
from gaia_small_agent.agent.types import AssistantTurn
from scripts.appworld_train_first_feedback_fork import (
    RecordingModel, ExactPrefixModel, digest, task_roster,
)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        return AssistantTurn(content="answer")

    def complete_final(self, messages, tools):
        return self.complete(messages, tools)


def test_frozen_prefix_yields_original_turn_without_second_model_call():
    fake = FakeModel()
    recorder = RecordingModel(fake)
    old = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    recorder.complete(old, [])
    first_feedback = old + [{"role": "tool", "content": "ToolError[HTTP_ERROR]: HTTP 422"}]
    recorder.complete(first_feedback, [])
    assert fake.calls == 2
    fork = ExactPrefixModel(fake, recorder.records, 1, "documented field")
    assert fork.complete(old, []).content == "answer"
    assert fake.calls == 2
    new_feedback = old + [{"role": "tool", "content":
        "ToolError[HTTP_ERROR]: HTTP 422 documented field"}]
    assert fork.complete(new_feedback, []).content == "answer"
    assert fork.one_observation_delta_verified
    assert fork.verified_original_prefix_calls == 1
    assert fork.new_model_calls == 1


def test_fork_fails_closed_if_any_prior_observation_changes():
    recorder = RecordingModel(FakeModel())
    original = [{"role": "user", "content": "unchanged"}]
    recorder.complete(original, [])
    recorder.complete(original + [{"role": "tool", "content": "A"}], [])
    fork = ExactPrefixModel(FakeModel(), recorder.records, 1, "X")
    with pytest.raises(ValueError, match="prior AppWorld state"):
        fork.complete([{"role": "user", "content": "changed"}], [])


def test_fork_rejects_multiple_feedback_or_nonfeedback_differences():
    recorder = RecordingModel(FakeModel())
    initial = [{"role": "user", "content": "u"}]
    original = initial + [{"role": "tool", "content": "failure"}]
    recorder.complete(initial, [])
    recorder.complete(original, [])
    fork = ExactPrefixModel(FakeModel(), recorder.records, 1, " hint")
    fork.complete(initial, [])
    with pytest.raises(ValueError, match="more than one"):
        fork.complete(initial + [{"role": "tool", "content": "failure hint"},
                                 {"role": "tool", "content": "extra"}], [])
    assert digest(original) != digest(initial)
