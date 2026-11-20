from hashlib import sha256

from gaia_small_agent.agent.answer_serializer import SERIALIZER_RULE, serialize_answer
from gaia_small_agent.agent.types import AssistantTurn, ModelRuntimeError


class StubModel:
    def __init__(self, text: str = "", *, fail: bool = False):
        self.text = text
        self.fail = fail
        self.messages = None
        self.tools = None

    def complete(self, messages, tools):
        self.messages, self.tools = messages, tools
        if self.fail:
            raise ModelRuntimeError("model_error", "offline")
        return AssistantTurn(content=self.text)


def test_frozen_serializer_rule_is_unchanged():
    assert sha256(SERIALIZER_RULE.encode()).hexdigest() == "a1f844029179ab91e9ff0ebd0b9b8a6293801c686e4080921edd5b3a82ef60a3"


def test_verbatim_final_extracts_explicit_answer_without_external_tools():
    model = StubModel("Moscow")
    source = "The answer is Moscow, according to the passage."
    result, status = serialize_answer("Which city?", source, model)
    assert (result, status) == ("Moscow", "accepted_verbatim")
    assert model.tools == []
    assert "Moscow" in model.messages[1]["content"]


def test_non_verbatim_candidate_keeps_original():
    original = "I have insufficient evidence to answer."
    result, status = serialize_answer("Which city?", original, StubModel("Moscow"))
    assert (result, status) == (original, "rejected_not_verbatim_or_not_short")


def test_formatter_failure_keeps_original():
    original = "The answer is Moscow."
    assert serialize_answer("Which city?", original, StubModel(fail=True)) == (original, "model_error")


def test_agent_optin_serializes_only_after_completed_original(tmp_path):
    from gaia_small_agent.agent.loop import AgentRuntime

    class TwoTurnModel:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append((messages, tools))
            if len(self.calls) == 1:
                return AssistantTurn(content="The answer is Moscow.", tool_calls=[])
            return AssistantTurn(content="Moscow", tool_calls=[])

    model = TwoTurnModel()
    result = AgentRuntime(model, [], max_steps=2, serialize_final=True).run("Which city?", tmp_path)
    assert result.completed is True
    assert result.answer == "Moscow"
    assert len(model.calls) == 2
    assert model.calls[1][1] == []


def test_default_agent_does_not_call_serializer(tmp_path):
    from gaia_small_agent.agent.loop import AgentRuntime

    model = StubModel("The answer is Moscow.")
    result = AgentRuntime(model, [], max_steps=2).run("Which city?", tmp_path)
    assert result.answer == "The answer is Moscow."
    assert result.completed is True
