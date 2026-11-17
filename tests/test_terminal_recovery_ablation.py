"""Controlled terminal-recovery ablation; default runner behavior must not change."""
from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ToolCall


class FinalToolMarkupModel:
    def __init__(self, recovery="42"):
        self.calls = []
        self.recovery = recovery

    def complete(self, messages, tools):
        self.calls.append(("auto", len(tools)))
        return AssistantTurn("<tool_call>search for more</tool_call>", [])

    def complete_final(self, messages, tools):
        self.calls.append(("final", len(tools), messages[-1]["role"]))
        # First forced-final response represents the original failure. The
        # one additional final call is the *only* opportunity for recovery.
        if len(self.calls) == 1:
            return AssistantTurn("<tool_call>another query</tool_call>", [])
        return AssistantTurn(self.recovery, [])


def test_default_loop_retains_exact_prior_invalid_final_behavior(tmp_path):
    model = FinalToolMarkupModel()
    result = AgentRuntime(model, [], max_steps=1).run("What is 6 times 7?", tmp_path)
    assert not result.completed
    assert result.stop_reason == "invalid_final"
    assert result.metrics.terminal_recovery_calls == 0
    assert len(model.calls) == 1


def test_opt_in_terminal_recovery_emits_observed_answer_and_counts_call(tmp_path):
    model = FinalToolMarkupModel()
    result = AgentRuntime(model, [], max_steps=1, terminal_recovery=True).run(
        "What is 6 times 7?", tmp_path
    )
    assert result.completed
    assert result.answer == "42"
    assert result.stop_reason == "terminal_recovery_final"
    assert result.metrics.steps == 1
    assert result.metrics.terminal_recovery_calls == 1
    assert model.calls == [("final", 0, "user"), ("final", 0, "user")]
    assert [e.kind for e in result.trace] == [
        "terminal_recovery_attempt", "terminal_recovery_final"
    ]


def test_invalid_recovery_fails_closed_and_does_not_claim_correctness(tmp_path):
    model = FinalToolMarkupModel(recovery="<tool_call>still searching</tool_call>")
    result = AgentRuntime(model, [], max_steps=1, terminal_recovery=True).run("x", tmp_path)
    assert not result.completed
    assert result.answer == ""
    assert result.stop_reason == "invalid_final"
    assert result.metrics.terminal_recovery_calls == 1
    assert result.trace[-1].kind == "terminal_recovery_failed"


def test_native_tool_call_at_last_step_does_not_execute_and_can_recover(tmp_path):
    class NativeModel:
        def __init__(self):
            self.recovery_calls = 0

        def complete_final(self, messages, tools):
            self.recovery_calls += 1
            if self.recovery_calls == 1:
                return AssistantTurn("", [ToolCall("c", "fake", {})])
            return AssistantTurn("verified", [])

        def complete(self, messages, tools):
            raise AssertionError("unexpected non-terminal model call")

    result = AgentRuntime(NativeModel(), [], max_steps=1, terminal_recovery=True).run("x", tmp_path)
    assert result.completed and result.answer == "verified"
    assert result.metrics.tool_calls == 0
    assert result.metrics.terminal_recovery_calls == 1


def test_early_invalid_markup_not_recovered(tmp_path):
    class EarlyModel:
        def complete(self, messages, tools):
            return AssistantTurn("<tool_call>still searching</tool_call>", [])

    result = AgentRuntime(EarlyModel(), [], max_steps=2, terminal_recovery=True).run("x", tmp_path)
    assert not result.completed and result.stop_reason == "invalid_final"
    assert result.metrics.terminal_recovery_calls == 0
