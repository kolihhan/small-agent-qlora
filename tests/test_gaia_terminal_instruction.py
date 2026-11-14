from gaia_small_agent.agent.types import AssistantTurn, ModelRuntimeError, ToolCall
from scripts.gaia_terminal_instruction import final_outcome, terminal_messages


def test_terminal_instruction_preserves_exact_tool_prefix_and_source_messages():
    original = [
        {"role": "system", "content": "Existing policy."},
        {"role": "user", "content": "Question"},
        {"role": "tool", "content": "Observation", "tool_call_id": "id-1"},
    ]
    changed = terminal_messages(original)
    assert changed[:-1] == original
    assert changed[-1]["role"] == "user"
    assert "No more tools" in changed[-1]["content"]
    assert original[0]["content"] == "Existing policy."


def test_terminal_scoring_rejects_tool_calls_even_if_text_contains_an_answer():
    cases = [
        (AssistantTurn("42"), ("42", "final")),
        (AssistantTurn(" <tool_call>raw</tool_call>"), ("", "invalid_final")),
        (AssistantTurn("42", [ToolCall("id", "search", {})]), ("", "max_steps")),
        (AssistantTurn(" "), ("", "empty_final")),
        (ModelRuntimeError("model_timeout", "ignored"), ("", "model_timeout")),
    ]
    for turn, expected in cases:
        assert final_outcome(turn) == expected
