from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn


class TextualToolCallModel:
    def complete(self, messages, tools):
        return AssistantTurn(
            content=(
                "<tool_call>\n"
                "<function=search>\n"
                "<parameter=query>Roy White 1977 Yankees walks</parameter>\n"
                "</function>\n"
                "</tool_call>"
            ),
            tool_calls=[],
        )


def test_textual_tool_call_markup_is_not_counted_as_completed_final(tmp_path):
    result = AgentRuntime(TextualToolCallModel(), [], max_steps=1).run("answer", tmp_path)

    assert result.completed is False
    assert result.answer == ""
    assert result.stop_reason == "invalid_final"
    assert result.trace[-1].kind == "invalid_final"
