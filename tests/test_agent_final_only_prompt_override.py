import pytest

from gaia_small_agent.agent.loop import AgentRuntime, SYSTEM_PROMPT
from gaia_small_agent.agent.types import AssistantTurn


class Stub:
    def __init__(self):
        self.systems=[]

    def complete(self, messages, tools):
        self.systems.append(messages[0]["content"])
        return AssistantTurn(content="one answer",tool_calls=[])


def test_original_agent_system_prompt_unchanged_with_opt_in_override(tmp_path):
    source=Stub()
    result=AgentRuntime(source,[],max_steps=3).run("example",tmp_path/"old")
    assert result.completed is True
    assert source.systems==[SYSTEM_PROMPT]
    treatment=Stub()
    extra=SYSTEM_PROMPT+"\nOn final response, write ONLY the answer, no explanation."
    result=AgentRuntime(treatment,[],max_steps=3,system_prompt=extra).run("example",tmp_path/"new")
    assert result.completed is True
    assert treatment.systems==[extra]


def test_prompt_override_fails_closed(tmp_path):
    for invalid in ("", " ", 19):
        with pytest.raises(ValueError,match="non-empty text"):
            AgentRuntime(Stub(),[],system_prompt=invalid)
