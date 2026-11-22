"""Synthetic protocol recovery controls: no real GAIA questions, inference or gold."""
import pytest

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ModelRuntimeError, ToolCall
from gaia_small_agent.protocol_recovery import ProtocolRepairModel


class MalformedOnceModel:
    def __init__(self):
        self.calls=0
        self.copy=None
    def complete(self, messages, tools):
        self.calls+=1
        if self.calls==1:
            raise ModelRuntimeError("model_error", "Invalid tool arguments JSON: Expecting ',' delimiter")
        self.copy=messages
        return AssistantTurn("",[ToolCall("one","look",{"query":"safe"})])
    def complete_final(self,messages,tools):
        return AssistantTurn("done")


def test_retry_only_invalid_json_with_one_native_tool_call():
    base=MalformedOnceModel()
    wrap=ProtocolRepairModel(base,max_retries=2)
    original=[{"role":"user","content":"What is 2+2?"}]
    ans=wrap.complete(original,[])
    assert len(ans.tool_calls)==1 and ans.tool_calls[0].arguments=={"query":"safe"}
    assert base.calls==2 and wrap.protocol_retries==1 and wrap.protocol_retry_successes==1
    assert len(original)==1
    assert "well-formed" in base.copy[-1]["content"]


def test_non_parser_model_error_does_not_retry():
    class Reject:
        def __init__(self): self.n=0
        def complete(self,messages,tools):
            self.n+=1
            raise ModelRuntimeError("model_error","Server returned invalid JSON")
    b=Reject()
    with pytest.raises(ModelRuntimeError,match="Server returned"):
        ProtocolRepairModel(b).complete([],[])
    assert b.n==1


def test_parser_retries_are_bounded_per_task():
    class AlwaysMalformed:
        def __init__(self): self.n=0
        def complete(self,messages,tools):
            self.n+=1
            raise ModelRuntimeError("model_error","Invalid tool arguments JSON: bad")
    b=AlwaysMalformed()
    model=ProtocolRepairModel(b,max_retries=2)
    for _ in range(3):
        with pytest.raises(ModelRuntimeError,match="Invalid tool arguments JSON"):
            model.complete([],[])
    assert b.n==5 and model.protocol_retries==2


class FinalModel:
    def __init__(self,second): self.calls=0;self.second=second
    def complete_final(self,messages,tools):
        self.calls+=1
        return AssistantTurn("<tool_call>{bad}</tool_call>") if self.calls==1 else self.second
    def complete(self,messages,tools):
        raise AssertionError("max_steps=1 should only request final")


def test_final_only_retry_generates_a_fresh_model_answer_without_tools(tmp_path):
    model=FinalModel(AssistantTurn("42"))
    out=AgentRuntime(model,[],max_steps=1,final_text_retry=True).run("What is 6*7?",tmp_path)
    assert model.calls==2 and out.completed and out.answer=="42"
    assert [e.kind for e in out.trace]==["final_retry_requested","final"]
    assert out.trace[1].data["recovery"]=="second_model_response"


def test_final_retry_never_treats_more_tool_markup_as_an_answer(tmp_path):
    model=FinalModel(AssistantTurn("<tool_call>{more}</tool_call>"))
    out=AgentRuntime(model,[],max_steps=1,final_text_retry=True).run("What is 6*7?",tmp_path)
    assert not out.completed and out.stop_reason=="invalid_final"
    assert model.calls==2


def test_default_protocol_finalization_does_not_retry(tmp_path):
    model=FinalModel(AssistantTurn("42"))
    out=AgentRuntime(model,[],max_steps=1).run("What is 6*7?",tmp_path)
    assert model.calls==1 and out.stop_reason=="invalid_final"


def test_second_final_must_not_contain_native_tool_call(tmp_path):
    model=FinalModel(AssistantTurn("",[ToolCall("x","read",{"source":"w"})]))
    out=AgentRuntime(model,[],max_steps=1,final_text_retry=True).run("What is 6*7?",tmp_path)
    assert not out.completed and out.stop_reason=="invalid_final"
