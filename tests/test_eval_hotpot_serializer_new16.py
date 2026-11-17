"""CPU-only prospective serializer, source, and aggregation contract tests."""
from __future__ import annotations

from scripts.eval_hotpot_serializer_new16 import (
    GATE, PROTOCOL, SERIALIZER_SHA, freeze, serialize, aggregate,
)


class FakeTurn:
    def __init__(self, content, calls=None):
        self.content=content
        self.tool_calls=calls or []


class FakeModel:
    def __init__(self, reply):
        self.reply=reply
        self.calls=[]

    def complete(self, messages, tools):
        self.calls.append((messages,tools))
        return FakeTurn(self.reply)


def test_question_aware_verbatim_answer_accepted_gold_blind():
    m=FakeModel("Canary Islands, Spain")
    reply=("Teide is located in Tenerife, Canary Islands, Spain. "
           "Garajonay is located in La Gomera, Canary Islands, Spain.")
    out,status=serialize("Where are those parks?",reply,m)
    assert out=="Canary Islands, Spain"
    assert status=="accepted_verbatim"
    assert m.calls[0][1]==[]
    assert "Canary Islands, Spain" in m.calls[0][0][1]["content"]
    assert "gold" not in m.calls[0][0][1]["content"].casefold()


def test_serializer_rejects_invented_text_and_multiline():
    for value in ("Adeline Virginia Woolf","unexpected\nvalue",""):
        r,s=serialize("Who?", "Virginia Woolf was born in 1882.", FakeModel(value))
        assert r=="Virginia Woolf was born in 1882."
        assert s.startswith("rejected")


def test_freeze_requires_exact_prior_artifact_sha():
    import pytest
    with pytest.raises(RuntimeError,match="immutable"):
        freeze([],{"n":8,"cases":[],"cases_sha256":"bad"})


def make_case(i,raw=False,new=False,kind="bridge"):
    return {
        "id":f"c{i}","type":kind,"completed":True,
        "raw":"unmodified","serialized":"candidate",
        "serialization_status":"accepted_verbatim",
        "raw_exact":raw,"lastline_exact":raw,
        "serialized_exact":new,"original_support_docs":1,
        "tool_calls":2,"tool_errors":0,"seconds":1.0,
    }


def test_aggregate_fail_closed_and_keeps_source_accuracy():
    digest="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    cases=[
        make_case(i,raw=(i>4),new=(i>1),
                  kind="bridge" if i%2==0 else "comparison")
        for i in range(16)
    ]
    parts=[{
        "status":"complete","protocol":PROTOCOL,
        "shard":i,"n":4,"source_sha":"frozen",
        "serializer_sha":SERIALIZER_SHA,"digest":digest,
        "cases":cases[i*4:(i+1)*4],
    } for i in range(4)]
    r=aggregate(parts)
    assert r["scores"]["raw_exact"]==11
    assert r["scores"]["serialized_exact"]==14
    assert r["rescues"]==3 and r["regressions"]==0
    assert r["positive_screen_only"] is True
    assert r["promotion_allowed"] is False
    assert r["preregistered_gate"]==GATE
    parts[3]["cases"][0]=parts[2]["cases"][0]
    import pytest
    with pytest.raises(RuntimeError,match="duplicate"):
        aggregate(parts)


def test_inference_does_not_use_benchmark_gold_in_serializer_source():
    import inspect
    from scripts.eval_hotpot_serializer_new16 import serialize, one_case
    source=inspect.getsource(serialize)
    assert "case['answer']" not in source
    assert "case[\"answer\"]" not in source
    assert 'tools=[]' in source
    assert "SERIALIZER_RULE" in source
