"""Safety and strict pairing contracts. No model inference in these tests."""
from hashlib import sha256
import pytest
from gaia_small_agent.agent.types import TraceEvent
from scripts import eval_hotpot16_final_salvage_v7 as m


def test_evidence_filter_only_successful_observations_and_bounded_last_ten():
    trace=[TraceEvent("tool_result",1,{"content":"bad","ok":False}),
           TraceEvent("tool_call",1,{"content":"irrelevant","ok":True})]
    trace.extend(TraceEvent("tool_result",i,{"ok":True,"content":f"source passage {i}"})
                 for i in range(2,15))
    obs=m._bounded_source_excerpts(trace)
    assert len(obs)==10 and obs[0]=="source passage 5"
    assert "bad" not in obs


def test_failure_recovery_rejects_unsourced_or_tool_call_or_long_answers():
    obs=["Read source: Alice Smith was born in 1955.", "Another excerpt."]
    assert m._proposal_accepted("Alice Smith",obs)
    assert m._proposal_accepted("alice smith",obs)
    for candidate in ["NO_ANSWER","Another name","<tool_call>{}</tool_call>",
                      "a"*97,"Alice\nSmith",""]:
        assert not m._proposal_accepted(candidate,obs)


def test_refuse_incomplete_prior_artifacts():
    with pytest.raises(ValueError):
        m.validate_prior({"cases":[]},{"cases":[]},{"cases":[]})


def test_fail_closed_incomplete_physical_result_set():
    with pytest.raises(ValueError):
        m.aggregate([])
    with pytest.raises(ValueError):
        m.aggregate([{}]*16)


def _part(mode,shard):
    cases=[]
    for j in range(2):
        i=shard*2+j
        family="bridge" if i<8 else "comparison"
        base={
            "correct":i<3,"calls":2,"elapsed_s":1.0,"completed":True,
            "stop_reason":"final","steps":2,"tool_calls":2,"tool_errors":0}
        salv={**base,"correct":i<3 or (i==9 and mode=="adapter"),
              "calls":base["calls"]+(i==9),"elapsed_s":1.1,
              "fallback_used":i==9,"fallback_accepted":i==9 and mode=="adapter",
              "fallback_seconds":0.1 if i==9 else 0,
              "source_excerpt_count":2 if i==9 else 0}
        cases.append({"task_sha256":sha256(str(i).encode()).hexdigest(),
                      "type":family,"arms":{
                         f"{mode}_frozen_formatter":base,
                         f"{mode}_failure_salvage":salv}})
    return {"status":"PHYSICAL_TWO_CASES_COMPLETE","protocol":m.PROTOCOL,
            "mode":mode,"shard":shard,"n":2,
            "source_sha":sha256(b"same").hexdigest(),
            "base_sha":m.BASE_SHA,"adapter_sha":m.V7_SHA,
            "serializer_sha":m.SERIALIZER_SHA,"salvage_policy_sha":m.POLICY_SHA,
            "cases":cases}


def test_paired_aggregate_guard_and_no_promotion():
    parts=[_part(mode,s) for mode in ("base","adapter") for s in range(8)]
    out=m.aggregate(parts)
    assert out["n"]==16 and out["shards"]==16
    assert out["totals"]["adapter_failure_salvage"]["correct"]==4
    assert out["comparisons"]["adapter"]["rescues"]==1
    assert out["comparisons"]["adapter"]["review_informative"] is False
    assert out["comparisons"]["adapter"]["gate_positive"] is False
    assert out["promotion_allowed"] is False
    with pytest.raises(ValueError):
        m.aggregate(parts[:-1])
    parts[0]["cases"][0]["task_sha256"]=parts[0]["cases"][1]["task_sha256"]
    with pytest.raises(ValueError):
        m.aggregate(parts)


def test_predeclared_unchanged_original_model_qa_setup():
    assert m.N==16 and m.SHARDS==8
    assert m.TYPES=={"bridge":8,"comparison":8}
    assert m.BASE_SHA.startswith("8c4cc24f")
    assert m.V7_SHA.startswith("475f5fb5")
    assert m.FAIL_REASONS=={"invalid_final","empty_final","max_steps"}
    assert m.GATE["min_net_rescues"]==2
    assert m.GATE["no_promotion_without_diverse_holdout"]
