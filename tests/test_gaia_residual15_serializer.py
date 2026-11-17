"""CPU only: protect GAIA partitions, unchanged trajectories and scorers."""
from __future__ import annotations

import json
from hashlib import sha256

import pytest

from scripts.eval_gaia_residual15_serializer import (
    MODEL, BASE_SHA, V7_SHA, SERIALIZER_SHA, EXPECTED_SERIALIZER_SHA,
    GATE, SHARDS, PROTOCOL, freeze, aggregate, SameTrajectoryRuntime,
)

# The agent model string is not a different base from original GAIA v7.
def test_rule_is_hotpot_preregistered_and_base_is_v7():
    assert SERIALIZER_SHA==EXPECTED_SERIALIZER_SHA
    assert MODEL=='qwen3.5-9b'
    assert BASE_SHA.startswith("8c4cc24f54473f3")
    assert V7_SHA.startswith("475f5fb59a5533")
    assert GATE=={"net_rescues":2,"max_regressions":0,"min_levels":2}
    assert SHARDS==3


def test_gaia_source_split_excludes_all_three_original_partitions(monkeypatch):
    # Fake all 165, including same protected selector contract from underlying
    # GAIA partition implementation. Source identities must not cross arms.
    import scripts.eval_gaia_residual15_serializer as m
    rows=[{"task_id":str(i),"Level":1 if i<55 else 2 if i<130 else 3}
          for i in range(165)]
    labels={
        "diagnostic":rows[:25],
        "evaluation":rows[25:125],
        "shadow":rows[125:150],
    }
    monkeypatch.setattr(m,"select_gaia_partition",
                        lambda _,partition,seed: labels[partition])
    frozen=freeze(rows)
    assert frozen["n"]==15
    assert frozen["serializer_rule_sha"]==SERIALIZER_SHA
    assert frozen["preregistered_gate"]==GATE
    assert set(frozen["hashed_tasks"]).isdisjoint({
        sha256(str(i).encode()).hexdigest() for i in range(150)
    })


def row(i, before, after, lvl=1):
    return {
        "task_hash":sha256(f"task{i}".encode()).hexdigest(),
        "level":lvl,"original_correct":before,
        "serialized_correct":after,
        "original_completed":True,"same_trace_no_tool_change":True,
        "tool_calls":2,"tool_errors":0,"stop_reason":"final",
        "formatter_status":"accepted_verbatim","extra_formatter_attempted":True,
        "raw_hash":sha256(f"before{i}".encode()).hexdigest(),
        "serialized_hash":sha256(f"after{i}".encode()).hexdigest(),
    }


def test_aggregate_exact_paired_gate_and_zero_misleading_promotions():
    rs=[row(i,False,True,lvl=1 if i==0 else 2) if i<2
        else row(i,i%2==0,i%2==0,lvl=3) for i in range(15)]
    shards=[
        {
            "status":"completed","protocol":PROTOCOL,
            "source_hash":"frozen-original",
            "serializer_sha":SERIALIZER_SHA,
            "base_sha":BASE_SHA,"adapter_sha":V7_SHA,
            "shard":i,"n":5,"cases":rs[i*5:(i+1)*5],
        } for i in range(3)
    ]
    summary=aggregate(shards)
    assert summary["original_correct"]==7
    assert summary["serialized_correct"]==9
    assert summary["rescues"]==2
    assert summary["regressions"]==0
    assert summary["positive_screen_only"] is True
    assert summary["promotion_allowed"] is False
    rs[8]["task_hash"]=rs[4]["task_hash"]
    with pytest.raises(RuntimeError,match="duplicate"):
        aggregate(shards)


def test_forbid_leak_of_benchmark_answer_into_formatter_source():
    import inspect
    from scripts.eval_gaia_residual15_serializer import SameTrajectoryRuntime
    from scripts.eval_hotpot_serializer_new16 import serialize
    routine=inspect.getsource(SameTrajectoryRuntime.run)
    assert "Final answer" not in routine
    assert "gaia_score" not in routine
    assert "serialize(question,raw,self.formatter)" in routine
    serializer=inspect.getsource(serialize)
    assert "tools=[]" in serializer
