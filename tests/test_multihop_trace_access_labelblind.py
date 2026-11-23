"""Synthetic label-blind trace tests; never use training or benchmark gold."""
import json
from pathlib import Path

from scripts.diagnose_multihop_trace_access import summarize, original_rows


def _row(source1="p1.txt", source2="p2.txt"):
    return {
        "stop_reason":"final",
        "tool_calls":2,
        "trace":[
            {"kind":"tool_call","data":{"id":"a","name":"read","arguments":{"source":source1}}},
            {"kind":"tool_result","data":{"id":"a","name":"read","ok":True,
              "content":"Alice Chen and Leon Martin worked together."}},
            {"kind":"tool_call","data":{"id":"b","name":"read","arguments":{"source":source2}}},
            {"kind":"tool_result","data":{"id":"b","name":"read","ok":True,
              "content":"Leon Martin later moved to Vienna."}},
        ],
        "correct":False,
        "answer":"SECRET THAT MUST NOT BE EXPORTED",
        "expected_answer":"GOLD SECRET THAT MUST NOT BE EXPORTED",
    }


def test_diagnostic_only_exports_aggregate_counters():
    row=_row()
    result=summarize([row])
    assert result["n"]==1
    assert result["task_counts"]["two_distinct_successful_read_sources"]==1
    assert result["task_counts"]["two_sources_share_capitalized_name"]==1
    assert "SECRET" not in json.dumps(result)
    assert "expected_answer" not in json.dumps(result)


def test_lexical_link_cannot_assert_true_semantic_relation():
    row=_row(source1="p1.txt",source2="p1.txt")
    result=summarize([row])
    assert result["task_counts"].get("two_distinct_successful_read_sources",0)==0
    assert result["task_counts"].get("two_sources_share_capitalized_name",0)==0


def test_no_read_diagnostic():
    row={"trace":[{"kind":"tool_call","data":{"id":"x","name":"search","arguments":{"query":"x"}}},
                 {"kind":"tool_result","data":{"id":"x","name":"search","ok":False,"error_code":"SEARCH_ERROR"}}],
         "tool_calls":1,"stop_reason":"max_steps"}
    result=summarize([row])
    assert result["task_counts"]["no_successful_reads"]==1
    assert result["tool_error_codes"]["SEARCH_ERROR"]==1


def test_frozen_shard_loader_requires_all_20_and_exact_100(tmp_path):
    for i in range(20):
        p=tmp_path/f"gaia-v8-paired-shard-{i}"/"base"/f"shard-{i}"/"results.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text("".join(json.dumps(_row())+"\n" for _ in range(5)))
    rows=original_rows(tmp_path,"base")
    assert len(rows)==100
    assert "correct" not in rows[0]
    assert "answer" not in rows[0]
