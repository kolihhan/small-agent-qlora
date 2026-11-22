"""Synthetic privacy and completeness controls for retrospective GAIA trace audit."""
import json
from pathlib import Path
import pytest
from scripts.audit_v8_gaia100_paired_regressions import analyze, paired_test_pvalue

def row(task, correct, tools, stop="final"):
    trace = [{"kind": "tool_call", "step": i + 1, "data": {
        "name": name, "arguments": {"private": "DO_NOT_LEAK"}}}
        for i, name in enumerate(tools)]
    trace.append({"kind": "final", "step": len(tools) + 1, "data": {
        "answer": "SECRET_ANSWER_DO_NOT_LEAK"}})
    return {"task_id":task, "correct":correct, "tool_calls":len(tools),
            "tool_errors":0, "stop_reason":stop, "completed":True,
            "steps":len(tools)+1, "trace":trace,
            "failure_label":"correct" if correct else "incorrect_after_tools",
            "capability_gap":None, "duplicate_calls_blocked":0}

def put(root, arm, cases):
    path = root / "gaia-v8-paired-shard-0" / arm / "shard-0" / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in cases)+"\n", encoding="utf-8")

def test_paired_rescues_regressions_and_redaction(tmp_path):
    put(tmp_path, "base", [row("a", False, ["search"]), row("b", True, ["read"])])
    put(tmp_path, "qlora-v8", [row("a", True, ["search", "read"]), row("b", False, [])])
    report = analyze(tmp_path, expected_n=2, shard_count=1, assert_frozen=False)
    assert report["outcomes"]["rescue"] == 1
    assert report["outcomes"]["regression"] == 1
    assert report["mcnemar_two_sided_exact_p"] == 1
    serialized = json.dumps(report)
    assert "SECRET_ANSWER_DO_NOT_LEAK" not in serialized
    assert "DO_NOT_LEAK" not in serialized
    assert '"task_id"' not in serialized
    assert len(report["changed_15_cases"]) == 2

def test_must_be_complete(tmp_path):
    put(tmp_path, "base", [row("a", False, [])])
    put(tmp_path, "qlora-v8", [row("b", True, [])])
    with pytest.raises(ValueError, match="identity"):
        analyze(tmp_path, expected_n=1, shard_count=1, assert_frozen=False)

def test_exact_mcnemar():
    assert paired_test_pvalue(8, 7) == 1
    assert paired_test_pvalue(4, 0) == 0.125
