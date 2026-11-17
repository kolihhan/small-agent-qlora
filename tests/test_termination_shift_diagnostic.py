import json

import pytest

from scripts.analyze_termination_shift import _load, summarize, _mcnemar_exact


def _row(task, *, done=True, good=False, steps=3, calls=1, duplicates=0, stop=None, markup=False):
    reason = stop or ("final" if done else "invalid_final")
    trace = [{"kind": "invalid_final", "step": steps, "data": {
        "content": "<tool_call>secret-example</tool_call>" if markup else "unparsed"
    }}] if reason == "invalid_final" else []
    return {
        "task_id": task, "level": 1, "question": "PRIVATE DO NOT PRINT",
        "model_answer": "PRIVATE ANSWER", "completed": done, "correct": good,
        "steps": steps, "stop_reason": reason, "trace": trace,
        "tool_calls": calls, "duplicate_calls_blocked": duplicates, "tool_errors": 0,
    }


def test_paired_termination_diagnostic_cohorts_and_sanitizes():
    v3b = {
        "secret-a": _row("secret-a", done=True, good=True, steps=2),
        "secret-b": _row("secret-b", done=True, steps=6),
        "secret-c": _row("secret-c", done=False, steps=12),
        "secret-d": _row("secret-d", done=False, steps=12),
    }
    v3d = {
        "secret-a": _row("secret-a", done=False, steps=12, calls=11, duplicates=2, markup=True),
        "secret-b": _row("secret-b", done=True, good=True, steps=10),
        "secret-c": _row("secret-c", done=True, good=True, steps=10),
        "secret-d": _row("secret-d", done=False, steps=12),
    }
    result = summarize(v3b, v3d)
    assert result["n"] == 4
    assert result["v3b"]["completed"] == 2
    assert result["v3d"]["completed"] == 2
    assert result["paired_completion"] == {
        "both_yes": 1, "v3b_only": 1, "v3d_only": 1, "both_no": 1,
        "exact_mcnemar_p": 1.0,
    }
    assert result["paired_correctness"]["v3d_only"] == 2
    assert result["cohorts_by_v3b_steps"]["01-03"]["v3d_completed"] == 0
    assert result["cohorts_by_v3b_steps"]["04-06"]["v3d_completed"] == 1
    assert result["v3d"]["step12_invalid_final"] == 2
    assert result["v3d"]["textual_tool_markup_invalid_final"] == 1
    data = json.dumps(result)
    for sensitive in ("secret-", "PRIVATE", "tool_call>", "task_id", "model_answer", "question"):
        assert sensitive not in data


def test_exact_mcnemar_small_discordance():
    assert _mcnemar_exact(20, 1) == pytest.approx(2.09808349609375e-05)
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(2, 1) == 1.0


def test_mismatched_paired_task_identity_is_rejected():
    with pytest.raises(ValueError, match="paired task identities differ"):
        summarize({"a": _row("a")}, {"b": _row("b")})


def test_duplicate_or_missing_or_invalid_records_fail_closed(tmp_path):
    path = tmp_path / "data.jsonl"
    row = _row("sensitive")
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task"):
        _load(path, 2)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2"):
        _load(path, 2)
    row["correct"] = True
    row["completed"] = False
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be correct"):
        _load(path, 1)


def test_cli_produces_aggregate_only(tmp_path, monkeypatch):
    from scripts.analyze_termination_shift import main
    from sys import argv
    b = tmp_path / "b.jsonl"
    d = tmp_path / "d.jsonl"
    out = tmp_path / "aggregate.json"
    b.write_text(json.dumps(_row("protected-1", done=True, good=True)) + "\n", encoding="utf-8")
    d.write_text(json.dumps(_row("protected-1", done=False, steps=12, markup=True)) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["diag", "--v3b", str(b), "--v3d", str(d),
                                    "--expected", "1", "--output", str(out)])
    main()
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["n"] == 1
    assert "protected-1" not in out.read_text(encoding="utf-8")
    assert summary["paired_completion"]["v3b_only"] == 1
