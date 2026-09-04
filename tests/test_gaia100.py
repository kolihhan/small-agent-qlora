import json

import pytest

from gaia_small_agent.benchmark.gaia100 import select_gaia100, summarize_results


def test_gaia100_selection_is_deterministic_and_stratified():
    rows = []
    for level, count in [(1, 53), (2, 86), (3, 26)]:
        for i in range(count):
            rows.append({"task_id": f"L{level}-{i:03d}", "Level": level})
    a = select_gaia100(rows, seed="gaia100-v1")
    b = select_gaia100(list(reversed(rows)), seed="gaia100-v1")
    assert [r["task_id"] for r in a] == [r["task_id"] for r in b]
    assert len(a) == 100
    assert sum(int(r["Level"]) == 1 for r in a) == 32
    assert sum(int(r["Level"]) == 2 for r in a) == 52
    assert sum(int(r["Level"]) == 3 for r in a) == 16


def test_summary_reports_portfolio_metrics():
    rows = [
        {"level": 1, "correct": True, "completed": True, "tool_calls": 2, "tool_successes": 2, "steps": 3, "model_calls": 3, "latency_ms": 100},
        {"level": 2, "correct": False, "completed": False, "tool_calls": 1, "tool_successes": 0, "steps": 5, "model_calls": 5, "latency_ms": 300},
    ]
    s = summarize_results(rows)
    assert s["correct"] == 1
    assert s["total"] == 2
    assert s["accuracy"] == 0.5
    assert s["completion_rate"] == 0.5
    assert s["tool_success_rate"] == 2 / 3
    assert s["average_steps"] == 4.0
    assert s["model_calls"] == 8
    assert s["mean_latency_ms"] == 200.0


def test_resolve_attachment_path_uses_dataset_snapshot(tmp_path):
    from gaia_small_agent.benchmark.gaia100 import resolve_file_paths
    attachment = tmp_path / "2023" / "validation" / "note.txt"
    attachment.parent.mkdir(parents=True)
    attachment.write_text("x")
    rows = resolve_file_paths([{"task_id":"x", "Level":1, "file_path":"2023/validation/note.txt"}], tmp_path)
    assert rows[0]["file_path"] == str(attachment.resolve())


def _synthetic_gaia_rows():
    rows = []
    for level, count in [(1, 53), (2, 86), (3, 26)]:
        for i in range(count):
            task_id = f"L{level}-{i:03d}"
            rows.append({
                "task_id": task_id,
                "Level": level,
                "Question": f"Return ok for {task_id}",
                "Final answer": "ok",
                "file_path": "",
            })
    return rows


def _fake_runtime(seen_questions):
    from gaia_small_agent.agent.types import AgentResult, RunMetrics

    class FakeRuntime:
        def run(self, question, workspace):
            seen_questions.append(question)
            return AgentResult(
                answer="ok",
                completed=True,
                stop_reason="final",
                trace=[],
                metrics=RunMetrics(steps=1),
            )

    return FakeRuntime()


def _run_config(**overrides):
    config = {
        "backend": "ollama",
        "model": "synthetic-model",
        "adapter": None,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": False,
        "quantize_4bit": None,
        "seed": "gaia100-v1",
        "dataset_revision": "synthetic-revision",
        "tool_observation_max_chars": 2000,
    }
    config.update(overrides)
    return config


def test_runner_resumes_without_repeating_persisted_tasks(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    first_questions = []
    run_gaia100(rows, lambda: _fake_runtime(first_questions), tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())
    second_questions = []
    summary = run_gaia100(rows, lambda: _fake_runtime(second_questions), tmp_path, manifest_path=manifest_path, limit=3, run_config=_run_config())

    assert len(first_questions) == 2
    assert len(second_questions) == 1
    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({record["task_id"] for record in records}) == 3
    assert summary["total"] == 3


def test_runner_accepts_unicode_line_separators_inside_valid_json_strings(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    run_gaia100(rows, lambda: _fake_runtime([]), tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())
    results_path = tmp_path / "results.jsonl"
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    records[0]["model_answer"] = "first\u0085second\u2028third\u2029fourth"
    results_path.write_bytes(
        "\r\n".join(json.dumps(record, ensure_ascii=False) for record in records).encode("utf-8") + b"\r\n"
    )
    before = results_path.read_bytes()

    def forbidden_factory():
        raise AssertionError("runtime should not be created")

    run_gaia100(rows, forbidden_factory, tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())

    assert results_path.read_bytes() == before


def test_runner_reuses_one_runtime_for_all_missing_tasks(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    created = []
    seen_questions = []

    def factory():
        created.append(object())
        return _fake_runtime(seen_questions)

    run_gaia100(rows, factory, tmp_path, manifest_path=manifest_path, limit=3, run_config=_run_config())

    assert len(created) == 1
    assert len(seen_questions) == 3


def test_runner_does_not_create_runtime_when_selected_tasks_are_already_persisted(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    run_gaia100(rows, lambda: _fake_runtime([]), tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())

    def forbidden_factory():
        raise AssertionError("runtime should not be created")

    run_gaia100(rows, forbidden_factory, tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())


def test_runner_preserves_completed_checkpoint_when_reused_runtime_raises(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    calls = 0

    class Runtime:
        def run(self, question, workspace):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic runtime failure")
            return _fake_runtime([]).run(question, workspace)

    with pytest.raises(RuntimeError, match="synthetic runtime failure"):
        run_gaia100(rows, lambda: Runtime(), tmp_path, manifest_path=manifest_path, limit=3, run_config=_run_config())

    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_runner_checkpoints_capacity_failure_continues_and_rerun_skips_byte_identically(tmp_path):
    from gaia_small_agent.agent.loop import AgentRuntime
    from gaia_small_agent.agent.types import AssistantTurn, ModelCapacityError, ToolCall
    from gaia_small_agent.benchmark.runner import run_gaia100
    from gaia_small_agent.tools.base import Tool, ToolResult

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    factory_calls = []

    class Echo(Tool):
        name = "echo"
        description = "echo"
        schema = {"type": "object"}

        def run(self, arguments, workspace):
            return ToolResult(True, "before")

    class ScriptedModel:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return AssistantTurn("", [ToolCall("c1", "echo", {})])
            if self.calls == 2:
                raise ModelCapacityError
            return AssistantTurn("done", [])

    runtime = AgentRuntime(ScriptedModel(), [Echo()], max_steps=4)

    def factory():
        factory_calls.append(True)
        return runtime

    run_gaia100(rows, factory, tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())
    before = (tmp_path / "results.jsonl").read_bytes()
    assert len(before.splitlines()) == 2
    records = [json.loads(line) for line in before.splitlines()]
    assert records[1]["correct"] is False
    assert records[0]["completed"] is False
    assert records[0]["stop_reason"] == "model_capacity"
    assert [event["kind"] for event in records[0]["trace"]] == ["tool_call", "tool_result", "model_capacity"]
    assert records[0]["tool_successes"] == 1
    assert records[1]["completed"] is True
    assert records[1]["model_answer"] == "done"
    assert runtime.model.calls == 3
    assert len(factory_calls) == 1

    def forbidden_factory():
        raise AssertionError("runtime should not be created")

    run_gaia100(rows, forbidden_factory, tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())

    assert (tmp_path / "results.jsonl").read_bytes() == before


def test_runner_never_scores_incomplete_result_as_correct(tmp_path):
    from gaia_small_agent.agent.types import AgentResult, RunMetrics
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"

    class Runtime:
        def run(self, question, workspace):
            return AgentResult("ok", False, "model_capacity", [], RunMetrics(steps=1))

    summary = run_gaia100(rows, lambda: Runtime(), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())

    assert summary["correct"] == 0


def test_runner_rejects_manifest_mismatch_before_running(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    seen_questions = []
    run_gaia100(rows, lambda: _fake_runtime(seen_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())
    results_path = tmp_path / "results.jsonl"
    before = results_path.read_bytes()

    try:
        run_gaia100(
            rows,
            lambda: (_ for _ in ()).throw(AssertionError("runtime should not be created")),
            tmp_path,
            seed="different-seed",
            manifest_path=manifest_path,
            limit=1,
            dataset_revision="different-revision",
            run_config=_run_config(dataset_revision="different-revision"),
        )
    except ValueError as exc:
        assert str(exc).startswith("GAIA manifest mismatch")
    else:
        raise AssertionError("manifest mismatch was accepted")
    assert results_path.read_bytes() == before


def test_runner_recovers_truncated_final_results_line(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    first_questions = []
    run_gaia100(rows, lambda: _fake_runtime(first_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())
    results_path = tmp_path / "results.jsonl"
    results_path.write_bytes(results_path.read_bytes() + b'{"task_id":"partial"')

    second_questions = []
    summary = run_gaia100(rows, lambda: _fake_runtime(second_questions), tmp_path, manifest_path=manifest_path, limit=2, run_config=_run_config())

    assert len(second_questions) == 1
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(record["task_id"] != "partial" for record in records)
    assert summary["total"] == 2


def test_runner_rejects_malformed_nonfinal_results_line(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    seen_questions = []
    run_gaia100(rows, lambda: _fake_runtime(seen_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())
    results_path = tmp_path / "results.jsonl"
    valid_line = results_path.read_text(encoding="utf-8").splitlines()[0]
    results_path.write_text(f"{valid_line}\n{{malformed\n{valid_line}\n", encoding="utf-8")

    try:
        run_gaia100(rows, lambda: _fake_runtime(seen_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())
    except ValueError as exc:
        assert "results" in str(exc).lower()
    else:
        raise AssertionError("interior malformed results line was accepted")


def test_runner_rejects_base_adapter_config_mismatch_before_running(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    seen_questions = []
    base_config = _run_config()
    run_gaia100(rows, lambda: _fake_runtime(seen_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=base_config)
    manifest_before = manifest_path.read_bytes()
    results_path = tmp_path / "results.jsonl"
    results_before = results_path.read_bytes()
    config_before = (tmp_path / "run_config.json").read_bytes()

    with pytest.raises(ValueError, match="GAIA run configuration mismatch"):
        run_gaia100(
            rows,
            lambda: (_ for _ in ()).throw(AssertionError("runtime should not be created")),
            tmp_path,
            manifest_path=manifest_path,
            limit=1,
            run_config=_run_config(adapter="synthetic-adapter"),
        )
    assert manifest_path.read_bytes() == manifest_before
    assert results_path.read_bytes() == results_before
    assert (tmp_path / "run_config.json").read_bytes() == config_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "transformers"),
        ("model", "different-model"),
        ("max_steps", 99),
        ("max_new_tokens", 999),
        ("thinking", True),
        ("quantize_4bit", False),
        ("tool_observation_max_chars", 1999),
    ],
)
def test_runner_rejects_runtime_option_mismatch_before_running(tmp_path, field, value):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    seen_questions = []
    run_gaia100(rows, lambda: _fake_runtime(seen_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())
    manifest_before = manifest_path.read_bytes()
    results_path = tmp_path / "results.jsonl"
    results_before = results_path.read_bytes()
    config_before = (tmp_path / "run_config.json").read_bytes()
    changed_config = _run_config(**{field: value})

    with pytest.raises(ValueError, match="GAIA run configuration mismatch"):
        run_gaia100(
            rows,
            lambda: (_ for _ in ()).throw(AssertionError("runtime should not be created")),
            tmp_path,
            manifest_path=manifest_path,
            limit=1,
            run_config=changed_config,
        )
    assert manifest_path.read_bytes() == manifest_before
    assert results_path.read_bytes() == results_before
    assert (tmp_path / "run_config.json").read_bytes() == config_before


def test_runner_rejects_newline_terminated_malformed_final_line(tmp_path):
    from gaia_small_agent.benchmark.runner import run_gaia100

    rows = _synthetic_gaia_rows()
    manifest_path = tmp_path / "manifest.json"
    seen_questions = []
    run_gaia100(rows, lambda: _fake_runtime(seen_questions), tmp_path, manifest_path=manifest_path, limit=1, run_config=_run_config())
    results_path = tmp_path / "results.jsonl"
    valid_line = results_path.read_text(encoding="utf-8").splitlines()[0]
    results_path.write_text(f"{valid_line}\n{{malformed\n", encoding="utf-8")
    before = results_path.read_bytes()

    with pytest.raises(ValueError, match="results"):
        run_gaia100(
            rows,
            lambda: (_ for _ in ()).throw(AssertionError("runtime should not be created")),
            tmp_path,
            manifest_path=manifest_path,
            limit=1,
            run_config=_run_config(),
        )
    assert results_path.read_bytes() == before


def test_summary_reports_absolute_tool_and_duplicate_counts():
    from gaia_small_agent.benchmark.gaia100 import summarize_results

    rows = [
        {"level": 1, "correct": False, "completed": False, "tool_calls": 3, "tool_successes": 1, "tool_errors": 2, "duplicate_calls_blocked": 1, "steps": 4, "model_calls": 4, "latency_ms": 10.0, "capability_gap": None, "failure_label": "max_steps", "failure_signals": ["max_steps", "duplicate_block", "tool_error"]},
        {"level": 1, "correct": True, "completed": True, "tool_calls": 1, "tool_successes": 1, "tool_errors": 0, "duplicate_calls_blocked": 0, "steps": 2, "model_calls": 2, "latency_ms": 20.0, "capability_gap": None, "failure_label": "correct", "failure_signals": []},
    ]
    summary = summarize_results(rows)
    assert summary["tool_calls"] == 4
    assert summary["tool_successes"] == 2
    assert summary["tool_errors"] == 2
    assert summary["duplicate_calls_blocked"] == 1
    assert summary["model_calls"] == 6
