import json
from collections import Counter
from pathlib import Path

import pytest


def _rows():
    rows = []
    for level, count in [(1, 53), (2, 86), (3, 26)]:
        for i in range(count):
            rows.append({
                "task_id": f"L{level}-{i:03d}",
                "Level": level,
                "Question": f"Question {level}-{i}",
                "Final answer": "ok",
                "file_path": "",
            })
    return rows


def test_local_partitions_are_deterministic_stratified_and_disjoint():
    from gaia_small_agent.benchmark.gaia100 import select_gaia_partition

    rows = _rows()
    diagnostic = select_gaia_partition(rows, "diagnostic", seed="local-v2")
    shadow = select_gaia_partition(list(reversed(rows)), "shadow", seed="local-v2")
    evaluation = select_gaia_partition(list(reversed(rows)), "evaluation", seed="local-v2")

    assert len(diagnostic) == 25
    assert [sum(int(r["Level"]) == level for r in diagnostic) for level in (1, 2, 3)] == [8, 13, 4]
    assert len(shadow) == 25
    assert Counter(int(r["Level"]) for r in shadow) == {1: 8, 2: 13, 3: 4}
    assert len(evaluation) == 100
    assert [sum(int(r["Level"]) == level for r in evaluation) for level in (1, 2, 3)] == [32, 52, 16]

    diagnostic_ids = {r["task_id"] for r in diagnostic}
    shadow_ids = {r["task_id"] for r in shadow}
    evaluation_ids = {r["task_id"] for r in evaluation}
    assert diagnostic_ids.isdisjoint(shadow_ids)
    assert diagnostic_ids.isdisjoint(evaluation_ids)
    assert shadow_ids.isdisjoint(evaluation_ids)

    assert [r["task_id"] for r in diagnostic] == [
        r["task_id"] for r in select_gaia_partition(rows, "diagnostic", seed="local-v2")
    ]
    assert [r["task_id"] for r in shadow] == [
        r["task_id"] for r in select_gaia_partition(rows, "shadow", seed="local-v2")
    ]
    assert [r["task_id"] for r in evaluation] == [
        r["task_id"] for r in select_gaia_partition(rows, "evaluation", seed="local-v2")
    ]


def test_shadow_partition_fails_when_unused_level_quota_is_insufficient():
    from gaia_small_agent.benchmark.gaia100 import select_gaia_partition

    rows = [row for row in _rows() if not (int(row["Level"]) == 3 and row["task_id"] in {"L3-022", "L3-023", "L3-024", "L3-025"})]

    with pytest.raises(ValueError, match="shadow"):
        select_gaia_partition(rows, "shadow", seed="local-v2")


def test_protected_question_hashes_reject_exact_gaia_content_even_with_safe_source(tmp_path):
    from gaia_small_agent.training.protection import build_protected_question_hashes
    from gaia_small_agent.training.qlora import load_verified_rows

    protected = tmp_path / "protected.json"
    build_protected_question_hashes([{"Question": "What is the capital of France?"}], protected)

    row = {
        "task_id": "safe-1",
        "verified": True,
        "source": "synthetic-safe-name",
        "expected_answer": "Paris",
        "required_tools": [],
        "provenance": {"license": "CC0-1.0", "generator": "unit", "generator_version": "1", "oracle_type": "exact", "oracle_version": "1"},
        "verification": {"completed": True, "final_answer_correct": True, "zero_tool_errors": True, "zero_duplicate_blocks": True, "required_tools_satisfied": True, "provenance_complete": True},
        "tools": [],
        "messages": [
            {"role": "user", "content": "  What is the capital of France?  "},
            {"role": "assistant", "content": "Paris"},
        ],
    }
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected GAIA question overlap"):
        load_verified_rows(data, protected_questions_path=protected)


def _write_run(root: Path, *, adapter, answers):
    root.mkdir(parents=True)
    config = {
        "backend": "transformers",
        "model": "Qwen/Qwen3.5-4B",
        "adapter": adapter,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": False,
        "quantize_4bit": True,
        "seed": "local-v2",
        "dataset_revision": "rev",
        "tool_observation_max_chars": 2000,
        "partition": "evaluation",
    }
    (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    records = []
    for idx, (task_id, correct, steps, calls, errors) in enumerate(answers, 1):
        records.append({
            "index": idx,
            "task_id": task_id,
            "level": 1,
            "model_answer": "x",
            "correct": correct,
            "completed": True,
            "stop_reason": "final",
            "steps": steps,
            "tool_calls": calls,
            "tool_successes": calls - errors,
            "tool_errors": errors,
            "duplicate_calls_blocked": 0,
            "capability_gap": None,
            "failure_label": "correct" if correct else "incorrect_after_tools",
            "trace": [],
        })
    (root / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_compare_runs_requires_adapter_as_only_treatment_and_reports_transitions(tmp_path):
    from gaia_small_agent.benchmark.compare import compare_runs

    base = tmp_path / "base"
    tuned = tmp_path / "tuned"
    unchanged = [(f"same-{index:03d}", False, 2, 1, 0) for index in range(98)]
    _write_run(base, adapter=None, answers=[("a", False, 4, 3, 1), ("b", True, 2, 1, 0), *unchanged])
    _write_run(tuned, adapter="adapter-dir", answers=[("a", True, 3, 2, 0), ("b", False, 4, 2, 1), *unchanged])

    result = compare_runs(base, tuned)

    assert result["wrong_to_correct"] == 1
    assert result["correct_to_wrong"] == 1
    assert result["net_correct_change"] == 0
    assert result["base"]["correct"] == 1
    assert result["adapter"]["correct"] == 1
    assert result["supported"]["wrong_to_correct"] == 1
    assert result["supported"]["correct_to_wrong"] == 1
    assert result["supported"]["net_correct_change"] == 0


def test_compare_runs_rejects_non_adapter_configuration_drift(tmp_path):
    from gaia_small_agent.benchmark.compare import compare_runs

    base = tmp_path / "base"
    tuned = tmp_path / "tuned"
    _write_run(base, adapter=None, answers=[("a", False, 4, 3, 1)])
    _write_run(tuned, adapter="adapter-dir", answers=[("a", True, 3, 2, 0)])
    config = json.loads((tuned / "run_config.json").read_text(encoding="utf-8"))
    config["max_steps"] = 99
    (tuned / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="only adapter may differ"):
        compare_runs(base, tuned)


def test_known_capability_gap_marks_semantic_media_only():
    from gaia_small_agent.benchmark.runner import known_capability_gap

    assert known_capability_gap("chart.png") == "semantic_image_unsupported"
    assert known_capability_gap("lecture.mp4") == "semantic_video_unsupported"
    assert known_capability_gap("audio.wav") == "semantic_audio_unsupported"
    assert known_capability_gap("report.pdf") is None
    assert known_capability_gap("table.xlsx") is None


def test_failure_label_distinguishes_capability_and_policy_proxy():
    from gaia_small_agent.agent.types import AgentResult, RunMetrics
    from gaia_small_agent.benchmark.runner import classify_failure_label

    assert classify_failure_label(
        AgentResult("wrong", True, "final", [], RunMetrics(steps=1, tool_calls=0)),
        correct=False,
        capability_gap=None,
    ) == "premature_final_proxy"
    assert classify_failure_label(
        AgentResult("wrong", True, "final", [], RunMetrics(steps=2, tool_calls=1, tool_errors=1)),
        correct=False,
        capability_gap=None,
    ) == "tool_error_present"
    assert classify_failure_label(
        AgentResult("wrong", True, "final", [], RunMetrics(steps=2, tool_calls=1)),
        correct=False,
        capability_gap="semantic_image_unsupported",
    ) == "capability_gap"


def test_summary_reports_supported_and_capability_gap_populations():
    from gaia_small_agent.benchmark.gaia100 import summarize_results

    rows = [
        {"level": 1, "correct": True, "completed": True, "tool_calls": 0, "tool_successes": 0, "steps": 1, "capability_gap": None, "failure_label": "correct"},
        {"level": 2, "correct": False, "completed": True, "tool_calls": 0, "tool_successes": 0, "steps": 1, "capability_gap": "semantic_image_unsupported", "failure_label": "capability_gap"},
    ]
    summary = summarize_results(rows)

    assert summary["supported"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert summary["capability_gap"] == {"correct": 0, "total": 1, "accuracy": 0.0}
    assert summary["failure_labels"] == {"capability_gap": 1, "correct": 1}


def test_collector_rejects_exact_protected_question_overlap(tmp_path):
    from gaia_small_agent.agent.types import AgentResult, RunMetrics, TraceEvent
    from gaia_small_agent.training.protection import build_protected_question_hashes
    from gaia_small_agent.training.trajectories import collect_verified_trajectories

    protected = tmp_path / "protected.json"
    build_protected_question_hashes([{"Question": "Protected benchmark question"}], protected)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({
        "id": "copy",
        "source": "synthetic",
        "license": "CC0-1.0",
        "generator": "unit",
        "generator_version": "1",
        "oracle_type": "exact",
        "oracle_version": "1",
        "required_tools": [],
        "question": "Protected benchmark question",
        "expected_answer": "yes",
    }) + "\n", encoding="utf-8")

    class Runtime:
        tools = {}
        def run(self, question, workspace):
            return AgentResult("yes", True, "final", [TraceEvent("final", 1, {"answer": "yes"})], RunMetrics(steps=1))

    output = tmp_path / "rows.jsonl"
    summary = collect_verified_trajectories(
        tasks,
        output,
        Runtime,
        tmp_path / "runs",
        protected_questions_path=protected,
    )

    assert summary["rejected_protected"] == 1
    assert output.read_text(encoding="utf-8") == ""


def test_failure_signals_preserve_overlapping_diagnostic_evidence():
    from gaia_small_agent.agent.types import AgentResult, RunMetrics
    from gaia_small_agent.benchmark.runner import collect_failure_signals

    result = AgentResult(
        "", False, "max_steps", [],
        RunMetrics(steps=12, tool_calls=10, tool_successes=7, tool_errors=3, duplicate_calls_blocked=2),
    )
    signals = collect_failure_signals(result, correct=False, capability_gap="semantic_image_unsupported")
    assert signals == ["max_steps", "duplicate_block", "tool_error", "capability_gap"]


def test_known_capability_gap_detects_extensionless_media_by_magic(tmp_path):
    from gaia_small_agent.benchmark.runner import known_capability_gap

    png = tmp_path / "image-attachment"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    wav = tmp_path / "audio-attachment"
    wav.write_bytes(b"RIFF" + (36).to_bytes(4, "little") + b"WAVE" + b"x" * 32)
    mp4 = tmp_path / "video-attachment"
    mp4.write_bytes((24).to_bytes(4, "big") + b"ftypisom" + b"x" * 32)

    assert known_capability_gap(png) == "semantic_image_unsupported"
    assert known_capability_gap(wav) == "semantic_audio_unsupported"
    assert known_capability_gap(mp4) == "semantic_video_unsupported"
