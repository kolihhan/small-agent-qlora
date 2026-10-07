import json
from pathlib import Path

import pytest


def _write_partial_run(root: Path, *, adapter: str | None) -> None:
    root.mkdir(parents=True)
    config = {
        "backend": "transformers",
        "model": "Qwen/Qwen3.5-4B",
        "adapter": adapter,
        "max_steps": 12,
        "max_new_tokens": 512,
        "thinking": False,
        "quantize_4bit": True,
        "seed": "gaia-local-v2",
        "dataset_revision": "rev",
        "tool_observation_max_chars": 2000,
        "partition": "evaluation",
        "cache_implementation": "default",
        "tool_names": ["search", "read", "inspect", "python"],
    }
    (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    row = {
        "index": 1,
        "task_id": "task-1",
        "level": 1,
        "model_answer": "x",
        "correct": False,
        "completed": True,
        "stop_reason": "final",
        "steps": 1,
        "tool_calls": 0,
        "tool_successes": 0,
        "tool_errors": 0,
        "duplicate_calls_blocked": 0,
        "capability_gap": None,
        "failure_label": "incorrect_after_tools",
        "trace": [],
    }
    (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_compare_runs_rejects_partial_frozen_evaluation(tmp_path):
    from gaia_small_agent.benchmark.compare import compare_runs

    base = tmp_path / "base"
    tuned = tmp_path / "tuned"
    _write_partial_run(base, adapter=None)
    _write_partial_run(tuned, adapter="adapter-dir")

    with pytest.raises(ValueError, match="100-task evaluation"):
        compare_runs(base, tuned)
